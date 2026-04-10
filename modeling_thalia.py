# -*- coding: utf-8 -*-
# modeling_thalia.py 
import torch
import os
import json
import sys
import torch.nn as nn
import torch.nn.functional as F
from bidirectional_exchange import BidirectionalExperienceExchange
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config, GPT2Tokenizer, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import logging
import math
import time
import random
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Full backward hook is firing when gradients"
)
warnings.filterwarnings("ignore", message="Some weights of Thalia were not initialized")
try:
    from config import ThaliaConfig
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from config import ThaliaConfig

logger = logging.getLogger(__name__)

# ===================================================================
# 🧠 TEMPORAL HEBBIAN PATTERN LAYER - МЕТА-ПЛАСТИЧНОСТЬ v5.4
# ===================================================================
class TemporalHebbLayer(nn.Module):
    """
    Параллельное концептуальное пространство для одного блока трансформера.  
    ЖИЗНЕННЫЙ ЦИКЛ СЛОТА v5.4:
    EMPTY → ACTIVE (при создании)
    ACTIVE (даже с utility=0.01) → остаётся ACTIVE пока есть свободные слоты
    ACTIVE → CANDIDATE только если n_init == num_slots И utility < threshold
    CANDIDATE → ACTIVE (если utility восстановилась)
    CANDIDATE → EMPTY только после grace period И при полной ёмкости
    v5.3: Слоты "спят" вместо смерти пока есть свободное место.
    """
    SLOT_EMPTY = 0
    SLOT_CANDIDATE = 1
    SLOT_ACTIVE = 2
  
    def __init__(
        self,
        config,
        num_slots: int = 64,
        layer_idx=None,
        total_layers: int = None,
        consolidation_interval: int = 50,
        consolidation_lr: float = 0.05,
        consolidate_in_eval: bool = False,
        candidate_threshold: float = 0.08,
        dead_threshold: float = 0.02,
        candidate_grace_steps: int = 2000,
        decay_interval_mult: int = 10,
        new_slot_sim_threshold: float = 0.85,
    ):
        super().__init__()
        self.dim = config.n_embd
        self.num_slots = num_slots
        self.layer_idx = layer_idx if layer_idx is not None else "?"
        self.total_layers = total_layers # None = неизвестно, определится позже
        self.consolidation_interval = consolidation_interval
        self.consolidation_lr = consolidation_lr
        self.consolidate_in_eval = consolidate_in_eval
        self.candidate_threshold = candidate_threshold
        self.dead_threshold = dead_threshold
        self.candidate_grace_steps = candidate_grace_steps
        self.decay_interval_mult = decay_interval_mult
        self.new_slot_sim_threshold = new_slot_sim_threshold
      
        # ── Паттерны ──────────────────────────────────────────────
        self.register_buffer("patterns", torch.zeros(num_slots, self.dim))
        self.register_buffer("patterns_norm_cache", torch.zeros(num_slots, self.dim))
        self.register_buffer("patterns_norm_dirty", torch.tensor(True))
      
        # ── Аккумуляторы для существующих слотов ──────────────────
        self.register_buffer("accum_sum", torch.zeros(num_slots, self.dim))
        self.register_buffer("accum_weight", torch.zeros(num_slots))
      
        # ── Аккумуляторы для новых слотов ─────────────────────────
        self.register_buffer("new_slot_accum_sum", torch.zeros(self.dim))
        self.register_buffer("new_slot_accum_weight", torch.tensor(0.0))
        self.register_buffer("new_slot_steps", torch.tensor(0, dtype=torch.long))
      
        # 🔥 АККУМУЛЯТОР ДЛЯ "НЕПОХОЖЕГО" (v5-FIX-H)
        self.register_buffer("new_slot_accum_sum_residual", torch.zeros(self.dim))
        self.register_buffer("new_slot_accum_weight_residual", torch.tensor(0.0))
        self.register_buffer("new_slot_steps_residual", torch.tensor(0, dtype=torch.long))
      
        # ── Статистика слотов ──────────────────────────────────────
        self.register_buffer("slot_age", torch.zeros(num_slots, dtype=torch.long))
        self.register_buffer("slot_last_used", torch.zeros(num_slots, dtype=torch.long))
        self.register_buffer("slot_usage_count", torch.zeros(num_slots))
        self.register_buffer("slot_utility", torch.zeros(num_slots))
      
        # ── Статус и счётчик кандидата ─────────────────────────────
        self.register_buffer("slot_status", torch.zeros(num_slots, dtype=torch.long))
        self.register_buffer("slot_candidate_since", torch.zeros(num_slots, dtype=torch.long))
      
        # ── STDP ───────────────────────────────────────────────────
        self.register_buffer("transition_matrix", torch.zeros(num_slots, num_slots))
        self.register_buffer("prev_best_idx", torch.zeros(4096, dtype=torch.long))
        self.register_buffer("prev_batch_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prev_idx_valid", torch.tensor(False))
        self.register_buffer("prev_write_gate", torch.zeros(4096))
      
        # ── Инициализированные слоты ───────────────────────────────
        self.register_buffer("initialized_slots", torch.tensor(0, dtype=torch.long))
      
        # ── Счётчики ───────────────────────────────────────────────
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_reads", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_accum_pending_snapshot", torch.tensor(0, dtype=torch.long))
      
        # ── Заморозка / версионирование ───────────────────────────
        self.register_buffer("patterns_frozen", torch.tensor(False))
        self.register_buffer("_patterns_version", torch.tensor(0, dtype=torch.long))
      
        # ── Статистика для логов ───────────────────────────────────
        self.register_buffer("_last_active_slots", torch.tensor(0, dtype=torch.long))
        self.register_buffer("new_slot_created_this_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_surprise_used", torch.tensor(0.0))
        self.register_buffer("_last_hebb_query", torch.zeros(self.dim))
      
        # ── Aux loss буферы ────────────────────────────────────────
        self.register_buffer("aux_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_aux_loss_counter", torch.tensor(0, dtype=torch.long))
        _cl = torch.tensor(0.0, dtype=torch.float32)
        _cl.add_(1e-12); _cl.sub_(1e-12)
        self.register_buffer("_controller_loss", _cl)
        self.register_buffer("_controller_loss_for_backward", torch.tensor(0.0, dtype=torch.float32))
      
        self._dynamic_tied_weights_keys = ["aux_loss", "*controller_loss"]
      
        # ── Счётчик доминирования ────────────────────────────
        self.register_buffer("_dominant_share_ema", torch.tensor(0.0))
        self._last_consolidation_step = 0
        self.register_buffer("_last_decay_step", torch.tensor(0, dtype=torch.long))
      
        # 🔥 FIX: Очищаем кэш пар при дефрагментации
        self._last_merged_pairs = set()
        self._merge_cooldown_step = 0
      
        # ========== ОБУЧАЕМЫЕ ЧАСТИ ==========
        self.query_norm_ctrl = nn.LayerNorm(self.dim)
        ctrl_dim = self.dim * 2 + 3
        self.controller = nn.Sequential(
            nn.Linear(ctrl_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        self.ctrl_norm = nn.LayerNorm(ctrl_dim)
        self.read_proj = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self.read_proj.weight)
        self.temperature_logit = nn.Parameter(torch.tensor(0.0))
        self.memory_scale = nn.Parameter(torch.tensor(0.1))
      
        # ========== ГИПЕРПАРАМЕТРЫ ==========
        self.merge_threshold = 0.95
        self.decay_factor = 0.9995
        self.decay_factor_new = 0.9999
        self.stdp_lr = 0.02
        self.accum_min_weight = 0.005
      
        # 🔴 v5.2: Потолок utility
        self.utility_ceiling = 0.95
      
        # 🔥 v5.2-ARCH-2: Минимум шагов для создания нового слота
        self.min_steps_for_new_slot = 3
      
        # 🆕 v5.3: Управление ёмкостью
        self.aggressive_merge_threshold = 0.88
        self.strengthen_sim_threshold = 0.92
        self.evict_only_if_novel_threshold = 0.75
      
        # v5.3.4: Приоритет "приткнуться"
        self.attach_threshold = 0.88 # если sim выше — всегда усиливаем
        self.merge_if_above = 0.78 # если между 0.78 и 0.88 — пытаемся слить
        self.evict_only_if = 0.68 # только тогда убиваем
      
        # 🆕 v5.3.2: ИММУНИТЕТ ПО АССОЦИАЦИЯМ (защита связанных слотов)
        self.register_buffer("_association_immunity", torch.zeros(num_slots))
        self.immunity_decay = 0.95
        self.immunity_boost = 0.1
        self.association_weight = nn.Parameter(torch.tensor(0.3))
      
        # v5.3.2: бонус от связей
        self.link_bonus_scale = 0.015 # основной множитель
        self.link_bonus_max = 0.08 # потолок бонуса за один consolidate
        self.link_bonus_decay = 0.992 # лёгкий decay старых связей при расчёте
      
        # v5.3.6: Долговременная память — порог перехода к "усилению"
        self.fill_threshold_for_creation = 0.90 # когда заполнено > 90% — перестаём активно создавать новые слоты
       
# ================================================================
# FORWARD
# ================================================================
    def forward(self, hidden_states, attention_mask=None, signals=None,
                        surprise=0.0, target_signals=None, **kwargs):
        if self._is_meta():
            return hidden_states, self.aux_loss
          
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            was_2d = True
        else:
            was_2d = False
          
        b, s, d = hidden_states.shape
        device = hidden_states.device
      
        self.step_counter.add_(1)
        self._aux_loss_counter.add_(1)
      
        # ── Распаковка signals ────────────────────────────────────
        if signals is None:
            signals = {}
        if surprise != 0.0 and 'surprise' not in signals:
            signals['surprise'] = surprise
        if target_signals is not None and 'target_signals' in signals:
            signals['target_signals'] = target_signals
           
        # Сохраняем заполненность группы (если нет, считаем по себе как фоллбэк)
        self._temp_group_fill = signals.get('group_fill_ratio',
                                           int(self.initialized_slots.item()) / self.num_slots)
        self._temp_group_id = signals.get('group_id', 0)
          
        if 'surprise' in signals:
            sv = signals['surprise']
            surprise_val = float(sv.mean().item() if isinstance(sv, torch.Tensor) else sv)
        else:
            surprise_val = float(surprise) if isinstance(surprise, (int, float)) else 0.0
      
        # ── Защита от NaN/Inf ─────────────────────────────────────
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            return (hidden_states.squeeze(1) if was_2d else hidden_states), self.aux_loss
      
        # ========== 1. QUERY (УЛУЧШЕННЫЙ ANTI-MUSH) ==========
        # ANTI-MUSH: На длинных текстах выбираем случайное "окно внимания"
        if s > 64 and self.training:
            window_size = 64
            start_idx = torch.randint(0, max(1, s - window_size), (1,), device=device).item()
            focal_states = hidden_states[:, start_idx : start_idx + window_size, :]
           
            if attention_mask is not None:
                mask_chunk = attention_mask[:, start_idx : start_idx + window_size].unsqueeze(-1).float()
                query_vec = (focal_states * mask_chunk).max(dim=1)[0]
            else:
                query_vec = focal_states.max(dim=1)[0]
            
            if query_vec.requires_grad:
                query_vec.register_hook(lambda grad: grad / window_size)
        else:
            if attention_mask is not None:
                if attention_mask.dim() == 2 and attention_mask.shape[1] != s and s == 1:
                    attention_mask = attention_mask[:, :1]
                mask = attention_mask.unsqueeze(-1).float()
                query_vec = (hidden_states * mask).max(dim=1)[0]
            else:
                query_vec = hidden_states.max(dim=1)[0]
            
            if query_vec.requires_grad and s > 1:
                query_vec.register_hook(lambda grad: grad / s)

        # Нормализация query
        query_vec = torch.nan_to_num(query_vec, 0.0)
        query_for_search = F.normalize(query_vec.detach(), dim=-1)
        query_for_ctrl = self.query_norm_ctrl(query_vec)
      
        # ========== 2. Нет инициализированных слотов ==========
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            if not self.patterns_frozen.item():
                with torch.no_grad():
                    seq_mean = hidden_states.detach().mean(dim=1)
                    self.new_slot_accum_sum.add_(seq_mean.sum(dim=0))
                    self.new_slot_accum_weight.add_(float(b))
                    self.new_slot_steps.add_(1)
                with torch.no_grad():
                    self.total_reads.add_(b * s)
                self._maybe_consolidate()
            return (hidden_states.squeeze(1) if was_2d else hidden_states), self.aux_loss
      
        # ========== 3. ARCH-5: ATTENTION ТОЛЬКО ПО ACTIVE СЛОТАМ ==========
        slot_active = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        n_active = int(slot_active.sum().item())
      
        if n_active == 0:
            with torch.no_grad():
                self.total_reads.add_(b * s)
            self._maybe_consolidate()
            return (hidden_states.squeeze(1) if was_2d else hidden_states), self.aux_loss
      
        patterns_norm = self._get_patterns_norm()[:n_init]
        similarity = torch.matmul(query_for_search, patterns_norm.T)
      
        NEG_INF = -30000.0
        if not slot_active.all():
            similarity = similarity.masked_fill(
                (~slot_active).unsqueeze(0), NEG_INF
            )
      
        # ===========================================================
        # 🔥 Динамическая температура с учётом surprise
        # ===========================================================
        temperature = self._get_temperature()
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, device=device)
        else:
            temperature = temperature.clone()
          
        if surprise_val != 0.0:
            temperature = temperature * (1.0 + surprise_val * 1.5)
        temperature = torch.clamp(temperature, 0.5, 8.0)
      
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=10.0, neginf=-10.0)
        scaled_sim = similarity / temperature.clamp(min=1e-8)
        attn_weights = F.softmax(scaled_sim, dim=-1)
      
        active_patterns = self.patterns[:n_init].detach()
        combined = torch.matmul(attn_weights, active_patterns)
      
        # ========== 4. TOP-1 для статистики ==========
        with torch.no_grad():
            valid_sim = similarity.clone()
            valid_sim[valid_sim < -1000] = -1.0
            best_sim, best_idx = valid_sim.max(dim=-1)
            novelty = (1.0 - best_sim).clamp(0.0, 1.0)
          
            if surprise_val != 0.0:
                novelty = novelty * 0.7 + surprise_val * 0.3
          
            dominant_idx = best_idx.mode().values.item()
            dominant_share = float((best_idx == dominant_idx).float().mean().item())
            self._dominant_share_ema.mul_(0.95).add_(dominant_share * 0.05)
      
        # ========== 5. CONTROLLER ==========
        max_usage = self.slot_usage_count[:n_init].max().clamp(min=1.0)
        usage_norm = (self.slot_usage_count[best_idx] / max_usage).detach()
      
        ctrl_input = torch.cat([
            query_for_ctrl,
            self.patterns[best_idx].detach(),
            best_sim.unsqueeze(-1).detach(),
            usage_norm.unsqueeze(-1).detach(),
            novelty.unsqueeze(-1).detach(),
        ], dim=-1)
      
        ctrl_input = self.ctrl_norm(ctrl_input).clamp(-5.0, 5.0)
        ctrl_out = self.controller(ctrl_input)
        read_gate = torch.sigmoid(ctrl_out[:, 0:1])
        write_gate = torch.sigmoid(ctrl_out[:, 1:2])
      
        if 'target_signals' in signals and signals['target_signals'] is not None:
            tgt = signals['target_signals']
            if isinstance(tgt, dict) and 'write_gate' in tgt:
                tw = torch.tensor(tgt['write_gate'], device=device).expand_as(write_gate)
                write_gate = write_gate * 0.9 + tw * 0.1
      
        # ========== 6. ЧТЕНИЕ ==========
        memory = self.read_proj(combined)
        
        # ===========================================================
        # 🔥 Динамический масштаб памяти с учётом surprise
        # ===========================================================
        base_scale = self.memory_scale.clamp(0.0, 0.5)
        scale = torch.clamp(base_scale * (1.0 + surprise_val), 0.0, 0.7)
        
        output = hidden_states + read_gate.unsqueeze(1) * memory.unsqueeze(1) * scale
      
        # ========== 7. НАКОПЛЕНИЕ (ИСПРАВЛЕНО: используем query_vec вместо hidden_states) ==========
        if not self.patterns_frozen.item():
            # 7a. Передаем острый query_vec вместо размазанного hidden_states
            self._accumulate_weighted(
                query_vec.detach(), # <--- ИЗМЕНЕНИЕ: используем query_vec
                attn_weights=attn_weights.detach(),
                write_gate=write_gate.detach(),
                n_init=n_init,
            )
          
            # 7b. Передаем острый query_vec
            self._accumulate_new_slot(
                query_vec.detach(), # <--- ИЗМЕНЕНИЕ: используем query_vec
                novelty,
                write_gate.detach(),
                patterns_norm if n_init > 0 else None
            )
          
            # Фиксация n_init для снапшота
            with torch.no_grad():
                n_init_fixed = int(self.initialized_slots.item())
                pending = int((self.accum_weight[:n_init_fixed] > 0).sum().item())
                new_pend_main = 1 if self.new_slot_accum_weight.item() > 0 else 0
                new_pend_res = 1 if self.new_slot_accum_weight_residual.item() > 0 else 0
                self._accum_pending_snapshot.fill_(pending + new_pend_main + new_pend_res)
      
        # ========== 8. СТАТИСТИКА + STDP ==========
        with torch.no_grad():
            self.slot_usage_count[best_idx] += 1.0
            self.slot_last_used[best_idx] = self.step_counter
            self.slot_age[best_idx] += 1
            self.total_reads.add_(b * s)
          
            # ARCH-3: Utility растёт от write_gate (важность)
            for bi in range(b):
                idx = best_idx[bi].item()
                st = int(self.slot_status[idx])
                wg = float(write_gate[bi].item())
              
                if st == self.SLOT_ACTIVE:
                    self.slot_utility[idx] = min(
                        self.utility_ceiling,
                        float(self.slot_utility[idx]) + wg * 0.02
                    )
                elif st == self.SLOT_CANDIDATE:
                    new_util = float(self.slot_utility[idx]) + wg * 0.1
                    self.slot_utility[idx] = min(self.utility_ceiling, new_util)
                    if self.slot_utility[idx] >= self.candidate_threshold:
                        self.slot_status[idx] = self.SLOT_ACTIVE
          
            # ARCH-4: STDP только для ACTIVE→ACTIVE
            if not self.patterns_frozen.item():
                self._update_stdp(best_idx, write_gate)
          
            active = (self.slot_status[:n_init] == self.SLOT_ACTIVE).sum()
            self._last_active_slots.fill_(active.item())
            self._last_hebb_query = query_for_search.mean(dim=0)
            self.last_surprise_used = torch.tensor(surprise_val, device=device)
      
        # ========== 9. AUX LOSS ==========
        if self.training:
            attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
            entropy_loss = attn_entropy * 0.01
            rg_mean = read_gate.mean()
            wg_mean = write_gate.mean()
            gate_reg = ((rg_mean - 0.5)**2 + (wg_mean - 0.5)**2) * 0.1
            total_aux = torch.clamp(entropy_loss + gate_reg, 0.0, 5.0)
            
            # Регистрируем aux_loss в градиентном графе
            if total_aux.requires_grad:
                self.aux_loss = total_aux + 0.0
            else:
                self.aux_loss = total_aux
        else:
            self.aux_loss = torch.tensor(0.0, device=device, requires_grad=False)
      
        # ========== 10. КОНСОЛИДАЦИЯ ==========
        self._maybe_consolidate()
      
        return (output.squeeze(1) if was_2d else output), self.aux_loss
       
# ================================================================
# КОНСОЛИДАЦИЯ
# ================================================================
    def _maybe_consolidate(self):
        if self.patterns_frozen.item():
            return
          
        step = int(self.step_counter.item())
      
        if not hasattr(self, '_last_consolidation_step'):
            self._last_consolidation_step = 0
        if not hasattr(self, '_last_decay_step'):
            self.register_buffer("_last_decay_step", torch.tensor(0, dtype=torch.long))
      
        # Консолидация
        if step - self._last_consolidation_step >= self.consolidation_interval:
            with torch.no_grad():
                self._consolidate()
                self._consolidate_new_slot()
                #self._merge_similar()
            self._last_consolidation_step = step
      
        # Decay
        decay_interval = self.consolidation_interval * self.decay_interval_mult
        steps_since_decay = step - self._last_decay_step.item()
      
        if steps_since_decay >= decay_interval:
            times = steps_since_decay // decay_interval
            with torch.no_grad():
                for _ in range(times):
                    self._decay_unused()
                    self._mark_candidates()
                    self._replace_expired()
            self._last_decay_step.add_(times * decay_interval)
       
# ================================================================
# ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# ================================================================
    @torch.no_grad()
    def _accumulate_weighted(self, query_vec, attn_weights, write_gate, n_init):
        """🔥 ARCH-1: Накопление ТОЛЬКО для ACTIVE слотов"""
        if n_init == 0:
            return
      
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        if not active_mask.any():
            return
      
        weights = attn_weights * write_gate
        weights = weights * active_mask.unsqueeze(0).float()
      
        # query_vec уже плоский [b, d], умножаем напрямую
        contrib = torch.matmul(weights.T, query_vec)
        w_sum = weights.sum(dim=0)
       
        self.accum_sum[:n_init].add_(contrib)
        self.accum_weight[:n_init].add_(w_sum)
  
    @torch.no_grad()
    def _accumulate_new_slot(self, query_vec, novelty, write_gate, patterns_norm):
        """
        Накопление для новых слотов с использованием плоского query_vec.
       
        Args:
            query_vec: [batch, dim] - уже сконденсированное представление (результат anti-mush)
            novelty: [batch] - степень новизны для каждого элемента батча
            write_gate: [batch, 1] - гейт записи
            patterns_norm: [n_init, dim] - нормализованные существующие паттерны
        """
        b, d = query_vec.shape # Убрали 's' (seq_len), вектор уже плоский
       
        # ===== БАЗОВЫЙ ПОРОГ ОТ ЗАПОЛНЕННОСТИ =====
        filled_ratio = self.initialized_slots.item() / self.num_slots
        if filled_ratio < 0.1:
            base_threshold = 0.10
        elif filled_ratio < 0.3:
            base_threshold = 0.15
        elif filled_ratio < 0.6:
            base_threshold = 0.18
        else:
            base_threshold = 0.20
       
        # ===== МОДИФИКАТОР ОТ ГЛУБИНЫ СЛОЯ =====
        depth_mod = 1.0
        if self.layer_idx is not None and self.layer_idx != "?" and self.total_layers:
            layer_id = int(self.layer_idx)
            n_layers = self.total_layers if self.total_layers else (layer_id + 1)
            depth_ratio = layer_id / max(n_layers - 1, 1) # 0.0 (нижний) → 1.0 (верхний)
           
            # Нижние слои (синтаксис) - порог чуть ниже (легче создавать)
            # Верхние слои (абстракции) - порог чуть выше (строже)
            depth_mod = 0.9 + depth_ratio * 0.2 # от 0.9 до 1.1
       
        # ===== КОМБИНИРОВАННЫЙ ПОРОГ =====
        novelty_threshold = base_threshold * depth_mod
       
        # Дополнительная модуляция от write_gate (важность)
        if write_gate is not None and write_gate.numel() > 0:
            wg_mean = write_gate.mean().item()
            if wg_mean > 0.3:
                novelty_threshold *= (1.0 - wg_mean * 0.3) # Высокий write_gate = легче создавать
       
        # Дополнительная модуляция от surprise (если высокое удивление - легче создавать)
        if hasattr(self, 'last_surprise_used') and self.last_surprise_used.item() > 0.3:
            surprise_mod = max(0.7, 1.0 - self.last_surprise_used.item() * 0.3)
            novelty_threshold *= surprise_mod
       
        # Финальное ограничение
        novelty_threshold = max(0.05, min(0.25, novelty_threshold))
       
        # ===== ОПРЕДЕЛЯЕМ КАНДИДАТОВ =====
        is_candidate = novelty >= novelty_threshold
       
        if not is_candidate.any():
            return
           
        n_init = self.initialized_slots.item()
       
        # Берем кандидатов прямо из плоского query_vec
        cand_states = query_vec[is_candidate] # shape: [num_candidates, d]
        cand_norm = F.normalize(cand_states, dim=-1)
       
        if n_init > 0 and patterns_norm is not None:
            sim_to_all = torch.matmul(cand_norm, patterns_norm.T)
            max_sim, closest_slot = sim_to_all.max(dim=-1)
           
            similar_mask = max_sim > self.new_slot_sim_threshold
            novel_mask = ~similar_mask
           
            if similar_mask.any():
                similar_states = cand_states[similar_mask]
                n_tok_sim = similar_mask.sum().item() # Убрали умножение на s
               
                for idx in torch.unique(closest_slot[similar_mask]):
                    slot_mask = (closest_slot == idx) & similar_mask
                    if slot_mask.any():
                        slot_states = cand_states[slot_mask]
                        # sum вместо mean, так как у нас уже плоские векторы
                        self.accum_sum[idx].add_(slot_states.sum(dim=0))
                        self.accum_weight[idx].add_(float(slot_mask.sum().item()))
           
            if novel_mask.any():
                novel_states = cand_states[novel_mask]
                n_tok_novel = novel_mask.sum().item() # Убрали умножение на s
               
                # residual — для по-настоящему новых
                self.new_slot_accum_sum_residual.add_(novel_states.sum(dim=0))
                self.new_slot_accum_weight_residual.add_(float(n_tok_novel))
                self.new_slot_steps_residual.add_(1)
               
                # main — для всего novel
                self.new_slot_accum_sum.add_(novel_states.sum(dim=0))
                self.new_slot_accum_weight.add_(float(n_tok_novel))
                self.new_slot_steps.add_(1)
        else:
            # Нет существующих слотов - все кандидаты идут в новые
            self.new_slot_accum_sum.add_(cand_states.sum(dim=0))
            self.new_slot_accum_weight.add_(float(cand_states.shape[0]))
            self.new_slot_steps.add_(1)
  
    @torch.no_grad()
    def _consolidate(self):
        """🆕 v5.3: Выравнивание utility всегда + diversity bonus"""
        if torch.is_grad_enabled():
            logger.warning(f"⚠ HebbLayer[{self.layer_idx}]: _consolidate с градиентами — пропуск")
            return
          
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            self.accum_sum.zero_()
            self.accum_weight.zero_()
            return
      
        # === v5.3: ВЫРАВНИВАНИЕ UTILITY ВСЕГДА ===
        self._entropy_align_utilities()
      
        active_mask = self.accum_weight[:n_init] > self.accum_min_weight
        if not active_mask.any():
            self.accum_sum.zero_()
            self.accum_weight.zero_()
            return
      
        total_usage = self.slot_usage_count[:n_init].sum().clamp(min=1.0)
        usage_share = self.slot_usage_count[:n_init] / total_usage
      
        updated = False
        for i in active_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            w = float(self.accum_weight[i])
            if w < self.accum_min_weight:
                continue
            new_emb = self.accum_sum[i] / w
            if new_emb.norm() < 1e-6:
                continue
              
            new_emb_n = F.normalize(new_emb, dim=-1)
            importance = float(self.slot_utility[i])
          
            diversity_penalty = float(usage_share[i]) * 0.5
            lr = self.consolidation_lr * (0.5 + 0.5 * importance) * (1.0 - diversity_penalty)
            lr = max(lr, 0.005)
          
            updated_pat = (1.0 - lr) * self.patterns[i] + lr * new_emb_n
            self.patterns.data[i] = F.normalize(updated_pat, dim=-1)
          
            self.slot_utility[i] = min(self.utility_ceiling, float(self.slot_utility[i]) + 0.1)
          
            if self.slot_status[i] == self.SLOT_CANDIDATE \
               and self.slot_utility[i] >= self.candidate_threshold:
                self.slot_status[i] = self.SLOT_ACTIVE
          
            self.total_updates.add_(1)
            updated = True
      
        # === v5.3: DIVERSITY BONUS для редких слотов ===
        if n_init > 3:
            utils = self.slot_utility[:n_init]
            usage = self.slot_usage_count[:n_init]
            rare_mask = (usage < usage.mean() * 0.3) & (utils > 0.05)
          
            if rare_mask.any():
                # Правильный способ: создаём полноразмерный бонус
                bonus = 0.03 * (1.0 - utils[rare_mask])
                bonus_full = torch.zeros_like(utils)
                bonus_full[rare_mask] = bonus
              
                self.slot_utility[:n_init] = torch.clamp(
                    utils + bonus_full,
                    max=self.utility_ceiling
                )
      
        if updated:
            self._patterns_version.add_(1)
          
        # === v5.3.2: бонус от связей ===
        self._apply_link_based_utility_bonus()
        self.accum_sum.zero_()
        self.accum_weight.zero_()
        self.patterns_norm_dirty.fill_(True)
        # === v5.3.2: ИММУНИТЕТ ПО АССОЦИАЦИЯМ ===
        self._update_association_immunity()
      
    @torch.no_grad()
    def _entropy_align_utilities(self):
        """v5.3.4: Мягкое выравнивание (не убиваем utility)"""
        n_init = int(self.initialized_slots.item())
        if n_init <= 1:
            return
        utils = self.slot_utility[:n_init].clone()
        mean_u = float(utils.mean())
      
        # Сильно слабее, чем было
        new_utils = utils * 0.94 + mean_u * 0.06
      
        self.slot_utility[:n_init] = new_utils
  
    @torch.no_grad()
    def _consolidate_new_slot(self):
        """🔥 v5.4 FIX: Исправлен безусловный return после residual"""
        if torch.is_grad_enabled():
            return
       
        n_init = int(self.initialized_slots.item())
        created_anything = False
       
        # Residual (самые новые)
        w_res = float(self.new_slot_accum_weight_residual)
        steps_res = int(self.new_slot_steps_residual)
        if w_res >= self.accum_min_weight and steps_res >= self.min_steps_for_new_slot:
            new_emb = self.new_slot_accum_sum_residual / w_res
            if new_emb.norm() > 1e-6:
                new_norm = F.normalize(new_emb, dim=-1)
                success = self._try_create_or_strengthen(new_norm, is_residual=True)
                if success:
                    self._clear_residual_accumulator()
                    created_anything = True
                    return  # ✅ При успехе выходим, иначе даём шанс main-аккумулятору
                # ❌ Если success=False, продолжаем к main аккумулятору!
       
        # Main аккумулятор
        w_main = float(self.new_slot_accum_weight)
        steps_main = int(self.new_slot_steps)
        if w_main >= self.accum_min_weight * 2 and steps_main >= self.min_steps_for_new_slot:
            new_emb = self.new_slot_accum_sum / w_main
            if new_emb.norm() > 1e-6:
                new_norm = F.normalize(new_emb, dim=-1)
                success = self._try_create_or_strengthen(new_norm, is_residual=False)
                if success:
                    self._clear_main_accumulator()
                    created_anything = True
       
        # Очищаем ТОЛЬКО если прошли лимит шагов (старые данные сбрасываем)
        max_accum_steps = self.consolidation_interval * 20
        if int(self.new_slot_steps) > max_accum_steps:
            self._clear_main_accumulator()
        if int(self.new_slot_steps_residual) > max_accum_steps:
            self._clear_residual_accumulator()
  
    @torch.no_grad()
    def _try_create_or_strengthen(self, new_norm: torch.Tensor, is_residual: bool) -> bool:
        n_init = int(self.initialized_slots.item())
        has_space = n_init < self.num_slots
        group_fill_ratio = getattr(self, '_temp_group_fill', n_init / self.num_slots)

        # 🔥 Явный приоритет: физическое место важнее группового fill_ratio
        if has_space:
            logger.debug(f"🟢 HebbLayer[{self.layer_idx}]: есть место ({n_init}/{self.num_slots}), создаём/усиливаем")
            
            # проверка на дубликат (даже если есть место)
            if n_init > 0:
                active_pnorms = self._get_patterns_norm()[:n_init]
                sim = float(torch.matmul(new_norm.unsqueeze(0), active_pnorms.T).max().item())
                # 🔥 ИСПРАВЛЕНИЕ: используем self.new_slot_sim_threshold вместо хардкода
                if sim > self.new_slot_sim_threshold:
                    # усиливаем ближайший
                    self._strengthen_slot(int(torch.argmax(torch.matmul(new_norm.unsqueeze(0), active_pnorms.T)).item()), new_norm)
                    return True
            
            self._create_slot(new_norm)
            return True

        # ── Места нет → только стабилизация ──
        logger.debug(f"🔴 HebbLayer[{self.layer_idx}]: слоты заполнены ({n_init}/{self.num_slots}), режим стабилизации")
        
        active_idxs = (self.slot_status[:n_init] == self.SLOT_ACTIVE).nonzero(as_tuple=False).squeeze(-1)
        if len(active_idxs) == 0:
            return False

        active_pnorms = self._get_patterns_norm()[active_idxs]
        sims = torch.matmul(new_norm.unsqueeze(0), active_pnorms.T).squeeze(0)
        max_sim = float(sims.max().item())

        # 🔥 ИСПРАВЛЕНИЕ: используем self.strengthen_threshold вместо хардкода 0.85
        strengthen_threshold = getattr(self, 'strengthen_threshold', 0.85)
        if max_sim > strengthen_threshold:
            closest = int(active_idxs[sims.argmax().item()].item())
            self._strengthen_slot(closest, new_norm)
            return True

        # 🔥 ИСПРАВЛЕНИЕ: используем self.merge_threshold вместо хардкода 0.75
        merge_threshold = getattr(self, 'merge_threshold', 0.75)
        if max_sim > merge_threshold:
            self._merge_similar_aggressive()
            return False

        # уникальный, но места нет → откладываем (не убиваем!)
        return False
      
    @torch.no_grad()
    def _merge_similar_aggressive(self):
        """🆕 v5.3: Агрессивное слияние при попытке добавить слот"""
        orig = self.merge_threshold
        self.merge_threshold = self.aggressive_merge_threshold
        self._merge_similar()
        self.merge_threshold = orig
  
    @torch.no_grad()
    def _strengthen_slot(self, idx: int, new_norm: torch.Tensor):
        """🆕 v5.3: Усиливаем существующий слот с защитой от переусиления"""
        # Если utility уже на потолке — не усиливаем (защита)
        if float(self.slot_utility[idx]) >= self.utility_ceiling - 0.01:
            return
          
        lr = 0.15 * (1.0 - float(self.slot_utility[idx]))
        updated = (1.0 - lr) * self.patterns[idx] + lr * new_norm
        self.patterns.data[idx] = F.normalize(updated, dim=-1)
        self.slot_utility[idx] = min(self.utility_ceiling, float(self.slot_utility[idx]) + 0.12)
        self._patterns_version.add_(1)
        self.patterns_norm_dirty.fill_(True)
  
    @torch.no_grad()
    def _evict_weakest(self, new_norm: torch.Tensor):
        """
        v5.3.5: Принудительный эвикт слабейшего слота при полной ёмкости
        и отсутствии CANDIDATE с истёкшим grace.
        Защищает: слоты с высоким иммунитетом, молодые слоты (age < 1000),
        слоты с высокой utility.
        """
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            return
        best_victim = -1
        best_score = float('inf') # меньше = хуже слот
        for i in range(n_init):
            if int(self.slot_status[i]) != self.SLOT_ACTIVE:
                continue
            u = float(self.slot_utility[i])
            immunity = float(self._association_immunity[i])
            age = int(self.slot_age[i])
            # Молодые слоты не трогаем
            if age < 1000:
                continue
            # Сильно защищённые не трогаем
            if immunity > 0.5:
                continue
            # Чем ниже utility и чем меньше иммунитет — тем хуже слот
            score = u + immunity * 0.5
            if score < best_score:
                best_score = score
                best_victim = i
        if best_victim == -1:
            # Совсем некого трогать — слияние как последний шанс
            self._merge_similar_aggressive()
            return
        # Помечаем жертву CANDIDATE и сразу даём истёкший grace
        # (чтобы _replace_expired её забрал на следующем шаге)
        self.slot_status[best_victim] = self.SLOT_CANDIDATE
        # Ставим время в прошлое — grace уже истёк
        self.slot_candidate_since[best_victim] = max(
            0,
            int(self.step_counter.item()) - self.candidate_grace_steps - 1
        )
        logger.info(
            f"💀 HebbLayer[{self.layer_idx}]: слот {best_victim} помечен на эвикт "
            f"(utility={best_score:.3f})"
        )
        # Немедленный эвикт + создание нового слота
        self._replace_expired()
        n_after = int(self.initialized_slots.item())
        if n_after < self.num_slots:
            self._create_slot(new_norm)
  
    @torch.no_grad()
    def _apply_link_based_utility_bonus(self):
        """v5.3.2: Связи между слотами повышают utility связанных концептов"""
        n_init = int(self.initialized_slots.item())
        if n_init < 4:
            return # слишком мало слотов — пропускаем
      
        tm = self.transition_matrix[:n_init, :n_init].clone()
      
        # Лёгкий decay старых переходов (чтобы свежие связи были важнее)
        steps_since = int(self.step_counter.item() - self._last_consolidation_step)
        decay_exp = min(steps_since, 50) # защита от взрыва
        tm.mul_(self.link_bonus_decay ** decay_exp)
      
        # Исходящие + входящие связи (степень вершины в направленном графе)
        out_degree = tm.sum(dim=1) # сколько раз этот слот предшествовал другим
        in_degree = tm.sum(dim=0) # сколько раз другие предшествовали этому
        total_links = out_degree + in_degree
      
        # Нормализация (чтобы бонус был в разумных пределах)
        if total_links.sum() > 1e-6:
            norm_links = total_links / total_links.max().clamp(min=1e-6)
        else:
            norm_links = torch.zeros_like(total_links)
      
        # Бонус пропорционален связности
        bonus = norm_links * self.link_bonus_scale
        bonus = bonus.clamp(max=self.link_bonus_max)
      
        # Применяем только к ACTIVE слотам
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        bonus_full = torch.zeros_like(self.slot_utility[:n_init])
        bonus_full[active_mask] = bonus[active_mask] # ← если bonus тоже sliced
      
        self.slot_utility[:n_init] = torch.clamp(
            self.slot_utility[:n_init] + bonus_full,
            max=self.utility_ceiling
        )
      
        if bonus.max() > 0.005:
            logger.debug(f"🔗 HebbLayer[{self.layer_idx}]: link bonus до {bonus.max():.4f}")
  
    @torch.no_grad()
    def _update_association_immunity(self):
        """v5.3.2: Связанные слоты получают иммунитет от смерти + FIX-5: убран спам"""
        n_init = int(self.initialized_slots.item())
        if n_init < 4: # слишком рано — смысла нет
            return
      
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        if not active_mask.any():
            return
      
        # Переходы только по активным слотам
        tm = self.transition_matrix[:n_init, :n_init].clone()
      
        for i in range(n_init):
            if not active_mask[i]:
                continue
              
            # Связанные слоты (входящие + исходящие)
            outgoing = tm[i] > 0.05
            incoming = tm[:, i] > 0.05
            associated = (outgoing | incoming)
            associated[i] = False # себя не усиливаем
          
            if associated.any():
                boost = self.immunity_boost * float(self.slot_utility[i])
                self._association_immunity[:n_init][associated] += boost
              
                # 🔹 FIX-5: спам убран — логируем редко и только при значимых изменениях
                if self.step_counter.item() % 5000 == 0 and associated.sum() > 5:
                    logger.debug(
                        f"🔗 HebbLayer[{self.layer_idx}]: слот {i} усилил иммунитет "
                        f"{associated.sum().item()} связанных слотов (+{boost:.3f})"
                    )
      
        # Decay иммунитета
        self._association_immunity[:n_init] *= self.immunity_decay
      
        # Применяем влияние на utility
        influence = torch.sigmoid(self.association_weight) * self._association_immunity[:n_init]
        self.slot_utility[:n_init] = torch.clamp(
            self.slot_utility[:n_init] + influence,
            min=0.0,
            max=self.utility_ceiling
        )
  
    @torch.no_grad()
    def _create_slot(self, pattern):
        """
        v5.4: Создаёт слот в первом EMPTY месте.
        - Если idx < current_init: слот был освобождён merge/evict → заполняем дыру,
          initialized_slots не трогаем (он уже включает этот индекс).
        - Если idx >= current_init: новый слот → initialized_slots растёт до idx+1.
        initialized_slots НИКОГДА не уменьшается здесь.
        """
        empty_idx = None
        for i in range(self.num_slots):
            if self.slot_status[i] == self.SLOT_EMPTY:
                empty_idx = i
                break
        if empty_idx is None:
            logger.warning(f"⚠️ HebbLayer[{self.layer_idx}]: нет свободного слота!")
            return
        idx = empty_idx
        self.patterns.data[idx] = pattern
        self.slot_utility[idx] = 0.5
        self.slot_status[idx] = self.SLOT_ACTIVE
        self.slot_age[idx] = 0
        self.slot_last_used[idx] = self.step_counter
        self.slot_usage_count[idx] = 0.0
        self.accum_sum[idx].zero_()
        self.accum_weight[idx] = 0.0
        current_init = int(self.initialized_slots.item())
        if idx >= current_init:
            # Новый слот — расширяем границу
            self.initialized_slots.fill_(idx + 1)
        # Если idx < current_init — заполняем дыру, initialized_slots не трогаем
        self.total_updates.add_(1)
        self._patterns_version.add_(1)
        self.new_slot_created_this_step.fill_(1)
        self.patterns_norm_dirty.fill_(True)
        logger.info(
            f"🎉 HebbLayer[{self.layer_idx}]: создан слот {idx} "
            f"(теперь {int(self.initialized_slots.item())}/{self.num_slots})"
        )
  
    @torch.no_grad()
    def _clear_new_accumulators(self):
        self.new_slot_accum_sum.zero_()
        self.new_slot_accum_weight.fill_(0.0)
        self.new_slot_steps.fill_(0)
        self._clear_residual_accumulator()
  
    @torch.no_grad()
    def _clear_main_accumulator(self):
        self.new_slot_accum_sum.zero_()
        self.new_slot_accum_weight.fill_(0.0)
        self.new_slot_steps.fill_(0)
  
    @torch.no_grad()
    def _clear_residual_accumulator(self):
        self.new_slot_accum_sum_residual.zero_()
        self.new_slot_accum_weight_residual.fill_(0.0)
        self.new_slot_steps_residual.fill_(0)
  
    @torch.no_grad()
    def _mark_candidates(self):
        """v5.3.2: CANDIDATE только при полной ёмкости + иммунитет"""
        if torch.is_grad_enabled():
            return
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            return
        step = int(self.step_counter.item())
        is_full = (n_init >= self.num_slots)
        is_full_tensor = torch.tensor(is_full, device=self.slot_status.device)
       
        # 🔥 ВЕКТОРИЗОВАННАЯ ВЕРСИЯ
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        utils = self.slot_utility[:n_init]
        immunity = self._association_immunity[:n_init]
       
        effective_thresh = self.candidate_threshold * (1.0 - 0.35 * torch.clamp(immunity, max=1.0))
        effective_thresh = torch.clamp(effective_thresh, min=0.01)
       
        to_candidate = active_mask & (utils < effective_thresh) & is_full_tensor
        self.slot_status[:n_init][to_candidate] = self.SLOT_CANDIDATE
        self.slot_candidate_since[:n_init][to_candidate] = step
       
        # Возврат из CANDIDATE в ACTIVE
        candidate_mask = (self.slot_status[:n_init] == self.SLOT_CANDIDATE)
        to_active = candidate_mask & (utils >= self.candidate_threshold)
        self.slot_status[:n_init][to_active] = self.SLOT_ACTIVE
  
    @torch.no_grad()
    def _replace_expired(self):
        """🆕 v5.3: Эвикшн только при полной ёмкости"""
        if torch.is_grad_enabled():
            return
        n_init = int(self.initialized_slots.item())
        if n_init == 0 or n_init < self.num_slots: # пока есть место — НЕ убиваем
            return
      
        step = int(self.step_counter.item())
        max_rep = max(1, n_init // 10)
        to_replace = []
      
        for i in range(n_init):
            if len(to_replace) >= max_rep:
                break
            st = int(self.slot_status[i])
            grace_expired = (
                st == self.SLOT_CANDIDATE and
                (step - int(self.slot_candidate_since[i])) >= self.candidate_grace_steps
            )
            if grace_expired:
                to_replace.append(i)
      
        if not to_replace:
            return
      
        w_total = float(self.new_slot_accum_weight)
        accum_available = w_total >= self.accum_min_weight
      
        if accum_available:
            new_emb = self.new_slot_accum_sum / w_total
            new_norm = F.normalize(new_emb, dim=-1) if new_emb.norm() > 1e-6 else None
        else:
            new_norm = None
      
        replaced = 0
        used_accum = False
      
        for idx in to_replace:
            if replaced >= max_rep:
                break
            replaced_with_data = False
          
            if new_norm is not None and not used_accum:
                others = [j for j in range(n_init) if j != idx and int(self.slot_status[j]) == self.SLOT_ACTIVE]
                can_place = True
                if others:
                    o_norms = self._get_patterns_norm()[others]
                    sims = torch.matmul(new_norm.unsqueeze(0), o_norms.T).squeeze(0)
                    if sims.max().item() > self.new_slot_sim_threshold:
                        can_place = False
              
                if can_place:
                    self.patterns.data[idx] = new_norm
                    self.slot_utility[idx] = 0.4
                    self.slot_status[idx] = self.SLOT_ACTIVE
                    self.slot_age[idx] = 0
                    self.slot_last_used[idx] = self.step_counter
                    self.slot_usage_count[idx] = 0.0
                    self.accum_sum[idx].zero_()
                    self.accum_weight[idx] = 0.0
                    replaced_with_data = True
                    used_accum = True
          
            if not replaced_with_data:
                self._clear_slot(idx)
          
            replaced += 1
      
        if used_accum:
            self._clear_main_accumulator()
      
        if replaced > 0:
            n_empty = int((self.slot_status[:int(self.initialized_slots.item())] == self.SLOT_EMPTY).sum().item())
            if n_empty > 0:
                self._compact_with_defrag()
                # 🔥 FIX: Очищаем кэш пар после дефрагментации (индексы изменились)
                self._last_merged_pairs.clear()
            self.patterns_norm_dirty.fill_(True)
            self._patterns_version.add_(1)
  
    @torch.no_grad()
    def _clear_slot(self, slot_idx: int):
        self.patterns.data[slot_idx].zero_()
        self.slot_utility[slot_idx] = 0.0
        self.slot_status[slot_idx] = self.SLOT_EMPTY
        self.slot_usage_count[slot_idx] = 0.0
        self.slot_age[slot_idx] = 0
        self.accum_sum[slot_idx].zero_()
        self.accum_weight[slot_idx] = 0.0
        self.transition_matrix[slot_idx] = 0.0
        self.transition_matrix[:, slot_idx] = 0.0
  
    @torch.no_grad()
    def _merge_similar(self):
        """v5.5: Умное слияние БЕЗ ЧЁРНОЙ ДЫРЫ слот 0"""
        if torch.is_grad_enabled():
            return
       
        n_init = int(self.initialized_slots.item())
        if n_init < 16 or n_init < self.num_slots // 2:
            return
        step = int(self.step_counter.item())
        if step - self._merge_cooldown_step < 50:
            return
        p_norm = self._get_patterns_norm()[:n_init]
        sim_mat = torch.matmul(p_norm, p_norm.T)
        upper = torch.triu(sim_mat, diagonal=1)
        pairs = (upper > self.merge_threshold).nonzero(as_tuple=False)
        if pairs.numel() == 0:
            return
        # 🔥 ОЧИЩАЕМ СТАРЫЕ ПАРЫ (оставляем последние 100)
        if len(self._last_merged_pairs) > 100:
            self._last_merged_pairs = set(list(self._last_merged_pairs)[-100:])
        merged_info = []
        merged = set()
        for pair in pairs[:5]:
            i, j = int(pair[0]), int(pair[1])
            pair_key = tuple(sorted([i, j]))
           
            if pair_key in self._last_merged_pairs or i in merged or j in merged:
                continue
               
            if int(self.slot_status[i]) != self.SLOT_ACTIVE or \
               int(self.slot_status[j]) != self.SLOT_ACTIVE:
                continue
            # 🔥 НОРМАЛИЗОВАННЫЙ СКОР
            max_utility = max(float(self.slot_utility[i]), float(self.slot_utility[j]))
            max_connections = max(float(self.transition_matrix[i].sum()), float(self.transition_matrix[j].sum()))
           
            norm_utility_i = float(self.slot_utility[i]) / max_utility if max_utility > 0 else 0
            norm_utility_j = float(self.slot_utility[j]) / max_utility if max_utility > 0 else 0
           
            norm_conn_i = float(self.transition_matrix[i].sum()) / max_connections if max_connections > 0 else 0
            norm_conn_j = float(self.transition_matrix[j].sum()) / max_connections if max_connections > 0 else 0
           
            # Комбинированный скор (60% utility, 40% связи)
            score_i = norm_utility_i * 0.6 + norm_conn_i * 0.4
            score_j = norm_utility_j * 0.6 + norm_conn_j * 0.4
            # 🔥 ЗАЩИТА ОТ ДОМИНИРОВАНИЯ СЛОТА 0
            if i == 0 and score_i - score_j < 0.2: # если преимущество < 20%
                keep, drop = j, i
            elif j == 0 and score_j - score_i < 0.2:
                keep, drop = i, j
            else:
                keep = i if score_i >= score_j else j
                drop = j if keep == i else i
            # Слияние
            self.transition_matrix[keep] += self.transition_matrix[drop]
            self.transition_matrix[:, keep] += self.transition_matrix[:, drop]
           
            rs = self.transition_matrix[keep].sum()
            if rs > 1e-4:
                self.transition_matrix[keep] /= rs
            cs = self.transition_matrix[:, keep].sum()
            if cs > 1e-4:
                self.transition_matrix[:, keep] /= cs
            self.slot_utility[keep] = max(float(self.slot_utility[i]), float(self.slot_utility[j]))
            self._clear_slot(drop)
            merged.add(i)
            merged.add(j)
            self._last_merged_pairs.add(pair_key)
           
            merged_info.append(f"{drop}→{keep} (u:{self.slot_utility[keep]:.2f})")
        if merged_info:
            self._merge_cooldown_step = step
            self._patterns_version.add_(1)
            self.patterns_norm_dirty.fill_(True)
  
    @torch.no_grad()
    def _decay_unused(self):
        """v5.3.2: Decay с учётом иммунитета по ассоциациям"""
        if torch.is_grad_enabled():
            return
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            return
        is_full = (n_init >= self.num_slots)
        stale_thr = self.consolidation_interval * self.decay_interval_mult * 5
        time_since = (self.step_counter - self.slot_last_used[:n_init]).float()
        stale_mask = time_since > stale_thr
        if not stale_mask.any():
            return
        for i in stale_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            usage = float(self.slot_usage_count[i])
            immunity = float(self._association_immunity[i])
            age = int(self.slot_age[i])
            if usage == 0.0:
                if age < 1000 or not is_full:
                    continue
              
                # Иммунитет сильно защищает
                if immunity > 0.5:
                    self.slot_utility[i] *= (self.decay_factor_new * (1.0 - 0.3 * immunity))
                    logger.info(f"🛡️ HebbLayer[{self.layer_idx}]: слот {i} защищён иммунитетом {immunity:.2f}")
                else:
                    self.slot_utility[i] *= self.decay_factor_new
            else:
                self.slot_utility[i] *= self.decay_factor
  
    @torch.no_grad()
    def _compact_with_defrag(self):
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            return
          
        alive = [i for i in range(n_init)
                if int(self.slot_status[i]) != self.SLOT_EMPTY]
      
        if len(alive) == n_init:
            return
        
        # 🔥 СОХРАНЯЕМ СТАРУЮ МАТРИЦУ ПЕРЕД ИЗМЕНЕНИЯМИ
        old_tm = self.transition_matrix.clone()
      
        for new_i, old_i in enumerate(alive):
            if new_i == old_i:
                continue
            self.patterns.data[new_i] = self.patterns.data[old_i].clone()
            self.slot_utility[new_i] = self.slot_utility[old_i]
            self._association_immunity[new_i] = self._association_immunity[old_i]
            self.slot_status[new_i] = self.slot_status[old_i]
            self.slot_candidate_since[new_i] = self.slot_candidate_since[old_i]
            self.slot_usage_count[new_i] = self.slot_usage_count[old_i]
            self.slot_age[new_i] = self.slot_age[old_i]
            self.slot_last_used[new_i] = self.slot_last_used[old_i]
            self.accum_sum[new_i] = self.accum_sum[old_i].clone()
            self.accum_weight[new_i] = self.accum_weight[old_i]
      
        tail_start = len(alive)
        for i in range(tail_start, n_init):
            self._clear_slot(i)
            self.slot_candidate_since[i] = 0
      
        # Перестраиваем матрицу ИЗ СТАРОЙ КОПИИ
        self.transition_matrix.zero_()
      
        if len(alive) > 1:
            for new_i, old_i in enumerate(alive):
                for new_j, old_j in enumerate(alive):
                    self.transition_matrix[new_i, new_j] = old_tm[old_i, old_j]
      
        self.initialized_slots.fill_(len(alive))
        self.prev_idx_valid.fill_(False)
  
    @torch.no_grad()
    def _update_stdp(self, best_idx, write_gate):
        """🔥 ARCH-4: STDP только для ACTIVE→ACTIVE переходов"""
        curr_n = len(best_idx)
        n_write = min(curr_n, self.prev_best_idx.shape[0])
      
        n_init = int(self.initialized_slots.item())
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
      
        if self.prev_idx_valid.item():
            prev_n = int(self.prev_batch_size)
            n = min(prev_n, curr_n, n_write)
            if n > 0:
                prev = self.prev_best_idx[:n]
                curr = best_idx[:n]
                w = write_gate.squeeze(-1)[:n].float()
              
                # 🔥 Фильтр ACTIVE→ACTIVE
                prev_active = active_mask[prev]
                curr_active = active_mask[curr]
                valid = prev_active & curr_active
              
                if valid.any():
                    p = prev[valid]
                    c = curr[valid]
                    weff = w[valid] * self.stdp_lr
                    
                    # 🔥 ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: проверяем индексы
                    p_clamped = torch.clamp(p, 0, self.num_slots - 1)
                    c_clamped = torch.clamp(c, 0, self.num_slots - 1)
                    lin = (p_clamped * self.num_slots + c_clamped).long()
                    
                    self.transition_matrix.reshape(-1).index_add_(
                        0, lin, weff.to(self.transition_matrix.dtype)
                    )
                    
                    for ri in torch.unique(p_clamped):
                        row = self.transition_matrix[ri]
                        rs = row.sum()
                        if rs > 1e-4:
                            self.transition_matrix[ri] = row / rs
      
        self.prev_best_idx[:n_write] = best_idx[:n_write]
        self.prev_write_gate[:n_write] = write_gate.squeeze(-1)[:n_write]
        self.prev_batch_size.fill_(curr_n)
        self.prev_idx_valid.fill_(True)
        
    @torch.no_grad()
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        🔥 v5.5: Авто-расширение слотов + FIX для квадратных матриц (transition_matrix).
        Поддерживает расширение:
        - Тензоров [N, ...] → [M, ...] (паттерны, utility, статусы)
        - Квадратных матриц [N, N] → [M, M] (transition_matrix)
        """
        current_slots = self.num_slots
        device = self.patterns.device
        fixed_state_dict = {}
        
        for key, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                fixed_state_dict[key] = param
                continue

            # 🔥 FIX: Явная проверка None вместо 'or'
            target = self._buffers.get(key, None)
            if target is None:
                target = self._parameters.get(key, None)
                
            if target is None:
                fixed_state_dict[key] = param
                continue

            target_shape = target.shape
            param_shape = param.shape

            if target_shape == param_shape:
                fixed_state_dict[key] = param
                continue

            # 1. Расширение по первому измерению [N, D] → [M, D]
            needs_expansion_1d = (
                len(target_shape) == len(param_shape) and
                param_shape[0] < current_slots and
                target_shape[0] == current_slots and
                target_shape[1:] == param_shape[1:]
            )

            if needs_expansion_1d:
                new_param = torch.zeros(target_shape, dtype=target.dtype, device=device)
                old_n = param_shape[0]
                new_param[:old_n] = param.to(device)
                fixed_state_dict[key] = new_param
                logger.info(f"📦 HebbLayer[{self.layer_idx}]: расширен {key} {param_shape} → {target_shape}")
                continue

            # 2. 🔥 Расширение квадратных матриц [N, N] → [M, M]
            needs_expansion_sq = (
                len(target_shape) == 2 and len(param_shape) == 2 and
                param_shape[0] == param_shape[1] and
                target_shape[0] == target_shape[1] and
                param_shape[0] < target_shape[0] and
                target_shape[0] == current_slots
            )

            if needs_expansion_sq:
                new_param = torch.zeros(target_shape, dtype=target.dtype, device=device)
                old_n = param_shape[0]
                # Копируем старые связи в левый верхний угол
                new_param[:old_n, :old_n] = param.to(device)
                fixed_state_dict[key] = new_param
                logger.info(f"📦 HebbLayer[{self.layer_idx}]: расширена матрица {key} {param_shape} → {target_shape}")
                continue

            # Несовпадение, которое нельзя исправить
            fixed_state_dict[key] = param
            if target_shape != param_shape:
                logger.warning(f"⚠️ HebbLayer[{self.layer_idx}]: несовпадение {key} {param_shape} vs {target_shape} (не расширено)")

        if "patterns" in fixed_state_dict:
            self.patterns_norm_dirty.fill_(True)

        return super().load_state_dict(fixed_state_dict, strict=strict, assign=assign)
    
    # ================================================================
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ================================================================
    def consolidate_now(self):
        if not self.patterns_frozen.item():
            with torch.no_grad():
                self._consolidate()
                self._consolidate_new_slot()
                self._merge_similar()
                self._decay_unused()
                self._mark_candidates()
                self._replace_expired()
  
    def freeze_patterns(self):
        self.patterns_frozen.fill_(True)
        self.prev_idx_valid.fill_(False)
        with torch.no_grad():
            self.accum_sum.zero_()
            self.accum_weight.zero_()
            self._clear_new_accumulators()
  
    def unfreeze_patterns(self):
        self.patterns_frozen.fill_(False)
  
    def is_patterns_frozen(self):
        return bool(self.patterns_frozen.item())
  
    def _is_meta(self):
        return self.patterns.device.type == 'meta'
  
    def _get_patterns_norm(self):
        if self.patterns_norm_dirty.item():
            self.patterns_norm_cache = F.normalize(self.patterns.detach(), dim=-1)
            self.patterns_norm_dirty.fill_(False)
        return self.patterns_norm_cache
  
    def _get_temperature(self):
        if self._is_meta():
            return torch.tensor(2.0, device='meta')
        temp = torch.exp(self.temperature_logit.clamp(-2.0, 2.0)).clamp(0.5, 4.0)
        temp = torch.nan_to_num(temp, nan=2.0, posinf=4.0, neginf=0.5)
        return torch.clamp(temp, 0.5, 8.0)
        
    @torch.no_grad()
    def analyze_transitions(self):
        """Анализирует качество связей между слотами в Hebb"""
        n_init = int(self.initialized_slots.item())
        if n_init < 2:
            logger.info(f"🔗 HebbLayer[{self.layer_idx}]: недостаточно слотов для анализа")
            return {
                'density': 0.0,
                'mean_strength': 0.0,
                'entropy': 0.0,
                'non_zero': 0,
                'total': 0,
                'active_slots': 0
            }
       
        # 🔥 ИСПРАВЛЕНИЕ: берём ТОЛЬКО активные слоты
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        active_indices = torch.where(active_mask)[0]
       
        if len(active_indices) < 2:
            return {
                'density': 0.0,
                'mean_strength': 0.0,
                'entropy': 0.0,
                'non_zero': 0,
                'total': 0,
                'active_slots': len(active_indices)
            }
       
        # Матрица переходов для активных слотов
        tm = self.transition_matrix[:n_init, :n_init]
        trans = tm[active_indices][:, active_indices]
       
        non_zero = (trans > 0.01).sum().item()
        total = len(active_indices) ** 2
        density = non_zero / total if total > 0 else 0
       
        if non_zero > 0:
            mean_strength = trans[trans > 0.01].mean().item()
        else:
            mean_strength = 0.0
       
        # Энтропия распределений
        row_sums = trans.sum(dim=1, keepdim=True) + 1e-8
        probs = trans / row_sums
        valid_rows = row_sums.squeeze() > 0.01
        if valid_rows.any():
            entropy = -(probs[valid_rows] * (probs[valid_rows] + 1e-10).log()).sum(dim=1).mean().item()
        else:
            entropy = 0.0
       
        logger.info(f"🔗 HebbLayer[{self.layer_idx}]: плотность={density:.4f} ({non_zero}/{total}), "
                    f"средняя сила={mean_strength:.3f}, энтропия={entropy:.3f}")
       
        return {
            'density': density,
            'mean_strength': mean_strength,
            'entropy': entropy,
            'non_zero': non_zero,
            'total': total,
            'active_slots': len(active_indices)
        }
    
    @torch.no_grad()
    def show_hubs(self, top_k=5):
        """Показывает самые связные слоты (хабы) в Hebb"""
        n_init = int(self.initialized_slots.item())
        if n_init == 0:
            logger.info(f"🌟 HebbLayer[{self.layer_idx}]: нет активных слотов")
            return []
       
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        if not active_mask.any():
            logger.info(f"🌟 HebbLayer[{self.layer_idx}]: нет активных слотов")
            return []
       
        # 🔥 ИСПРАВЛЕНИЕ: обрезаем матрицу до n_init x n_init
        tm = self.transition_matrix[:n_init, :n_init]
       
        threshold = 0.01
        in_degree = (tm > threshold).sum(dim=0).float() # [n_init]
        out_degree = (tm > threshold).sum(dim=1).float() # [n_init]
        total_degree = in_degree + out_degree # [n_init] - теперь одинаково!
       
        # Маскируем неактивные слоты
        total_degree[~active_mask] = -1
       
        k = min(top_k, active_mask.sum().item())
        if k == 0:
            return []
       
        values, indices = torch.topk(total_degree, k)
       
        logger.info(f"🌟 HebbLayer[{self.layer_idx}] Топ-{k} хабов:")
        hubs_info = []
        for i, (idx, val) in enumerate(zip(indices.tolist(), values.tolist())):
            utility = float(self.slot_utility[idx])
            age = int(self.slot_age[idx])
            usage = float(self.slot_usage_count[idx])
            immunity = float(self._association_immunity[idx]) if hasattr(self, '_association_immunity') else 0.0
           
            # Эмодзи для типа слота
            if immunity > 0.5:
                type_emoji = "🛡️" # защищённый
            elif utility > 0.8:
                type_emoji = "👑" # король
            elif utility > 0.5:
                type_emoji = "🟢" # активный
            else:
                type_emoji = "🟡" # слабый
           
            logger.info(f" #{i+1}: {type_emoji} слот {idx}, связей={int(val)}, "
                       f"utility={utility:.3f}, age={age}, использований={int(usage)}")
           
            hubs_info.append({
                'slot': idx,
                'connections': int(val),
                'utility': utility,
                'age': age,
                'usage': usage,
                'immunity': immunity,
                'type_emoji': type_emoji
            })
       
        return hubs_info
        
    @torch.no_grad()
    def analyze_network(self):
        """Комплексный анализ сети связей в Hebb"""
        n_init = int(self.initialized_slots.item())
        if n_init < 2:
            return {}
       
        active_mask = (self.slot_status[:n_init] == self.SLOT_ACTIVE)
        active_indices = torch.where(active_mask)[0]
       
        if len(active_indices) < 2:
            return {}
       
        # 🔥 ИСПРАВЛЕНИЕ: работаем ТОЛЬКО с матрицей размера n_init x n_init
        tm = self.transition_matrix[:n_init, :n_init]
       
        threshold = 0.01
        in_degree = (tm > threshold).sum(dim=0).float() # [n_init]
        out_degree = (tm > threshold).sum(dim=1).float() # [n_init]
       
        # Основные метрики
        trans_stats = self.analyze_transitions()
        hubs = self.show_hubs(top_k=3)
       
        # 🔥 ИСПРАВЛЕНИЕ: все тензоры должны быть одинакового размера [n_init]
        in_zero = (in_degree == 0) # [n_init]
        out_zero = (out_degree == 0) # [n_init]
       
        # Изолированные слоты (нет связей) - используем ПОЭЛЕМЕНТНОЕ И
        isolated_mask = active_mask & in_zero & out_zero # все три [n_init]
        isolated_count = isolated_mask.sum().item()
       
        # Мосты - слоты с высокой связностью
        betweenness = in_degree * out_degree # [n_init]
        median_val = betweenness[active_mask].median() if active_mask.any() else 0
        bridges_mask = active_mask & (betweenness > median_val)
        bridge_count = bridges_mask.sum().item()
       
        logger.info(f"🌐 HebbLayer[{self.layer_idx}] СЕТЬ: "
                    f"активных={len(active_indices)}, "
                    f"изолированных={isolated_count}, "
                    f"мостов={bridge_count}, "
                    f"плотность={trans_stats['density']:.4f}")
       
        return {
            'active_count': len(active_indices),
            'isolated_count': isolated_count,
            'bridge_count': bridge_count,
            'density': trans_stats['density'],
            'entropy': trans_stats['entropy'],
            'hubs': hubs
        }
  
    def log_init(self, total_layers=None):
        if self._is_meta() or self.layer_idx != 0:
            return
        # Обновляем total_layers если передан (для retroactive init)
        if total_layers is not None and self.total_layers is None:
            self.total_layers = total_layers
        n = self.total_layers or total_layers or "?"
        logger.info(f"🧠 TemporalHebbLayer v5.4: {n} слоёв × {self.num_slots} слотов")
   
    def get_stats(self):
        if self._is_meta():
            return self._fallback_stats()
        n_init = int(self.initialized_slots.item())
        n_active = int((self.slot_status[:n_init] == self.SLOT_ACTIVE).sum().item()) if n_init > 0 else 0
        n_cand = int((self.slot_status[:n_init] == self.SLOT_CANDIDATE).sum().item()) if n_init > 0 else 0
        n_dead = int((self.slot_status[:n_init] == self.SLOT_EMPTY).sum().item()) if n_init > 0 else 0
        top_transitions = []
        if n_init > 1:
            tm = self.transition_matrix[:n_init, :n_init]
            flat_vals, flat_idx = tm.reshape(-1).topk(min(3, tm.numel()))
            for v, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
                i, j = divmod(idx, n_init)
                if v > 1e-4:
                    top_transitions.append((i, j, round(v, 4)))
        temp_val = self._get_temperature()
        if isinstance(temp_val, torch.Tensor):
            temp_val = temp_val.detach().cpu().item()
        return {
            "layer_idx": self.layer_idx,
            "total_layers": self.total_layers,
            "depth_ratio": round(int(self.layer_idx) / max((self.total_layers or 1) - 1, 1), 3)
                                if self.layer_idx != "?" else None,
            "initialized": n_init,
            "active_slots": n_active,
            "candidate_slots": n_cand,
            "dead_slots": n_dead,
            "frozen": bool(self.patterns_frozen.item()),
            "total_updates": int(self.total_updates.item()),
            "total_reads": int(self.total_reads.item()),
            "avg_usage": float(self.slot_usage_count[:n_init].mean()) if n_init > 0 else 0.0,
            "utility_mean": float(self.slot_utility[:n_init].mean()) if n_init > 0 else 0.0,
            "utility_max": float(self.slot_utility[:n_init].max()) if n_init > 0 else 0.0,
            "temperature": temp_val,
            "memory_scale": float(self.memory_scale.item()),
            "accum_pending": int(self._accum_pending_snapshot.item()),
            "accum_now": int((self.accum_weight[:n_init] > 0).sum().item()) if n_init > 0 else 0,
            "new_slot_accum_weight": float(self.new_slot_accum_weight),
            "new_slot_accum_weight_residual": float(self.new_slot_accum_weight_residual),
            "new_slot_present": 1 if float(self.new_slot_accum_weight) > 0 else 0,
            "new_slot_present_residual": 1 if float(self.new_slot_accum_weight_residual) > 0 else 0,
            "new_slot_steps": int(self.new_slot_steps),
            "new_slot_steps_residual": int(self.new_slot_steps_residual),
            "top_transitions": top_transitions,
            "step": int(self.step_counter.item()),
            "aux_loss": float(self.aux_loss.item()),
            "last_query_norm": float(self._last_hebb_query.norm().item()),
            "patterns_version": int(self._patterns_version.item()),
            "last_surprise_used": float(self.last_surprise_used),
            "new_slot_sim_threshold": float(self.new_slot_sim_threshold),
            # v4.0: новая метрика доминирования
            "dominant_share_ema": float(self._dominant_share_ema.item()),
            # v5.3.2: иммунитет
            "avg_immunity": float(self._association_immunity[:n_init].mean()) if n_init > 0 else 0.0,
            "max_immunity": float(self._association_immunity[:n_init].max()) if n_init > 0 else 0.0,
            "association_weight": float(torch.sigmoid(self.association_weight).item()),
        }
  
    def _fallback_stats(self):
        return {
            "layer_idx": self.layer_idx,
            "initialized": 0, "active_slots": 0, "candidate_slots": 0, "dead_slots": 0,
            "frozen": False, "total_updates": 0, "total_reads": 0,
            "avg_usage": 0.0, "utility_mean": 0.0, "utility_max": 0.0,
            "temperature": 2.0, "memory_scale": 0.1,
            "accum_pending": 0, "accum_now": 0,
            "new_slot_accum_weight": 0.0, "new_slot_accum_weight_residual": 0.0,
            "new_slot_present": 0, "new_slot_present_residual": 0,
            "new_slot_steps": 0, "new_slot_steps_residual": 0,
            "top_transitions": [], "step": 0, "aux_loss": 0.0,
            "last_query_norm": 0.0, "patterns_version": 0,
            "last_surprise_used": 0.0, "new_slot_sim_threshold": 0.85,
            "dominant_share_ema": 0.0,
            "avg_immunity": 0.0,
            "max_immunity": 0.0,
            "association_weight": 0.5,
        }
        
# ===================================================================              
# НЕЙРОКОНТРОЛЛЕР v9.8
# ===================================================================
class ImprovedPsycheController(nn.Module):
    """🔥 Нейроконтроллер с правильным PPO"""
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Policy network (mean)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(output_dim) - 1.0)
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Буферы данных - без register_buffer для индексов
        self.batch_buffer = deque(maxlen=1024)
        self.buffer_data = []  # будет сохраняться через state_dict
        
        # Параметры с register_buffer
        self.register_buffer('exploration_rate', torch.tensor(1.0))
        self.register_buffer('training_steps', torch.tensor(0))
        
        self.register_buffer('last_state', torch.zeros(input_dim))
        self.register_buffer('last_action', torch.zeros(output_dim))
        self.register_buffer('last_log_prob', torch.tensor(0.0))
        self.register_buffer('last_action_raw', torch.zeros(output_dim))
        self.register_buffer('last_mean', torch.zeros(output_dim))
        self.register_buffer('last_std', torch.ones(output_dim))
        
        self.register_buffer('steps_since_update', torch.tensor(0))
        self.update_frequency = 32
        
        self.register_buffer('current_mood', torch.zeros(1))
        
        # Инициализация
        with torch.no_grad():
            self.policy_net[-1].weight.fill_(0.0)
            self.policy_net[-1].bias.fill_(0.0)
            self.value_net[-1].weight.fill_(0.0)
            self.value_net[-1].bias.fill_(0.0)
        
        logger.info(f"🧠 PPO контроллер v9.8: вход={input_dim}, выход={output_dim}, update_freq={self.update_frequency}")

    def forward(self, state, deterministic=False, mood=None):
        """
        Gaussian policy forward - возвращает действие и log_prob
        🔥 ИСПРАВЛЕННАЯ версия с правильным сохранением batch log_prob
        """
        device = state.device
        
        action_mean = self.policy_net(state)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=5.0, neginf=-5.0)
        action_mean = torch.clamp(action_mean, -10.0, 10.0)
        
        if deterministic:
            return action_mean, None
        
        if mood is not None:
            mood_val = mood.mean().cpu().item()
            self.current_mood.fill_(mood_val)
        
        log_std = self.log_std.to(device)
        mood_influence = abs(self.current_mood.item()) * 0.7
        current_log_std = log_std + mood_influence
        
        std = torch.exp(current_log_std)
        std = torch.nan_to_num(std, nan=0.1, posinf=5.0, neginf=0.1)
        std = torch.clamp(std, min=0.1, max=5.0)
        
        self.last_mean.data.copy_(action_mean.detach().clone())
        self.last_std.data.copy_(std.detach().clone())
        
        try:
            action_dist = torch.distributions.Normal(action_mean, std)
            action_raw = action_dist.rsample()
        except Exception as e:
            logger.warning(f"⚠ Ошибка в rsample: {e}, возвращаю zero action")
            action_raw = torch.zeros_like(action_mean)
        
        action_raw = torch.nan_to_num(action_raw, nan=0.0, posinf=3.0, neginf=-3.0)
        action_raw = torch.clamp(action_raw, -10.0, 10.0)
        
        self.last_action_raw.data.copy_(action_raw.detach().clone())
        
        action = torch.tanh(action_raw) * 2.0
        action = torch.nan_to_num(action, nan=0.0)
        
        # 🔥 ИСПРАВЛЕНИЕ: Вычисляем log_prob с правильной размерностью
        try:
            # Получаем поэлементные log_prob: [batch_size, output_dim]
            log_prob_per_dim = action_dist.log_prob(action_raw)
        except Exception as e:
            logger.warning(f"⚠ Ошибка в log_prob: {e}, возвращаю -10.0")
            log_prob_per_dim = torch.full_like(action_raw, -10.0)
        
        log_prob_per_dim = torch.nan_to_num(log_prob_per_dim, nan=-10.0, posinf=10.0, neginf=-10.0)
        
        # Производная tanh: 2.0 * (1 - tanh^2)
        tanh_deriv = 2.0 * (1 - torch.tanh(action_raw).pow(2))
        tanh_deriv = torch.clamp(tanh_deriv, min=1e-6)
        
        # 🔥 Поэлементная коррекция
        log_prob_corrected = log_prob_per_dim - torch.log(tanh_deriv)
        
        # Суммируем по выходной размерности
        log_prob = log_prob_corrected.sum(dim=-1)
        
        log_prob = torch.nan_to_num(log_prob, nan=-10.0, posinf=10.0, neginf=-10.0)
        log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
        
        # Сохраняем для истории
        if log_prob.dim() == 0:
            self.last_log_prob.data.copy_(log_prob.detach().clone())
            self.last_log_prob_batch = log_prob.detach().clone().unsqueeze(0)
        else:
            self.last_log_prob.data.copy_(log_prob.mean().detach().clone())
            self.last_log_prob_batch = log_prob.detach().clone()
        
        self.last_state.data.copy_(state.detach().clone())
        self.last_action.data.copy_(action.detach().clone())
        
        if self.training and self.training_steps.item() % 100 == 0:
            self.exploration_rate.mul_(0.998).clamp_(0.1, 1.0)
        
        self.training_steps.add_(1)
        self.steps_since_update.add_(1)
        
        return action, log_prob

    def get_current_distribution(self, state):
        """
        🔥 ПОЛУЧАЕТ ТЕКУЩЕЕ распределение (с текущими весами)
        """
        device = state.device
        
        action_mean = self.policy_net(state)
        
        log_std = self.log_std.to(device)
        mood_influence = abs(self.current_mood.item()) * 0.3
        current_log_std = log_std + mood_influence
        
        std = torch.exp(current_log_std)
        std = torch.clamp(std, min=0.1, max=5.0)
        
        return torch.distributions.Normal(action_mean, std)

    def compute_log_prob_current(self, state, action_raw):
        """
        🔥 ИСПРАВЛЕННОЕ вычисление log_prob для raw action используя ТЕКУЩИЕ веса
        С правильным согласованием размерностей
        """
        device = state.device
        
        action_mean = self.policy_net(state)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=5.0, neginf=-5.0)
        action_mean = torch.clamp(action_mean, -10.0, 10.0)
        
        log_std = self.log_std.to(device)
        mood_influence = abs(self.current_mood.item()) * 0.3
        current_log_std = log_std + mood_influence
        
        std = torch.exp(current_log_std)
        std = torch.nan_to_num(std, nan=0.1, posinf=5.0, neginf=0.1)
        std = torch.clamp(std, min=0.1, max=5.0)
        
        # 🔥 Создаем распределение
        try:
            action_dist = torch.distributions.Normal(action_mean, std)
            # action_raw: [batch_size, output_dim]
            # log_prob: [batch_size, output_dim] (не суммируем сразу!)
            log_prob = action_dist.log_prob(action_raw)
        except Exception as e:
            logger.warning(f"⚠ Ошибка в log_prob: {e}, возвращаю -10.0")
            return torch.full((state.shape[0],), -10.0, device=device)
        
        # 🔥 ИСПРАВЛЕНИЕ: Коррекция для tanh должна применяться поэлементно
        # action_raw: [batch_size, output_dim]
        # tanh_deriv: [batch_size, output_dim]
        tanh_deriv = 2.0 * (1 - torch.tanh(action_raw).pow(2))
        safe_deriv = torch.clamp(tanh_deriv, min=1e-6)
        
        # Поэлементное вычитание: [batch_size, output_dim] - [batch_size, output_dim] = [batch_size, output_dim]
        log_prob = log_prob - torch.log(safe_deriv)
        
        # 🔥 Суммируем по последнему измерению (output_dim) ТОЛЬКО после коррекции
        log_prob = log_prob.sum(dim=-1)
        
        # Защита от NaN/Inf
        log_prob = torch.nan_to_num(log_prob, nan=-10.0, posinf=10.0, neginf=-10.0)
        log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
        
        return log_prob
    
    def get_value(self, state):
        """Получение value estimation"""
        return self.value_net(state)
    
    def add_to_batch_buffer(self, state, action_raw, old_log_prob, reward, next_state, done, value):
        """
        С сохранением в буфер
        """
        idx = len(self.buffer_data)
        if idx >= 1024:
            self.buffer_data.pop(0)
            idx = 1023
        
        # Конвертируем всё в тензоры на CPU для сохранения
        item = {
            'state': state.detach().cpu() if torch.is_tensor(state) else torch.tensor(state),
            'action_raw': action_raw.detach().cpu() if torch.is_tensor(action_raw) else torch.tensor(action_raw),
            'old_log_prob': old_log_prob.detach().cpu() if torch.is_tensor(old_log_prob) else torch.tensor(old_log_prob),
            'reward': torch.tensor(reward) if not torch.is_tensor(reward) else reward.detach().cpu(),
            'next_state': next_state.detach().cpu() if torch.is_tensor(next_state) else torch.tensor(next_state),
            'done': torch.tensor(done) if not torch.is_tensor(done) else done.detach().cpu(),
            'value': value.detach().cpu() if value is not None else None
        }
        self.buffer_data.append(item)
    
    def should_update(self):
        """Проверяем, пора ли обновляться"""
        return len(self.buffer_data) >= self.update_frequency
    
    def clear_batch_buffer(self):
        """Очищаем буфер после обновления"""
        self.buffer_data.clear()
        # self.buffer_state_indices.zero_()    # ← ЗАКОММЕНТИРОВАТЬ
        # self.buffer_action_indices.zero_()   # ← ЗАКОММЕНТИРОВАТЬ
        self.steps_since_update.data.zero_()
    
    def get_minibatch(self):
        """
        🔥 ИСПРАВЛЕНО: Возвращает тензоры на CPU для последующего перемещения
        """
        batch = list(self.buffer_data)
        
        if not batch:
            return None
        
        # ✅ Возвращаем на CPU, основное устройство будет применено в _perform_ppo_update
        states = torch.stack([b['state'].cpu() for b in batch])
        action_raws = torch.stack([b['action_raw'].cpu() for b in batch])
        old_log_probs = torch.stack([b['old_log_prob'].cpu() for b in batch])
        rewards = torch.stack([b['reward'].cpu() for b in batch])
        next_states = torch.stack([b['next_state'].cpu() for b in batch])
        dones = torch.stack([b['done'].cpu() for b in batch])
        
        values = None
        if batch[0]['value'] is not None:
            values = torch.stack([b['value'].cpu() for b in batch])
        
        return states, action_raws, old_log_probs, rewards, next_states, dones, values

    def state_dict(self, *args, **kwargs):
        """Сохраняем состояние контроллера, буфер сохраняем отдельно"""
        state = super().state_dict(*args, **kwargs)
        
        # ВАЖНО: НЕ добавляем buffer_data в основной state_dict!
        # Он будет сохранён отдельно в thalia_enhanced_state.pt
        
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Загружаем состояние контроллера, буфер загружаем отдельно"""
        # Загружаем только основные параметры
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        
        # Буфер инициализируем пустым - он загрузится отдельно через set_psyche_state
        self.buffer_data = []
        
        return result

# ===================================================================
# ДИНАМИЧЕСКАЯ СИСТЕМА НАСТРОЕНИЯ (MOOD) v9.0
# ===================================================================
class MoodSystem(nn.Module):
    """
    🎭 Динамическая система настроения с адаптивным recovery
    """   
    def __init__(self, num_traits: int):
        super().__init__()
        
        # Обучаемые параметры
        self.mood_state = nn.Parameter(torch.tensor(0.0))
        self.mood_velocity = nn.Parameter(torch.tensor(0.0))
        
        # Динамические коэффициенты
        self.inertia = nn.Parameter(torch.tensor(0.85))
        self.recovery_rate = nn.Parameter(torch.tensor(0.08))
        
        # Чувствительность
        self.balance_sensitivity = nn.Parameter(torch.tensor(0.08))
        self.tension_sensitivity = nn.Parameter(torch.tensor(0.2))
        self.experience_sensitivity = nn.Parameter(torch.tensor(0.07))
        
        # Дополнительные параметры
        self.mood_mass = nn.Parameter(torch.tensor(0.8))
        self.mood_damping = nn.Parameter(torch.tensor(0.08))
        
        # История для анализа
        self.register_buffer("_mood_history", torch.zeros(100))
        self.register_buffer("_history_idx", torch.tensor(0))
        
        # Эмоциональное перенапряжение (ИСПРАВЛЕНО: register_buffer)
        self.register_buffer("_overload_factor", torch.tensor(1.0))
        self.register_buffer("_overload_recovery", torch.tensor(1.0))
        
        logger.info(f"🎭 Динамическая система настроения v9.0 инициализирована")
    
    def update(self, trait_values: torch.Tensor, 
               tension_level: float,
               experience_intensity: float = 0.5) -> None:
        device = trait_values.device
        
        with torch.no_grad():
            # 🔥 ОПТИМИЗИРОВАНО: векторизованные вычисления
            std_val = torch.std(trait_values).item()
            balance_score = max(-1.0, min(1.0, 1.0 - std_val * 3))
            
            tension_effect = -tension_level * self.tension_sensitivity.item() * self._overload_factor.item()
            experience_effect = ((experience_intensity - 0.5) ** 3) * self.experience_sensitivity.item() * 4
            
            current_mood = self.mood_state.item()
            recovery_strength = self.recovery_rate.item() + 2.0 * (current_mood ** 2)
            recovery_effect = -current_mood * recovery_strength * 2.0

            external_delta = (
                balance_score * self.balance_sensitivity.item() +
                tension_effect +
                experience_effect
            )
            
            external_delta = torch.clamp(torch.tensor(external_delta), -0.1, 0.1).item()
            
            total_delta = external_delta + recovery_effect
            effective_mass = self.mood_mass.item() * 0.5 
            mood_acceleration = total_delta / effective_mass
            
            new_velocity = (
                0.6 * self.mood_velocity +
                0.4 * mood_acceleration -
                self.mood_damping * self.mood_velocity
            )
            
            new_velocity = torch.clamp(new_velocity, -0.2, 0.2)
            self.mood_velocity.data.copy_(new_velocity)
            
            new_mood = self.mood_state + self.mood_velocity * 0.8
            new_mood = torch.clamp(new_mood, -1.0, 1.0)
            self.mood_state.data.copy_(new_mood)
            
            # 🔥 ДОБАВЛЕНО: ОБНОВЛЕНИЕ ИСТОРИИ
            idx = self._history_idx.item()
            self._mood_history[idx] = new_mood.item()
            self._history_idx.fill_((idx + 1) % 100)
            
            self._update_overload(current_mood)
    
    def get_modifiers(self) -> Dict[str, float]:
        """🔥 ОПТИМИЗИРОВАННЫЕ модификаторы на основе настроения"""
        mood = self.mood_state.item()
        
        if math.isnan(mood) or math.isinf(mood):
            mood = 0.0
            self.mood_state.data.fill_(0.0)
        
        # 🔥 ОПТИМИЗИРОВАНО: вычисляем sigmoid один раз
        mood_sigmoid = float(torch.sigmoid(torch.tensor(mood * 3.0)).item())
        plasticity_mult = 0.7 + 0.9 * mood_sigmoid
        resistance_mult = 1.8 - 1.2 * mood_sigmoid
        
        overload_effect = 1.0 / (1.0 + abs(mood) * self._overload_factor.item() * 1.2)
        
        return {
            "plasticity_mult": plasticity_mult * overload_effect,
            "resistance_mult": resistance_mult * overload_effect,
            "exploration_bonus": max(0, mood) ** 2 * 0.4 * overload_effect,
            "noise_mult": 1.0 + abs(mood) * 0.7,
            "learning_rate_mult": 0.7 + 0.6 * (1 - abs(mood)) * overload_effect,
            "overload_factor": self._overload_factor.item(),
            "overload_recovery": self._overload_recovery.item(),
        }
    
    def _update_overload(self, current_mood):
        """🔥 СИММЕТРИЧНАЯ коррекция экстремальных значений настроения"""
        if abs(current_mood) > 0.7:
            overload_growth = 1.5 + (abs(current_mood) - 0.7) * 2.0
            
            # 🔥 ИСПРАВЛЕНО: in-place обновление буферов
            new_overload = self._overload_factor.item() * overload_growth
            self._overload_factor.fill_(min(4.0, max(1.0, new_overload)))
            
            new_recovery = self._overload_recovery.item() - 0.12
            self._overload_recovery.fill_(max(0.05, min(1.0, new_recovery)))
        else:
            target_factor = torch.tensor(1.0, device=self._overload_factor.device)
            target_recovery = torch.tensor(1.0, device=self._overload_recovery.device)
            
            # Используем lerp для плавного перехода
            self._overload_factor.lerp_(target_factor, 0.25)
            self._overload_recovery.lerp_(target_recovery, 0.25)
       
    def get_trend(self) -> float:
        """Возвращает тренд настроения"""
        history = self._mood_history
        
        if history.sum() == 0:
            return 0.0
        
        recent = history[-20:]
        
        if torch.isnan(recent).any() or torch.isinf(recent).any():
            return 0.0
        
        if recent.sum() == 0:
            return 0.0
        
        x = torch.arange(20, dtype=torch.float32, device=history.device)
        denominator = (x ** 2).sum() - x.sum() ** 2 / 20
        
        if abs(denominator) < 1e-8:
            return 0.0
            
        slope = ((x * recent).sum() - x.sum() * recent.sum() / 20) / denominator
        
        if torch.isnan(slope) or torch.isinf(slope):
            return 0.0
        
        return float(torch.tanh(slope * 120).item())

# ===================================================================
# ДИНАМИЧЕСКИЕ КОЭФФИЦИЕНТЫ v6 
# ===================================================================
class DynamicCoefficientsV6(nn.Module):
    """🔥 УСИЛЕННЫЕ динамические коэффициенты v6 (ОПТИМИЗИРОВАННЫЕ)"""   
    def __init__(self, num_traits: int, num_drives: int, drive_names: List[str] = None):
        super().__init__()
        self.num_traits = num_traits
        self.num_drives = num_drives
        self.drive_names = drive_names or []
        
        # Обучаемые базовые параметры
        self.base_plasticity = nn.Parameter(torch.zeros(num_traits))
        self.base_resistance = nn.Parameter(torch.zeros(num_traits))
        self.base_drive_rate = nn.Parameter(torch.zeros(num_drives))
        
        # Улучшенные матрицы влияния
        self.trait_influence = nn.Parameter(torch.randn(num_traits, num_traits) * 0.02)
        self.trait_to_drive = nn.Parameter(torch.randn(num_traits, num_drives) * 0.02)
        self.drive_to_trait = nn.Parameter(torch.randn(num_drives, num_traits) * 0.02)
        
        with torch.no_grad():
            self.trait_influence.data.fill_diagonal_(0)
        
        # Горячий буфер для расчетов
        self.register_buffer("_rolling_traits", torch.zeros(3, num_traits))
        self.register_buffer("_rolling_drives", torch.zeros(3, num_drives))
        
        # История и метрики
        self.register_buffer("_trait_history", torch.zeros(100, num_traits))
        self.register_buffer("_drive_history", torch.zeros(100, num_drives))
        self.register_buffer("_history_idx", torch.tensor(0))
        
        self.register_buffer("_trait_velocity", torch.zeros(num_traits))
        self.register_buffer("_trait_acceleration", torch.zeros(num_traits))
        self.register_buffer("_drive_velocity", torch.zeros(num_drives))
        
        self.register_buffer("_system_entropy", torch.tensor(0.5))
        self.register_buffer("_system_energy", torch.tensor(1.0))
        self.register_buffer("_system_stability", torch.tensor(0.5))
        
        logger.info(f"📊 Динамические коэффициенты v6 (FAST): {num_traits} черт, {num_drives} драйвов")
    
    def compute_plasticity(self, trait_values: torch.Tensor,
                           experience_count: int,
                           mood_mult: float = 1.0,
                           drive_values: torch.Tensor = None) -> torch.Tensor:
        """🔥 УСИЛЕННАЯ пластичность"""
        device = trait_values.device
        
        if torch.isnan(trait_values).any() or torch.isinf(trait_values).any():
            return torch.ones_like(trait_values) * 0.5
        
        base = torch.sigmoid(self.base_plasticity.to(device)) * 0.9 + 0.05
        position_factor = 0.1 + 5.9 * trait_values * (1 - trait_values)
        velocity_factor = 1.0 / (1.0 + torch.abs(self._trait_velocity.to(device)) * 20)
        experience_factor = 0.4 + 0.6 * math.exp(-experience_count / 80000)
        
        influence = torch.mv(self.trait_influence.to(device), trait_values)
        influence_factor = torch.sigmoid(influence * 1.0) * 0.4 + 0.6
        
        drive_factor = torch.ones_like(trait_values)
        if drive_values is not None:
            drive_effect = torch.matmul(
                drive_values.unsqueeze(0),
                self.drive_to_trait.to(device)
            ).squeeze(0)
            drive_factor = torch.sigmoid(drive_effect * 1.2) * 0.5 + 0.5
        
        stability_factor = 0.7 + 0.6 * self._system_stability.to(device)
        
        plasticity = (base * position_factor * velocity_factor * 
                     experience_factor * influence_factor * drive_factor * stability_factor)
        
        plasticity = plasticity * mood_mult
        
        if torch.isnan(plasticity).any() or torch.isinf(plasticity).any():
            plasticity = torch.ones_like(plasticity) * 0.5
        
        return torch.clamp(plasticity, 1e-6, 1.0)
    
    def compute_resistance(self, trait_values: torch.Tensor,
                           target_min: torch.Tensor,
                           target_max: torch.Tensor,
                           mood_mult: float = 1.0,
                           drive_values: torch.Tensor = None) -> torch.Tensor:
        """🔥 УСИЛЕННОЕ сопротивление"""
        device = trait_values.device
        
        if torch.isnan(trait_values).any() or torch.isinf(trait_values).any():
            return torch.ones_like(trait_values) * 0.01
        
        base = torch.sigmoid(self.base_resistance.to(device)) * 0.03
        
        target_center = (target_min.to(device) + target_max.to(device)) / 2
        target_width = (target_max.to(device) - target_min.to(device)) / 2
        
        displacement = trait_values - target_center
        normalized_dist = displacement / (target_width + 1e-6)
        
        inside = torch.abs(normalized_dist) < 1.0
        
        range_factor = torch.where(
            inside,
            normalized_dist ** 2 * 0.12,
            torch.sign(normalized_dist) * (torch.exp(torch.abs(normalized_dist) - 1.0) - 1.0) * 0.05
        )
        
        accel_factor = 1.0 + torch.abs(self._trait_acceleration.to(device)) * 10
        entropy_factor = 0.8 + self._system_entropy.to(device) * 0.4
        
        drive_resist_factor = torch.ones_like(trait_values)
        if drive_values is not None:
            drive_resist_effect = torch.matmul(
                drive_values.unsqueeze(0),
                self.drive_to_trait.to(device) * 0.2
            ).squeeze(0)
            drive_resist_factor = torch.sigmoid(-drive_resist_effect) * 0.5 + 0.5
        
        stability_factor = 0.95 + 0.1 * self._system_stability.to(device)
        
        resistance = (base + range_factor * 0.12) * accel_factor * entropy_factor * mood_mult * drive_resist_factor * stability_factor
        
        if torch.isnan(resistance).any() or torch.isinf(resistance).any():
            resistance = torch.ones_like(resistance) * 0.01
        
        return torch.clamp(resistance, 1e-6, 0.3)
    
    def compute_drive_dynamics(self, drive_values: torch.Tensor,
                               trait_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔥 ОПТИМИЗИРОВАННАЯ динамика драйвов"""
        device = drive_values.device
        
        if torch.isnan(drive_values).any() or torch.isinf(drive_values).any():
            return {
                "growth": torch.zeros_like(drive_values),
                "decay": torch.zeros_like(drive_values),
                "net_change": torch.zeros_like(drive_values)
            }
        
        # Конфиги драйвов (без изменений)
        drive_configs = {
            "novelty": {"base_mult": 1.2, "saturation_exp": 3.0, "hunger_power": 1.3, "decay_mult": 1.5},
            "fatigue": {"base_mult": 0.8, "saturation_exp": 1.8, "hunger_power": 2.5, "decay_mult": 2.0},
            "meaning": {"base_mult": 1.1, "saturation_exp": 2.5, "hunger_power": 2.0, "decay_mult": 1.8},
            "competence": {"base_mult": 1.0, "saturation_exp": 2.2, "hunger_power": 1.6, "decay_mult": 1.5},
            "identity": {"base_mult": 1.0, "saturation_exp": 2.5, "hunger_power": 2.0, "decay_mult": 1.8},
            "social": {"base_mult": 1.0, "saturation_exp": 2.5, "hunger_power": 2.0, "decay_mult": 1.8},
            "share": {"base_mult": 0.9, "saturation_exp": 2.0, "hunger_power": 2.0, "decay_mult": 1.8},
        }
        
        base_rate_raw = torch.sigmoid(self.base_drive_rate.to(device)) * 0.008
        trait_influence = torch.mv(self.trait_to_drive.to(device).T, trait_values)
        trait_factor = torch.tanh(trait_influence * 2.0) * 0.3 + 0.7
        
        growth = torch.zeros_like(drive_values)
        decay = torch.zeros_like(drive_values)
        
        # 🔥 ОПТИМИЗИРОВАНО: векторизованные вычисления для всех драйвов
        for i in range(self.num_drives):
            if i < len(self.drive_names):
                drive_name = self.drive_names[i]
                config = drive_configs.get(drive_name, {"base_mult": 1.0, "saturation_exp": 1.0, "hunger_power": 1.0, "decay_mult": 1.0})
            else:
                config = {"base_mult": 1.0, "saturation_exp": 1.0, "hunger_power": 1.0, "decay_mult": 1.0}
            
            base_rate = base_rate_raw[i] * config["base_mult"]
            saturation = 1.0 - drive_values[i] ** (config["saturation_exp"] * 1.8)
            hunger = (1.0 - drive_values[i] + 1e-6) ** (config["hunger_power"] * 1.8)
            velocity_damping = 1.0 / (1.0 + torch.abs(self._drive_velocity[i]) * 15)
            stability_factor = 0.8 + 0.4 * self._system_stability
            
            growth[i] = base_rate * trait_factor[i] * saturation * hunger * velocity_damping * stability_factor * 1.2
            decay[i] = drive_values[i] ** 3 * 0.005 * config["decay_mult"]
        
        net_change = growth - decay
        
        # 🔥 ОПТИМИЗИРОВАНО: векторизованная балансировка вместо циклов
        mean_drive = drive_values.mean()
        
        if mean_drive < 0.3:
            balance_boost = (0.4 - mean_drive) * 0.025
            net_change = net_change + balance_boost
        elif mean_drive > 0.7:
            balance_reduction = (mean_drive - 0.6) * 0.03
            net_change = net_change - balance_reduction
        
        if mean_drive > 0.55:
            excess = mean_drive - 0.55
            correction_strength = torch.clamp(excess * 0.07, max=0.03)
            correction = excess * excess * correction_strength * 1.5
            net_change -= correction
        elif mean_drive < 0.45:
            deficit = 0.45 - mean_drive
            correction_strength = torch.clamp(deficit * 0.07, max=0.03)
            correction = deficit * deficit * correction_strength * 1.5
            net_change += correction
        
        if torch.isnan(net_change).any() or torch.isinf(net_change).any():
            net_change = torch.zeros_like(net_change)
        
        return {
            "growth": growth,
            "decay": decay,
            "net_change": net_change
        }
    
    def update_history(self, trait_values: torch.Tensor, drive_values: torch.Tensor):
        """🚀 РЕАКТИВНОЕ обновление истории"""
        device = trait_values.device
        idx = self._history_idx
        
        if not torch.isfinite(trait_values).all():
            trait_values = torch.zeros_like(trait_values)
        if not torch.isfinite(drive_values).all():
            drive_values = torch.ones_like(drive_values) * 0.5

        self._rolling_traits = torch.roll(self._rolling_traits, shifts=1, dims=0)
        self._rolling_drives = torch.roll(self._rolling_drives, shifts=1, dims=0)
        self._rolling_traits[0] = trait_values.detach()
        self._rolling_drives[0] = drive_values.detach()

        t_0, t_1, t_2 = self._rolling_traits[0], self._rolling_traits[1], self._rolling_traits[2]
        d_0, d_1 = self._rolling_drives[0], self._rolling_drives[1]

        self._trait_velocity.data.copy_(t_0 - t_1)
        self._drive_velocity.data.copy_(d_0 - d_1)
        
        prev_velocity = t_1 - t_2
        self._trait_acceleration.data.copy_(self._trait_velocity - prev_velocity)

        self._system_energy.data.copy_(torch.norm(t_0) / math.sqrt(self.num_traits))

        curr_idx_int = idx.item()
        self._trait_history[curr_idx_int] = t_0.cpu()
        self._drive_history[curr_idx_int] = d_0.cpu()

        if curr_idx_int >= 20:
            recent = self._trait_history[max(0, curr_idx_int-20):curr_idx_int+1]
            std_val = recent.std(dim=0).mean()
            self._system_stability.data.copy_(torch.clamp(1.0 - std_val * 1.5, 0.0, 1.0))
            
            var_val = recent.var(dim=0).mean()
            self._system_entropy.data.copy_(torch.clamp(var_val * 1000, 0.0, 1.0))

        self._history_idx.data.fill_((curr_idx_int + 1) % 100)

# ===================================================================
# СИСТЕМА ДОЛГОСРОЧНЫХ ЦЕЛЕЙ 
# ===================================================================
class GoalSystem(nn.Module):
    """🎯 Система долгосрочных целей с динамическим приоритетом"""
    
    def __init__(self, num_traits: int, num_goals: int = 5):
        super().__init__()
        self.num_traits = num_traits
        self.num_goals = num_goals
        
        self.goal_embeddings = nn.Embedding(num_goals, num_traits)
        self.goal_progress = nn.Parameter(torch.zeros(num_goals))
        self.goal_priority = nn.Parameter(torch.ones(num_goals))
        self.goal_urgency = nn.Parameter(torch.zeros(num_goals))
        
        self.progress_network = nn.Sequential(
            nn.Linear(num_traits * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer("_goal_history", torch.zeros(50, num_goals))
        self.register_buffer("_history_idx", torch.tensor(0))
        
        logger.info(f"🎯 Система долгосрочных целей инициализирована: {num_goals} целей")
    
    def update_progress(self, current_traits: torch.Tensor, target_traits: torch.Tensor):
        with torch.no_grad():
            for i in range(self.num_goals):
                goal_emb = self.goal_embeddings(torch.tensor(i, device=current_traits.device))
                dist = torch.norm(current_traits - goal_emb)
                actual_progress = torch.exp(-dist)
                
                self.goal_progress[i] = 0.95 * self.goal_progress[i] + 0.05 * actual_progress
                curr_p = self.goal_progress[i].item()
                
                if curr_p > 0.85:
                    target_priority = 0.3
                elif curr_p < 0.15:
                    target_priority = 0.5
                else:
                    target_priority = 1.5
                
                self.goal_priority[i].data.lerp_(torch.tensor(target_priority, device=self.goal_priority.device), 0.1)
    
    def get_goal_influence(self, current_traits: torch.Tensor) -> torch.Tensor:
        device = current_traits.device
        total_steer = torch.zeros_like(current_traits)
        
        for i in range(self.num_goals):
            goal_emb = self.goal_embeddings(torch.tensor(i, device=device))
            desired_direction = goal_emb - current_traits
            importance = self.goal_priority[i] * (1.0 - self.goal_progress[i])
            total_steer += desired_direction * importance
            
        steer_norm = torch.norm(total_steer)
        max_force = 0.1
        if steer_norm > max_force:
            total_steer = total_steer * (max_force / steer_norm)
            
        return total_steer
    
    def get_goal_report(self) -> Dict:
        """Отчет о целях"""
        return {
            "progress": {f"goal_{i}": self.goal_progress[i].item() for i in range(self.num_goals)},
            "priority": {f"goal_{i}": self.goal_priority[i].item() for i in range(self.num_goals)},
            "urgency": {f"goal_{i}": self.goal_urgency[i].item() for i in range(self.num_goals)},
        }

# ===================================================================
# УЛУЧШЕННЫЕ ВЕСА ДЕЙСТВИЙ
# ===================================================================
class EnhancedActionWeights(nn.Module):
    """🎯 Улучшенные веса влияния действий"""   
    def __init__(self, num_traits: int, action_names: List[str] = None):
        super().__init__()
        self.num_traits = num_traits
        self.action_names = action_names or [
            "research", "explore", "reflect", "think",
            "social", "communicate", "create", "creative",
            "structure", "organize"
        ]
        self.num_actions = len(self.action_names)
        
        self.action_to_idx = {name: i for i, name in enumerate(self.action_names)}
        self.action_weights = nn.Parameter(
            torch.randn(self.num_actions, num_traits) * 0.1
        )
        
        self.register_buffer("_action_efficacy", torch.ones(self.num_actions))
        self.register_buffer("_action_usage", torch.zeros(self.num_actions))
        self.register_buffer("_last_trait_change", torch.zeros(num_traits))
        
        self.register_buffer("_efficacy_history", torch.zeros(100, self.num_actions))
        self.register_buffer("_efficacy_idx", torch.tensor(0))
        
        logger.info(f"🎯 Улучшенные веса действий: {self.num_actions} действий")
    
    def get_weights(self, action_type: str, 
                    trait_values: torch.Tensor,
                    mood_modifier: float = 1.0) -> torch.Tensor:
        """🔥 УЛУЧШЕННЫЕ адаптивные веса для действия"""
        device = trait_values.device
        weights = torch.zeros(self.num_traits, device=device)
        matched_indices = []
        
        for i, action_name in enumerate(self.action_names):
            if action_name in action_type.lower():
                matched_indices.append(i)
        
        if matched_indices:
            for idx in matched_indices:
                weights += self.action_weights[idx].to(device)
            weights = weights / len(matched_indices)
        else:
            weights = self.action_weights[0].to(device) * 0.3
        
        if matched_indices:
            current_efficacy = torch.mean(self._action_efficacy[matched_indices]).to(device)
        else:
            current_efficacy = torch.tensor(0.5, device=device)
        
        edge_proximity = 4 * trait_values * (1 - trait_values)
        adaptation = 0.4 + 0.6 * edge_proximity
        
        usage_entropy = -torch.sum(
            F.softmax(self._action_usage.float(), dim=0) * 
            torch.log(F.softmax(self._action_usage.float(), dim=0) + 1e-8)
        )
        diversity_bonus = 1.0 + torch.sigmoid(usage_entropy - 1.0) * 0.2
        
        return weights * adaptation * current_efficacy * mood_modifier * diversity_bonus
    
    def update_efficacy(self, action_type: str,
                        trait_values_before: torch.Tensor,
                        trait_values_after: torch.Tensor,
                        target_center: torch.Tensor) -> None:
        """🔥 УЛУЧШЕННОЕ обновление эффективности"""
        device = trait_values_before.device
        
        with torch.no_grad():
            dist_before = torch.abs(trait_values_before - target_center.to(device))
            dist_after = torch.abs(trait_values_after - target_center.to(device))
            improvement = torch.mean(dist_before - dist_after).item()
            
            matched_indices = []
            for i, action_name in enumerate(self.action_names):
                if action_name in action_type.lower():
                    matched_indices.append(i)
            
            if not matched_indices:
                matched_indices = [0]
            
            for idx in matched_indices:
                efficacy_gain = 0.1 + max(0.0, improvement * 15.0)
                new_efficacy = 0.9 * self._action_efficacy[idx] + 0.1 * efficacy_gain
                self._action_efficacy[idx] = torch.clamp(new_efficacy, 0.1, 2.0)
                self._action_usage[idx] += 1
            
            self._last_trait_change = trait_values_after - trait_values_before
            
            idx_val = self._efficacy_idx.item()
            self._efficacy_history[idx_val] = self._action_efficacy.clone()
            self._efficacy_idx.fill_((idx_val + 1) % 100)
    
    def get_efficacy_report(self) -> Dict:
        """Отчет об эффективности действий"""
        return {
            "efficacy": {name: self._action_efficacy[i].item() 
                        for i, name in enumerate(self.action_names)},
            "usage": {name: self._action_usage[i].item() 
                     for i, name in enumerate(self.action_names)},
            "diversity": float(-torch.sum(
                F.softmax(self._action_usage.float(), dim=0) * 
                torch.log(F.softmax(self._action_usage.float(), dim=0) + 1e-8)
            ).item())
        }

# ===================================================================
# ДИНАМИЧНАЯ ПСИХИКА v9.0 с НЕЙРОКОНТРОЛЛЕРОМ
# ===================================================================
class DynamicPsycheCoreV6(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === ИМЕНА ===
        self.trait_names = getattr(config, 'trait_names', [
            "curiosity", "depth", "empathy", "confidence", "rigidity",
            "creativity", "reflection", "ethics", "persistence",
            "arrogance", "order"
        ])
        self.drive_names = getattr(config, 'drive_names', [
            "novelty", "fatigue", "meaning", "competence", "identity", "social", "share"
        ])
        
        self.num_traits = len(self.trait_names)
        self.num_drives = len(self.drive_names)
        
        self.trait_name_to_idx = {name: i for i, name in enumerate(self.trait_names)}
        self.drive_name_to_idx = {name: i for i, name in enumerate(self.drive_names)}
        
        # === СОСТОЯНИЕ ===
        self.trait_raw = nn.Parameter(torch.zeros(self.num_traits))
        self.trait_momentum = nn.Parameter(torch.zeros(self.num_traits))
        self.drive_values = nn.Parameter(torch.ones(self.num_drives) * 0.5)
        
        # === УСИЛЕННЫЕ ПОДСИСТЕМЫ ===
        self.dynamics = DynamicCoefficientsV6(
            self.num_traits, 
            self.num_drives, 
            drive_names=self.drive_names
        )
        
        self.mood = MoodSystem(self.num_traits)
        self.goal_system = GoalSystem(self.num_traits)
        self.action_weights = EnhancedActionWeights(self.num_traits)
        
        # =============================================
        # НЕЙРОКОНТРОЛЛЕР v9.8 с ПОЛНЫМ PPO
        # =============================================
        controller_input_size = self.num_traits + self.num_drives + 1
        controller_output_size = 10
        controller_hidden = getattr(config, 'controller_hidden_dim', 32)
        
        self.controller = ImprovedPsycheController(
            controller_input_size, 
            controller_output_size,
            hidden_dim=controller_hidden
        )
        
        # Value сеть для PPO
        self.controller_value_net = nn.Sequential(
            nn.Linear(controller_input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # 🔥 ВАЖНО: Оптимизатор будет создан при первом использовании
        self.controller_optimizer = None
        self._optimizers_initialized = False
        
        # PPO параметры
        self.register_buffer('controller_reward_baseline', torch.tensor(0.0))
        self.register_buffer('controller_gamma', torch.tensor(0.95))
        self.register_buffer('controller_lambda', torch.tensor(0.95))
        self.register_buffer('controller_clip_epsilon', torch.tensor(0.2))
        self.register_buffer('controller_value_coef', torch.tensor(0.5))
        self.register_buffer('controller_entropy_coef', torch.tensor(0.01))
        self.register_buffer('_prev_controller_log_prob', torch.tensor(0.0)) 
        
        self.register_buffer('controller_loss_history', torch.zeros(100))
        self.register_buffer('controller_reward_history', torch.zeros(100))
        self.register_buffer('controller_history_idx', torch.tensor(0))

        logger.info(f"🧠 Нейроконтроллер с PPO инициализирован (оптимизатор будет создан позже)")
        
        # === ЦЕЛЕВЫЕ ДИАПАЗОНЫ ===
        self.register_buffer("_trait_target_min", torch.tensor([
            0.3, 0.4, 0.3, 0.3, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.3
        ]))
        self.register_buffer("_trait_target_max", torch.tensor([
            0.8, 0.9, 0.8, 0.8, 0.7, 0.8, 0.9, 0.8, 0.8, 0.6, 0.7
        ]))
        self.register_buffer("_drive_target_min", torch.tensor([
            0.2, 0.1, 0.2, 0.3, 0.2, 0.2, 0.1
        ]))
        self.register_buffer("_drive_target_max", torch.tensor([
            0.8, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6
        ]))
        
        # === УЛУЧШЕННЫЕ ОППОЗИЦИОННЫЕ ВЕСА ===
        self.opposition_weights = nn.Parameter(self._init_enhanced_opposition_weights())
        
        # === СЧЁТЧИКИ ===
        self.register_buffer("experiences_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_ticks", torch.tensor(0, dtype=torch.long))
        
        # === ДИНАМИЧЕСКИЕ ПАРАМЕТРЫ ===
        self.register_buffer("_radicalization", torch.tensor(1.0))
        self.register_buffer("_learning_rate_factor", torch.tensor(1.0))
        
        # === СТОХАСТИЧЕСКИЙ РЕЗОНАНС ===
        self.resonance_frequency = nn.Parameter(torch.tensor(0.015))
        self.resonance_amplitude = nn.Parameter(torch.tensor(0.0008))
        
        # === RL ПАРАМЕТРЫ ===
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("last_reward", torch.tensor(0.0))
        self.register_buffer("reward_history", torch.zeros(100))
        self.register_buffer("reward_idx", torch.tensor(0))
 
        # ===========================================================
        # 🔥 ДОБАВЛЯЕМ LayerNorm 
        # ===========================================================
        self._influence_gate_logit = nn.Parameter(torch.tensor(-3.0))
        self._influence_layer_norm = nn.LayerNorm(config.n_embd)
        self.register_buffer('_influence_gate', torch.sigmoid(self._influence_gate_logit))
 
        # ===========================================================
        # 🔥 ВЛИЯНИЕ НА HIDDEN STATES (ДИНАМИЧЕСКИЙ ВЕС, БЕЗ ГРАДИЕНТОВ)
        # ===========================================================
        num_inputs = self.num_traits + self.num_drives
        self.input_norm = nn.LayerNorm(num_inputs)
        self.psyche_to_hidden = nn.Linear(num_inputs, config.n_embd, bias=False)
        
        # 🔥 ДИНАМИЧЕСКИЙ ВЕС ВЛИЯНИЯ (обновляется через tick, НЕ через градиенты!)
        self.register_buffer("psyche_influence_dynamic", torch.tensor(0.25))  # начальное значение
        self.register_buffer("psyche_influence_target", torch.tensor(0.25))
        self.register_buffer("psyche_influence_velocity", torch.tensor(0.0))
        
        # Параметры динамики (тоже буферы, не обучаемые)
        self.register_buffer("psyche_influence_inertia", torch.tensor(0.95))
        self.register_buffer("psyche_influence_max", torch.tensor(0.45))
        self.register_buffer("psyche_influence_min", torch.tensor(0.10))
        
        logger.info(f"🧠 Динамический вес влияния: range=[0.10, 0.45], inertia=0.95 (БЕЗ ГРАДИЕНТОВ)")
        
        # ===========================================================
        # 🔥 ЗАЩИТА ОСТАЛЬНЫХ ПАРАМЕТРОВ (кроме opposition_weights)
        # ===========================================================
        
        max_seq_len = getattr(config, "n_positions", 2048)
        decay = torch.linspace(1.0, 0.4, max_seq_len)
        self.register_buffer("decay_template", decay)
        
        # Буферы для истории эмбеддингов
        self.register_buffer("psyche_embed_history", torch.zeros(config.n_embd))
        self.register_buffer("embed_change_history", torch.zeros(10))
        self.register_buffer("embed_history_idx", torch.tensor(0))
        
        # === ДОПОЛНИТЕЛЬНЫЕ РЕГИСТРЫ ===
        self.register_buffer("_last_opposition_forces", torch.zeros(self.num_traits))
        self.register_buffer("_stability_counter", torch.tensor(0))
        self.register_buffer("_extreme_correction_counter", torch.tensor(0))
        
        self.register_buffer("_prev_controller_state", torch.zeros(controller_input_size))
        self.register_buffer("_prev_controller_action", torch.zeros(controller_output_size))
        self.register_buffer("_prev_controller_reward", torch.tensor(0.0))
 
        # Регистрация буферов для repulsion loss
        self.register_buffer('_repulsion_loss_buffer', torch.tensor(0.0))
        self._current_repulsion_loss = None
 
        # === ФЛАГ ДЛЯ УПРАВЛЕНИЯ PPO ===
        self.ppo_enabled = getattr(config, 'ppo_enabled', True)  # можно отключать при заморозке 
 
        logger.info(f"🧠 Динамичная психика v9.0 с полным PPO: {self.num_traits} черт, {self.num_drives} драйвов")
    
    def _ensure_optimizers(self, device=None):
        """🔥 Создаёт оптимизатор при первом использовании на нужном устройстве"""
        if self._optimizers_initialized:
            return
        
        if device is None:
            device = next(self.controller.parameters()).device
        
        logger.info(f"🔄 Создание оптимизатора контроллера на {device}")
        
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), 
            lr=getattr(self.config, 'controller_lr', 0.001),
            weight_decay=1e-5
        )
        
        self._optimizers_initialized = True
        logger.info(f"✅ Оптимизатор контроллера создан на {device}")
    
    def _ensure_optimizer_state_device(self, device):
        """
        🔥 ГАРАНТИРУЕТ, что все состояния оптимизатора на правильном устройстве
        """
        if not hasattr(self, 'controller_optimizer') or self.controller_optimizer is None:
            return
        
        optimizer = self.controller_optimizer
        
        # Перемещаем состояния оптимизатора
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param is None:
                    continue
                
                if param.device != device:
                    param.data = param.data.to(device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(device)
                
                # Перемещаем состояния оптимизатора
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in list(state.items()):
                        if isinstance(value, torch.Tensor) and value.device != device:
                            state[key] = value.to(device)
    
    def get_memory_control_signals(self) -> Dict[str, float]:
        """🔥 ОПТИМИЗИРОВАННАЯ - все 10 сигналов для управления памятью"""
        device = next(self.parameters()).device
       
        with torch.no_grad():
            trait_values = self.get_trait_values().to(device)
            drive_values = self.get_drive_values().to(device)
            mood_state = self.mood.mood_state.to(device).unsqueeze(0) if self.mood.mood_state.dim() == 0 else self.mood.mood_state.to(device)
           
            state = torch.cat([trait_values, drive_values, mood_state]).detach()
           
            raw_output, log_prob = self.controller(
                state,
                deterministic=not self.training,
                mood=mood_state
            )
           
            if log_prob is not None:
                self._prev_controller_log_prob.data.copy_(log_prob.detach().clone())
            
            # 🔥 ОПТИМИЗИРОВАНО: один проход по numpy для всех сигналов
            raw_np = raw_output.detach().cpu().numpy()
            mood_val = float(mood_state.mean().item())
            
            decoded = {
                "curiosity_threshold": float(0.05 + (1 / (1 + np.exp(-raw_np[0]))) * 0.35),
                "recall_threshold_mod": float(0.5 + (1 / (1 + np.exp(-raw_np[1]))) * 1.0),
                "write_intensity": float(0.5 + np.log1p(np.exp(raw_np[2])) * 2.0),
                "noise_mod": float(0.8 + np.log1p(np.exp(raw_np[3])) * 1.0),
                "negative_sensitivity": float(0.5 + (1 / (1 + np.exp(-raw_np[4]))) * 1.0),
                "exploration_bonus": float(np.log1p(np.exp(raw_np[5])) * 0.4),
                "learning_rate_mult": float(0.7 + (1 / (1 + np.exp(-raw_np[6]))) * 0.8),
               
                "hebb_write_gate": float(1 / (1 + np.exp(-raw_np[7]))),
                "hebb_lr_mult": float(0.3 + np.log1p(np.exp(raw_np[8])) * 0.7),
                "stability_factor": float((1 / (1 + np.exp(-raw_np[9]))) * 0.5),
               
                "attention_temperature": float(1.5 + np.log1p(np.exp(raw_np[8])) * 2.5),
                "mood_influence": mood_val,
            }
           
            decoded["creativity_level"] = float(trait_values[5] if len(trait_values) > 5 else 0.5)
            decoded["curiosity_level"] = float(trait_values[0] if len(trait_values) > 0 else 0.5)
           
            return decoded
    
    # ===================================================================
    #  update_controller
    # ===================================================================
    def update_controller(self, reward: float, done: bool = False):
        """Обновление контроллера с проверкой обучаемости"""
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        
        if abs(reward) < 0.001 or not self.training:
            return
        
        # 🔥 ПРОВЕРКА: есть ли обучаемые параметры в контроллере
        has_trainable_params = any(p.requires_grad for p in self.controller.parameters())
        if not has_trainable_params:
            return
        
        device = next(self.controller.parameters()).device
        self._ensure_optimizers(device)  
        
        current_state = torch.cat([
            self.get_trait_values().to(device), 
            self.get_drive_values().to(device), 
            self.mood.mood_state.to(device).view(1)
        ]).to(device)
        
        self._prev_controller_state.data.copy_(current_state.detach().clone())
        
        state = self._prev_controller_state.to(device).clone()
        action_raw = self.controller.last_action_raw.to(device).clone()
        old_log_prob = self.controller.last_log_prob.to(device).clone()
        
        with torch.no_grad():
            current_value = self.controller.get_value(state.unsqueeze(0)).squeeze()
            
            if not done:
                next_state = torch.cat([
                    self.get_trait_values().to(device), 
                    self.get_drive_values().to(device), 
                    self.mood.mood_state.to(device).view(1)
                ]).to(device)
                next_value = self.controller.get_value(next_state.unsqueeze(0)).squeeze()
            else:
                next_state = torch.zeros_like(state)
                next_value = torch.tensor(0.0, device=device)
        
        self.controller.add_to_batch_buffer(
            state=state,
            action_raw=action_raw,
            old_log_prob=old_log_prob,
            reward=reward,
            next_state=next_state,
            done=done,
            value=current_value
        )
        
        if self.controller.should_update():
            self._perform_ppo_update()
    
    def _perform_ppo_update(self):
        """
        🔥 PPO UPDATE - ИСПРАВЛЕННАЯ ВЕРСИЯ с проверкой обучаемости параметров
        """
        device = next(self.controller.parameters()).device
        
        # 🔥 КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли обучаемые параметры в контроллере?
        has_trainable_params = any(p.requires_grad for p in self.controller.parameters())
        
        if not has_trainable_params:
            # Все параметры контроллера заморожены — пропускаем PPO update
            if self.total_ticks.item() % 100 == 0:
                logger.debug("⏭️ PPO update пропущен: все параметры контроллера заморожены")
            self.controller.clear_batch_buffer()
            return
        
        # Убеждаемся, что оптимизатор существует
        if not self._optimizers_initialized:
            self._ensure_optimizers(device)
        
        # Получаем батч
        states, action_raws, old_log_probs, rewards, next_states, dones, _ = \
            self.controller.get_minibatch()
        
        if len(states) == 0:
            return
        
        # Перемещаем всё на device
        states = states.to(device)
        action_raws = action_raws.to(device)
        old_log_probs = old_log_probs.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        # ===========================================================
        # 🔥 ВЫЧИСЛЕНИЕ ADVANTAGES И TD TARGETS
        # ===========================================================
        with torch.no_grad():
            current_values = self.controller.get_value(states).squeeze(-1)
            next_values = self.controller.get_value(next_states).squeeze(-1)
            
            td_targets = rewards + self.controller_gamma * next_values * (1.0 - dones.float())
            td_targets = torch.clamp(td_targets, -10.0, 10.0)
            
            advantages = torch.zeros_like(td_targets)
            last_gae = 0.0
            
            for t in reversed(range(len(td_targets))):
                delta = td_targets[t] - current_values[t]
                advantages[t] = last_gae = delta + self.controller_gamma * self.controller_lambda * (1.0 - dones[t].float()) * last_gae
            
            if len(advantages) > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std
                advantages = torch.clamp(advantages, -5.0, 5.0)
        
        advantages = advantages.detach()
        td_targets = td_targets.detach()
        
        # ===========================================================
        # 🔥 PPO K-EPOCHS
        # ===========================================================
        k_epochs = 3
        
        for epoch in range(k_epochs):
            self.controller_optimizer.zero_grad()
            
            # Вычисляем текущие value
            current_values = self.controller.get_value(states).squeeze(-1)
            value_loss = F.mse_loss(current_values, td_targets)
            value_loss = torch.clamp(value_loss, 0.0, 100.0)
            
            # Вычисляем текущие log probabilities
            new_log_probs = self.controller.compute_log_prob_current(states, action_raws)
            new_log_probs = torch.nan_to_num(new_log_probs, nan=-10.0, posinf=10.0, neginf=-10.0)
            old_log_probs_clamped = torch.clamp(old_log_probs, -20.0, 20.0)
            
            # Вычисляем ratio
            log_ratio = new_log_probs - old_log_probs_clamped
            log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
            ratio = torch.exp(log_ratio)
            ratio = torch.clamp(ratio, 0.0, 20.0)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.controller_clip_epsilon, 
                                1.0 + self.controller_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            current_dist = self.controller.get_current_distribution(states)
            entropy = current_dist.entropy().mean()
            entropy = torch.clamp(entropy, 0.0, 10.0)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            total_loss = torch.clamp(total_loss, -100.0, 100.0)
            
            # 🔥 КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли градиентная связь?
            if total_loss.requires_grad and has_trainable_params:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
                self.controller_optimizer.step()
            else:
                # Нет градиентной связи — пропускаем шаг
                if epoch == 0 and self.total_ticks.item() % 100 == 0:
                    logger.debug(f"⏭️ PPO step {epoch} пропущен: loss.requires_grad={total_loss.requires_grad}")
                break
        
        # Очищаем буфер
        self.controller.clear_batch_buffer()
        
        # Обновляем baseline
        with torch.no_grad():
            self.controller_reward_baseline = 0.99 * self.controller_reward_baseline + 0.01 * rewards.mean()
        
    def get_controller_report(self) -> Dict:
        """Отчет о состоянии контроллера"""
        report = {
            "loss_history": {
                "mean": float(self.controller_loss_history.mean().item()),
                "std": float(self.controller_loss_history.std().item()),
                "min": float(self.controller_loss_history.min().item()),
                "max": float(self.controller_loss_history.max().item()),
                "recent": [float(x) for x in self.controller_loss_history[-10:]]
            },
            "reward_history": {
                "mean": float(self.controller_reward_history.mean().item()),
                "std": float(self.controller_reward_history.std().item()),
                "min": float(self.controller_reward_history.min().item()),
                "max": float(self.controller_reward_history.max().item()),
                "recent": [float(x) for x in self.controller_reward_history[-10:]]
            },
            "baseline": float(self.controller_reward_baseline.item()),
            "exploration_rate": float(self.controller.exploration_rate.item()),
            "training_steps": int(self.controller.training_steps.item()),
            "experience_buffer_size": len(self.controller.batch_buffer),
            "value_net_stats": {
                "weight_mean": float(self.controller.value_net[0].weight.mean().item()),
                "weight_std": float(self.controller.value_net[0].weight.std().item()),
                "bias_mean": float(self.controller.value_net[0].bias.mean().item()),
            },
            "ppo_params": {
                "gamma": float(self.controller_gamma.item()),
                "lambda": float(self.controller_lambda.item()),
                "clip_epsilon": float(self.controller_clip_epsilon.item()),
                "value_coef": float(self.controller_value_coef.item()),
                "entropy_coef": float(self.controller_entropy_coef.item())
            }
        }
        
        if self.training:
            controller_norm = 0.0
            
            for p in self.controller.parameters():
                if p.grad is not None:
                    controller_norm += p.grad.data.norm(2).item() ** 2
            
            report["gradient_norms"] = {
                "controller": controller_norm ** 0.5 if controller_norm > 0 else 0.0
            }
        
        return report
    
    def calculate_intrinsic_reward(self, prev_drives: torch.Tensor, 
                                  current_drives: torch.Tensor, 
                                  user_feedback: float = 0.0,
                                  memory_surprise: float = 0.0) -> float:
        """🔥 ОПТИМИЗИРОВАННАЯ контрастная награда"""
        with torch.no_grad():
            if torch.isnan(current_drives).any() or torch.isinf(current_drives).any():
                return 0.0
            
            safe_prev = torch.clamp(prev_drives, 0.0, 1.0)
            safe_current = torch.clamp(current_drives, 0.0, 1.0)
            
            # 🔥 ОПТИМИЗИРОВАНО: векторизованные вычисления
            drive_delta = torch.mean(torch.abs(safe_prev - safe_current)).item()
            movement_bonus = drive_delta * 3.0
            
            target_min = self._drive_target_min.to(current_drives.device)
            target_max = self._drive_target_max.to(current_drives.device)
            
            dist_to_target = torch.zeros_like(safe_current)
            below = safe_current < target_min
            above = safe_current > target_max
            dist_to_target[below] = target_min[below] - safe_current[below]
            dist_to_target[above] = safe_current[above] - target_max[above]
            
            target_distance = dist_to_target.mean().item()
            target_bonus = -target_distance * 2.0
            
            # 🔥 ОПТИМИЗИРОВАНО: векторизованный расчет штрафа
            extreme_penalty = torch.zeros(1, device=safe_current.device)
            
            low_mask = safe_current < 0.15
            high_mask = safe_current > 0.85
            mid_mask = (safe_current >= 0.3) & (safe_current <= 0.7)
            
            if low_mask.any():
                extreme_penalty += ((0.15 - safe_current[low_mask]) * 5.0).sum()
            if high_mask.any():
                extreme_penalty += ((safe_current[high_mask] - 0.85) * 5.0).sum()
            if mid_mask.any():
                extreme_penalty -= 0.1 * mid_mask.sum().float()
            
            extreme_penalty_val = extreme_penalty.item()
            
            external_bonus = user_feedback * 2.0 + memory_surprise * 2.0
            randomness = torch.randn(1).item() * 0.1
            
            # ИСПРАВЛЕНИЕ: правильный знак для штрафа
            total_reward = (
                movement_bonus * 0.2 +
                target_bonus * 0.3 -
                extreme_penalty_val * 0.3 +  # МИНУС, а не +(-...)
                external_bonus * 0.2 +
                randomness
            )
            
            return max(-2.0, min(2.0, total_reward))
    
    def tick(self, training: bool = False):
        self.total_ticks.add_(1)
        device = self.trait_raw.device
        exp_count = self.experiences_count.item()
        
        with torch.set_grad_enabled(training):
            self._core_tick_logic(device, exp_count)
        
        # ===========================================================
        # 🔥 ИСПРАВЛЕНИЕ: Вычисление repulsion loss ПРИ ОБУЧЕНИИ
        # ===========================================================
        if training:
            try:
                rep_loss = self.compute_repulsion_loss()
                
                if rep_loss is not None:
                    self._repulsion_loss_buffer.data.copy_(rep_loss.detach())
                    self._current_repulsion_loss = rep_loss
                    
                    if self.total_ticks.item() % 200 == 0:
                        logger.debug(f"🔄 Repulsion loss: {rep_loss.item():.6f}")
                    
            except Exception as e:
                logger.debug(f"⚠ Ошибка вычисления repulsion loss: {e}")
                self._current_repulsion_loss = None
        
        # ===========================================================
        # 🔥 ДИНАМИЧЕСКАЯ РЕГУЛИРОВКА СИЛЫ ВЛИЯНИЯ
        # ===========================================================
        with torch.no_grad():
            stability = getattr(self.dynamics, '_system_stability', torch.tensor(0.5)).item()
            entropy = getattr(self.dynamics, '_system_entropy', torch.tensor(0.5)).item()
            mood_abs = abs(self.mood.mood_state.item())
            
            target = 0.25
            
            if stability < 0.3:
                target -= 0.08
            elif stability > 0.7:
                target += 0.05
            
            if entropy > 0.7:
                target -= 0.06
            elif entropy < 0.3:
                target += 0.04
            
            target += mood_abs * 0.03
            
            if hasattr(self, 'psyche_influence_min'):
                target = max(self.psyche_influence_min.item(), 
                            min(self.psyche_influence_max.item(), target))
            
            if not hasattr(self, 'psyche_influence_target'):
                self.register_buffer('psyche_influence_target', torch.tensor(target))
                self.register_buffer('psyche_influence_dynamic', torch.tensor(target))
                self.register_buffer('psyche_influence_velocity', torch.tensor(0.0))
                self.register_buffer('psyche_influence_inertia', torch.tensor(0.85))
                self.register_buffer('psyche_influence_min', torch.tensor(0.10))
                self.register_buffer('psyche_influence_max', torch.tensor(0.45))
            
            self.psyche_influence_target.data = (
                self.psyche_influence_target * 0.95 + target * 0.05
            )
            
            acceleration = (self.psyche_influence_target - self.psyche_influence_dynamic) * 0.1
            self.psyche_influence_velocity.data = (
                self.psyche_influence_velocity * self.psyche_influence_inertia + acceleration
            )
            self.psyche_influence_dynamic.data += self.psyche_influence_velocity
            self.psyche_influence_dynamic.data.clamp_(
                self.psyche_influence_min.item(), 
                self.psyche_influence_max.item()
            )
            
            if self.total_ticks.item() % 200 == 0:
                logger.info(f"🎛️ Динамическая сила влияния: target={self.psyche_influence_target.item():.3f}, "
                           f"current={self.psyche_influence_dynamic.item():.3f}, "
                           f"stability={stability:.2f}, entropy={entropy:.2f}, mood={mood_abs:.2f}")
        
        # ===========================================================
        # 🔥 PPO ОБНОВЛЕНИЕ ТОЛЬКО ВНУТРИ _core_tick_logic
        # ===========================================================
        # Удаляем дублирующее PPO обновление здесь, так как оно уже происходит
        # внутри _core_tick_logic через update_controller
        
        return self.get_memory_control_signals() if hasattr(self, 'get_memory_control_signals') else {}
    
    def _core_tick_logic(self, device, exp_count):
        if torch.isnan(self.trait_raw).any() or torch.isinf(self.trait_raw).any():
            logger.warning("⚠ NaN/Inf в trait_raw, сбрасываю состояние психики")
            with torch.no_grad():
                self.trait_raw.data.zero_()
                self.trait_momentum.data.zero_()
                self.mood.mood_state.data.zero_()
                self.mood.mood_velocity.data.zero_()
                self.trait_raw.normal_(mean=0.0, std=0.15)
            return
        
        trait_values = self.get_trait_values().to(device)
        drive_values = self.get_drive_values().to(device)
        
        if torch.isnan(trait_values).any() or torch.isinf(trait_values).any():
            logger.warning("⚠ NaN/Inf в trait_values после вычисления, сбрасываю")
            with torch.no_grad():
                self.trait_raw.data.zero_()
                self.mood.mood_state.data.zero_()
                self.mood.mood_velocity.data.zero_()
            return
        
        self.dynamics.update_history(trait_values, drive_values)
        
        entropy = self.dynamics._system_entropy.item()
        stability = self.dynamics._system_stability.item()
        burnout_protection_active = self._apply_anti_burnout_protection(entropy, stability)
        
        target_center = (self._trait_target_min.to(device) + self._trait_target_max.to(device)) / 2
        self.goal_system.update_progress(trait_values, target_center)
        
        base_radical = 1.0 + exp_count / 25000
        entropy_factor = 1.0 - entropy * 0.3
        self._radicalization.data.copy_(
            torch.clamp(torch.tensor(base_radical * entropy_factor, device=device), 1.0, 3.0)
        )
        
        opposition_forces = self._compute_enhanced_opposition_forces(trait_values)
        tension_level = torch.abs(opposition_forces).mean().item()
        
        self.mood.update(trait_values, tension_level)
        mood_mods = self.mood.get_modifiers()
        
        lr_factor = self._adjust_learning_rate_factor()
        
        plasticity = self.dynamics.compute_plasticity(
            trait_values, exp_count, mood_mods["plasticity_mult"] * lr_factor, drive_values
        )
        resistance = self.dynamics.compute_resistance(
            trait_values, self._trait_target_min, self._trait_target_max,
            mood_mods["resistance_mult"] * lr_factor, drive_values
        )
        
        if burnout_protection_active:
            noise_scale = 0.3
        else:
            noise_scale = 1.0
            
        noise = self._apply_enhanced_stochastic_resonance(trait_values, plasticity) * noise_scale
        noise = noise * mood_mods["noise_mult"]
        
        homeostasis_forces = self._compute_enhanced_homeostasis_forces(trait_values)
        
        drive_influence = torch.matmul(
            drive_values.unsqueeze(0),
            self.dynamics.drive_to_trait.to(device)
        ).squeeze(0) * 0.002
        
        goal_influence = self.goal_system.get_goal_influence(trait_values)
        
        total_force = (
            noise +
            opposition_forces +
            homeostasis_forces +
            drive_influence +
            goal_influence
        )
        
        effective_force = total_force * plasticity * (1.0 - resistance)
        
        if torch.isnan(effective_force).any() or torch.isinf(effective_force).any():
            effective_force = torch.zeros_like(effective_force)
        
        inertia = 0.92 - torch.mean(resistance).item() * 0.4
        new_momentum = self.trait_momentum.data * inertia + effective_force
        
        max_momentum = 0.045 * torch.mean(plasticity)
        new_momentum = torch.tanh(new_momentum / (max_momentum + 1e-6)) * max_momentum
        
        self.trait_momentum.data.copy_(new_momentum)
        
        self.trait_raw.data.add_(new_momentum)
        self.trait_raw.data.copy_(torch.tanh(self.trait_raw.data / 3.0) * 3.0)
        
        if self.total_ticks.item() % 100 == 0:
            trait_values = self.get_trait_values()
            if "curiosity" in self.trait_name_to_idx and "creativity" in self.trait_name_to_idx:
                c1 = trait_values[self.trait_name_to_idx["curiosity"]]
                c2 = trait_values[self.trait_name_to_idx["creativity"]]
                diff = abs(c1 - c2)
                if diff > 0.2:
                    correction = (c2 - c1) * 0.01
                    self.trait_raw.data[self.trait_name_to_idx["curiosity"]] += correction * 0.3
        
        trait_values = self.get_trait_values()
        for i in range(self.num_traits):
            value = trait_values[i].item()
            if value > 0.85 or value < 0.15:
                correction = (0.5 - value) * 0.05
                self.trait_raw.data[i] = self.trait_raw.data[i] + correction
        
        self._apply_extreme_correction(trait_values)
        
        prev_drives = self.get_drive_values().detach().clone()
        
        drive_dynamics = self.dynamics.compute_drive_dynamics(drive_values, trait_values)
        self.drive_values.data.add_(drive_dynamics["net_change"])
        self.drive_values.data.copy_(torch.sigmoid((self.drive_values.data - 0.5) * 3.5))
        
        if not burnout_protection_active:
            noise_scale = 0.015 * (1.0 - stability)
            drive_noise = torch.randn_like(self.drive_values) * noise_scale
            self.drive_values.data.add_(drive_noise)
            
        self.drive_values.data.clamp_(0.05, 0.95)
        
        current_drives = self.get_drive_values().detach()
        reward = self.calculate_intrinsic_reward(prev_drives, current_drives)
        
        # 🔥 ПРОВЕРКА: обновляем контроллер только если его параметры обучаемы
        if hasattr(self, 'controller') and self.controller is not None:
            # Проверяем, есть ли обучаемые параметры в контроллере
            has_trainable = any(p.requires_grad for p in self.controller.parameters())
            
            if has_trainable:
                self.update_controller(reward, done=False)
            elif self.total_ticks.item() % 200 == 0:
                logger.debug("⏭️ update_controller пропущен: параметры контроллера заморожены")
        
        self.apply_reinforcement(reward, learning_rate=0.006)
        
        with torch.no_grad():
            self.opposition_weights.data.clamp_(min=-2.5, max=2.5)
            
            for i in range(self.num_traits):
                for j in range(i+1, self.num_traits):
                    if abs(self.opposition_weights[i, j].item()) > 0.5:
                        avg = (self.opposition_weights[i, j] + self.opposition_weights[j, i]) / 2
                        self.opposition_weights[i, j] = avg
                        self.opposition_weights[j, i] = avg
            
            self.opposition_weights.data.fill_diagonal_(0.0)
            
            trait_change = torch.norm(new_momentum).item()
            if trait_change < 0.0005:
                self._stability_counter.add_(1)
                if self._stability_counter.item() > 30:
                    perturbation = torch.randn_like(self.trait_raw) * 0.02
                    self.trait_raw.data.add_(perturbation)
                    self._stability_counter.zero_()
            else:
                self._stability_counter.data.clamp_(max=self._stability_counter.item() - 2)              
    
    def _compute_enhanced_homeostasis_forces(self, trait_values: torch.Tensor) -> torch.Tensor:
        device = trait_values.device
        
        target_center = (self._trait_target_min.to(device) + self._trait_target_max.to(device)) / 2
        target_width = (self._trait_target_max.to(device) - self._trait_target_min.to(device)) / 2
        
        displacement = trait_values - target_center
        normalized_dist = displacement / (target_width + 1e-6)
        
        inside = torch.abs(normalized_dist) < 1.0
        power_factor = 1.0 + torch.abs(normalized_dist) * 0.7
        
        force = torch.where(
            inside,
            -displacement ** 3 * 0.08 * power_factor,
            -torch.sign(normalized_dist) * (torch.exp(torch.abs(normalized_dist) - 0.7) - 0.7) * 0.25
        )
        
        extreme_mask = torch.abs(normalized_dist) > 1.3
        if extreme_mask.any():
            impulse = -displacement * 0.005 * torch.abs(normalized_dist) * extreme_mask.float()
            force = force + impulse
        
        return force
 
    def compute_repulsion_loss(self) -> torch.Tensor:
        """Предотвращает semantic collapse черт личности"""
        if not hasattr(self, 'trait_raw'):
            return torch.tensor(0.0, device=self.trait_raw.device)
        
        trait_values = self.get_trait_values()
        
        # Нормализуем
        traits_norm = F.normalize(trait_values.unsqueeze(0), dim=-1).squeeze(0)
        
        # Матрица сходств
        sim_matrix = torch.matmul(traits_norm.unsqueeze(0), traits_norm.unsqueeze(1)).squeeze()
        
        # Убираем диагональ
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        
        # Поощряем ортогональность
        repulsion = (sim_matrix[mask] ** 2).mean()
        
        return repulsion * 0.001  # Очень маленький вес
 
    def _apply_extreme_correction(self, trait_values: torch.Tensor):
        """🔥 ОПТИМИЗИРОВАННАЯ экстремальная коррекция"""
        with torch.no_grad():
            device = trait_values.device
            target_min = self._trait_target_min.to(device)
            target_max = self._trait_target_max.to(device)
            
            # 🔥 ОПТИМИЗИРОВАНО: векторизованная коррекция без циклов
            correction = torch.zeros_like(self.trait_raw)
            
            # Случай 1: превышение максимума
            over_mask = trait_values > target_max * 1.1
            if over_mask.any():
                over_factor = (trait_values[over_mask] - target_max[over_mask]) / target_max[over_mask]
                correction[over_mask] = -(over_factor * 0.25)
            
            # Случай 2: ниже минимума
            under_mask = trait_values < target_min * 0.9
            if under_mask.any():
                under_factor = (target_min[under_mask] - trait_values[under_mask]) / target_min[under_mask]
                correction[under_mask] = under_factor * 0.25
            
            # Случай 3: экстремальные значения
            extreme_mask = (trait_values > 0.95) | (trait_values < 0.05)
            extreme_mask = extreme_mask & ~over_mask & ~under_mask  # не пересекаться с первыми двумя
            if extreme_mask.any():
                correction[extreme_mask] = (0.5 - trait_values[extreme_mask]) * 0.15
            
            if correction.abs().sum() > 0:
                smoothing = 0.4
                self.trait_raw.data.add_(correction * smoothing)
                self._extreme_correction_counter.add_(1)
    
    def _apply_anti_burnout_protection(self, entropy: float, stability: float):
        """🔥 Защита от выгорания (вынесено в отдельный метод)"""
        with torch.no_grad():
            if entropy > 0.8:
                self.temperature.data.mul_(0.9).clamp_(min=0.5)
                self._radicalization.mul_(0.85).clamp_(min=1.0, max=2.5)
                return True
            elif entropy > 0.6 and stability < 0.3:
                self.temperature.data.mul_(0.95).clamp_(min=0.7)
                return True
            
            return False
    
    def _compute_enhanced_opposition_forces(self, trait_values: torch.Tensor) -> torch.Tensor:
        device = trait_values.device
        W = self.opposition_weights.to(device)
        
        entropy = self.dynamics._system_entropy.to(device)
        stability = self.dynamics._system_stability.to(device)
        
        opp_strength = 0.08 * (1.0 + entropy * 0.4) * (1.0 - stability * 0.3)
        syn_strength = 0.04 * (1.0 - entropy * 0.3) * stability
        
        W_neg = torch.clamp(W, max=0)
        W_pos = torch.clamp(W, min=0)
        
        force_opp = torch.mv(W_neg, trait_values) * opp_strength
        partner_pull = torch.mv(W_pos, trait_values)
        self_weight = torch.sum(W_pos, dim=1)
        force_syn = (partner_pull - self_weight * trait_values) * syn_strength
        
        total_force = force_opp + force_syn
        self._last_opposition_forces.data.copy_(total_force.detach().clone())
        
        return total_force
    
    def _apply_enhanced_stochastic_resonance(self, trait_values: torch.Tensor,
                                             plasticity: torch.Tensor) -> torch.Tensor:
        device = trait_values.device
        ticks = float(self.total_ticks.item())
        
        noise_level = plasticity * self.resonance_amplitude * 15
        base_noise = torch.randn(self.num_traits, device=device) * noise_level
        
        uncertainty = 4 * trait_values * (1 - trait_values)
        
        oscillation1 = torch.sin(torch.tensor(ticks * self.resonance_frequency.item(), device=device))
        oscillation2 = torch.sin(torch.tensor(ticks * self.resonance_frequency.item() * 2.5, device=device))
        oscillation3 = torch.sin(torch.tensor(ticks * self.resonance_frequency.item() * 1.7, device=device))
        
        resonance = uncertainty * (oscillation1 * 0.5 + oscillation2 * 0.3 + oscillation3 * 0.2) * self.resonance_amplitude * 1.5
        
        return base_noise + resonance
    
    def _adjust_learning_rate_factor(self):
        entropy = self.dynamics._system_entropy.item()
        stability = self.dynamics._system_stability.item()
        
        if entropy > 0.7:
            lr_factor = 1.8
        elif entropy < 0.2 and stability > 0.8:
            lr_factor = 2.5
        elif stability < 0.2:
            lr_factor = 0.6
        elif entropy > 0.8:
            lr_factor = 0.4
        else:
            lr_factor = 1.0
        
        self._learning_rate_factor.data.fill_(lr_factor)
        return lr_factor
    
    def _init_enhanced_opposition_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.num_traits, self.num_traits)
        idx = self.trait_name_to_idx
        
        oppositions = [
            ("curiosity", "rigidity", -2.0),
            ("empathy", "arrogance", -1.8),
            ("creativity", "order", -1.5),
            ("depth", "rigidity", -1.0),
        ]
        for t1, t2, w in oppositions:
            if t1 in idx and t2 in idx:
                weights[idx[t1], idx[t2]] = w
                weights[idx[t2], idx[t1]] = w
        
        synergies = [
            ("curiosity", "creativity", 1.4),
            ("depth", "reflection", 1.5),
            ("empathy", "ethics", 2.0),
            ("confidence", "persistence", 1.5),
            ("curiosity", "depth", 1.2),
        ]
        for t1, t2, w in synergies:
            if t1 in idx and t2 in idx:
                weights[idx[t1], idx[t2]] = w
                weights[idx[t2], idx[t1]] = w
        
        return weights
    
    def _dynamic_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        steepness = 2.2 * self._radicalization
        return 1 / (1 + torch.exp(-steepness * x))
    
    def get_trait_values(self) -> torch.Tensor:
        values = self._dynamic_sigmoid(self.trait_raw)
        values = torch.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
        return torch.clamp(values, 0.0, 1.0)

    def get_repulsion_loss(self) -> Optional[torch.Tensor]:
        """
        🔥 Возвращает repulsion loss для интеграции в общий loss модели
        ВОЗВРАЩАЕТ ТЕНЗОР С ГРАДИЕНТАМИ! Не detached!
        """
        if hasattr(self, '_current_repulsion_loss') and self._current_repulsion_loss is not None:
            return self._current_repulsion_loss
        return None

    def get_drive_values(self) -> torch.Tensor:
        values = torch.clamp(self.drive_values, 0.0, 1.0)
        values = torch.nan_to_num(values, nan=0.5)
        return values
    
    def ingest_memory_experience(self, surprise_value: float, effort_value: float, recall_success: float = 0.0):
        with torch.no_grad():
            surprise_value = min(1.0, max(0.0, surprise_value))
            effort_value = min(1.0, max(0.0, effort_value))
            recall_success = min(1.0, max(0.0, recall_success))
            
            current_fatigue = self.drive_values[self.drive_name_to_idx.get("fatigue", 1)].item()
            current_novelty = self.drive_values[self.drive_name_to_idx.get("novelty", 0)].item()
            fatigue_factor = max(0.3, 1.0 - current_fatigue * 0.7)
            novelty_factor = max(0.2, current_novelty)
            mood_factor = max(0.5, 1.0 - abs(self.mood.mood_state.item()) * 0.5)
            
            new_drive_values = self.drive_values.clone()
            
            novelty_idx = self.drive_name_to_idx.get("novelty", 0)
            novelty_satisfaction = surprise_value * 0.35 * novelty_factor
            new_drive_values[novelty_idx] = new_drive_values[novelty_idx] - novelty_satisfaction
            
            fatigue_idx = self.drive_name_to_idx.get("fatigue", 1)
            fatigue_cost = (0.0004 + (effort_value * 0.1)) * fatigue_factor
            
            if effort_value > 0.7:
                extra_fatigue = effort_value * 0.15
                fatigue_cost += extra_fatigue * 0.3
            
            new_drive_values[fatigue_idx] = new_drive_values[fatigue_idx] + fatigue_cost
            
            meaning_idx = self.drive_name_to_idx.get("meaning", -1)
            if meaning_idx == -1: 
                meaning_idx = self.drive_name_to_idx.get("competence", 3)
            
            meaning_satisfaction = (recall_success * 0.5) + (surprise_value * 0.1)
            new_drive_values[meaning_idx] = new_drive_values[meaning_idx] - meaning_satisfaction
            
            max_change = 0.08
            change = new_drive_values - self.drive_values
            change_norm = torch.norm(change)
            if change_norm > max_change:
                scale = max_change / (change_norm + 1e-8)
                new_drive_values = self.drive_values + change * scale
            
            new_drive_values = torch.clamp(new_drive_values, 0.05, 0.95)
            self.drive_values.data.copy_(new_drive_values)
            
            if surprise_value > 0.2:
                joy_boost = surprise_value * 0.008 * mood_factor
                current_velocity = self.mood.mood_velocity.clone()
                new_velocity = current_velocity + joy_boost
                new_velocity = torch.clamp(new_velocity, -0.02, 0.02)
                self.mood.mood_velocity.data.copy_(new_velocity)
    
    def apply_reinforcement(self, reward: float, learning_rate: float = 0.01):
        with torch.no_grad():
            current_temp = self.temperature.item()
            target_temp = 1.0
            
            if abs(reward) < 0.01:
                new_temp = current_temp + (target_temp - current_temp) * 0.2
            elif reward > 0.01:
                cooling = 0.9 + (0.05 * min(1.0, reward * 10))
                new_temp = max(0.6, current_temp * cooling)
            else:
                heating = 1.1 + (0.05 * min(1.0, abs(reward) * 10))
                new_temp = min(1.8, current_temp * heating)
            
            if abs(new_temp - target_temp) > 0.5:
                new_temp = new_temp + (target_temp - new_temp) * 0.1
            
            self.temperature.data.fill_(new_temp)
            
            # УДАЛЕН мёртвый код с opposition_weights.grad
            # Вместо этого используем случайное блуждание
            if reward > 0.015:
                # Прямое обновление opposition_weights без градиентов
                update_scale = min(2.5, 1.0 + reward * 4.0)
                # Случайное блуждание вместо градиентов
                noise = torch.randn_like(self.opposition_weights) * learning_rate * update_scale * 0.1
                self.opposition_weights.data.add_(noise)
                self.opposition_weights.data.clamp_(min=-2.5, max=2.5)
    
    def process_experience(self, action_type: str, intensity: float = 0.5):
        self.experiences_count.add_(1)
        device = self.trait_raw.device
        
        with torch.no_grad():
            trait_values_before = self.get_trait_values().to(device)
            mood_mods = self.mood.get_modifiers()
            
            weights = self.action_weights.get_weights(
                action_type, trait_values_before, mood_mods["learning_rate_mult"]
            )
            
            drive_values = self.get_drive_values().to(device)
            plasticity = self.dynamics.compute_plasticity(
                trait_values_before, self.experiences_count.item(), 
                mood_mods["plasticity_mult"], drive_values
            )
            
            impulse = weights * intensity * plasticity * 0.03
            
            if "creativity" in action_type.lower() and "curiosity" in self.trait_names:
                curiosity_idx = self.trait_name_to_idx["curiosity"]
                soft_boost = impulse.mean() * 0.1
                self.trait_momentum.data[curiosity_idx] += soft_boost
            
            if "research" in action_type.lower() or "explore" in action_type.lower():
                curiosity_idx = self.trait_name_to_idx.get("curiosity")
                creativity_idx = self.trait_name_to_idx.get("creativity")
                if curiosity_idx is not None and creativity_idx is not None:
                    synergy_boost = intensity * 0.02
                    self.trait_momentum.data[curiosity_idx] += synergy_boost
                    self.trait_momentum.data[creativity_idx] += synergy_boost * 0.8
            
            if torch.isnan(impulse).any() or torch.isinf(impulse).any():
                logger.warning("⚠ NaN/Inf в impulse, пропускаю")
                return
            
            prev_drives = self.get_drive_values().detach().clone()
            self.trait_momentum.data.add_(impulse)
            self.tick()
            
            trait_values_after = self.get_trait_values().to(device)
            target_center = (self._trait_target_min.to(device) + self._trait_target_max.to(device)) / 2
            
            self.action_weights.update_efficacy(
                action_type, trait_values_before, trait_values_after, target_center
            )
            
            current_drives = self.get_drive_values().detach()
            reward = self.calculate_intrinsic_reward(prev_drives, current_drives)
            
            self.update_controller(reward)
            self.apply_reinforcement(reward, learning_rate=0.012)
                         
    def influence_hidden_states(self, hidden_states) -> torch.Tensor:
        """Влияние психики на hidden_states"""
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or tuple, got {type(hidden_states)}")
        
        original_dim = hidden_states.dim()
        was_2d = (original_dim == 2)
        
        if was_2d:
            hidden_states = hidden_states.unsqueeze(1)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        trait_tensors = self.get_trait_values()
        drive_tensors = self.get_drive_values()
        
        if torch.isnan(trait_tensors).any() or torch.isnan(drive_tensors).any():
            return hidden_states.squeeze(1) if was_2d else hidden_states
        
        with torch.no_grad():
            input_vec = torch.cat([trait_tensors, drive_tensors], dim=0)
            input_vec = self.input_norm(input_vec)
            psyche_embed_raw = torch.tanh(self.psyche_to_hidden(input_vec))
        
        psyche_embed = psyche_embed_raw.detach()
        
        prev_embed = self.psyche_embed_history.detach().to(device)
        embed_change = torch.norm(psyche_embed - prev_embed).item()
        
        idx = self.embed_history_idx.item()
        self.embed_change_history[idx] = embed_change
        self.embed_history_idx.fill_((idx + 1) % 10)
        
        avg_change = torch.mean(self.embed_change_history).item()
        
        if avg_change < 0.08:
            smoothing = 0.92
        elif avg_change < 0.25:
            smoothing = 0.75
        else:
            smoothing = 0.55
        
        psyche_embed = smoothing * prev_embed + (1 - smoothing) * psyche_embed
        psyche_embed = F.normalize(psyche_embed, dim=0) * torch.norm(psyche_embed, dim=0).mean()
        psyche_embed = torch.tanh(psyche_embed)
        
        self.psyche_embed_history.data.copy_(psyche_embed.detach())
        
        if seq_len <= len(self.decay_template):
            decay = self.decay_template[:seq_len].to(device)
        else:
            decay = torch.linspace(1.0, 0.3, seq_len, device=device)
        
        mood_abs = min(abs(self.mood.mood_state.item()), 1.0)
        mood_factor = 1.0 + mood_abs * 0.4
        
        influence_scale = self.psyche_influence_dynamic.item()
        
        influence = psyche_embed.view(1, 1, -1) * decay.view(1, -1, 1)
        influence = influence * influence_scale * mood_factor
        influence = torch.tanh(influence * 0.6)
        influence = torch.clamp(influence, -1.5, 1.5)
        
        # 🔥 ИСПРАВЛЕНО: используем self._influence_layer_norm (уже в __init__)
        gate = torch.sigmoid(self._influence_gate_logit)
        influence_normed = self._influence_layer_norm(influence)
        
        result = hidden_states + gate * influence_normed * 0.25
        result = torch.clamp(result, -10.0, 10.0)
        
        if was_2d:
            return result.squeeze(1)
        return result
    
    def get_detailed_report(self) -> Dict:
        """🔥 ОПТИМИЗИРОВАННЫЙ подробный отчёт"""
        device = self.trait_raw.device
        
        with torch.no_grad():
            try:
                self.tick(training=False)
            except Exception as e:
                logger.warning(f"⚠ Ошибка в tick при генерации отчета: {e}")
                return self._get_fallback_report(device)
            
            trait_values = self.get_trait_values().detach().cpu()
            drive_values = self.get_drive_values().detach().cpu()
            
            mood_mods = self.mood.get_modifiers()
            
            plasticity = self.dynamics.compute_plasticity(
                trait_values.to(device), 
                self.experiences_count.item(),
                mood_mods["plasticity_mult"],
                drive_values.to(device)
            ).detach().cpu()
            
            resistance = self.dynamics.compute_resistance(
                trait_values.to(device),
                self._trait_target_min.to(device),
                self._trait_target_max.to(device),
                mood_mods["resistance_mult"],
                drive_values.to(device)
            ).detach().cpu()
            
            drive_dynamics = self.dynamics.compute_drive_dynamics(
                drive_values.to(device),
                trait_values.to(device)
            )
            drive_net_change = drive_dynamics["net_change"].detach().cpu()
            
            # 🔥 ОПТИМИЗИРОВАНО: собираем все скаляры в одном месте
            scalar_tensors = {
                "mood_state": self.mood.mood_state,
                "mood_velocity": self.mood.mood_velocity,
                "radicalization": self._radicalization,
                "system_entropy": self.dynamics._system_entropy,
                "system_energy": self.dynamics._system_energy,
                "system_stability": self.dynamics._system_stability,
                "temperature": self.temperature,
                "learning_rate_factor": self._learning_rate_factor,
            }
            
            scalars = {k: float(v.cpu().item()) for k, v in scalar_tensors.items()}
            
            # Векторизованные метрики
            trait_velocity = self.dynamics._trait_velocity.detach().cpu()
            drive_velocity = self.dynamics._drive_velocity.detach().cpu()
            
            mood_trend = self.mood.get_trend()
            
            reward_history = self.reward_history.cpu()
            recent_rewards = reward_history[reward_history != 0]
            avg_reward = float(recent_rewards.mean().item()) if len(recent_rewards) > 0 else 0.0
            
            goal_report = self.goal_system.get_goal_report()
            action_report = self.action_weights.get_efficacy_report()
            controller_report = self.get_controller_report()
            
            control_signals = self.get_memory_control_signals()
            
            # Векторизованные метрики симметрии
            symmetry_metrics = {
                "trait_symmetry": float(torch.mean((trait_values - 0.5).abs()).item()),
                "drive_symmetry": float(torch.mean((drive_values - 0.5).abs()).item()),
                "mood_symmetry": float(abs(scalars["mood_state"])),
                "overload_symmetry": float(abs(self.mood._overload_factor.item() - 1.0)),
                "reward_symmetry": float(abs(avg_reward)) if avg_reward != 0 else 0.0
            }
            
            # Динамика черт (векторизовано)
            trait_dynamics = {}
            for i, name in enumerate(self.trait_names):
                trait_dynamics[name] = {
                    "value": float(trait_values[i].item()),
                    "plasticity": float(plasticity[i].item()),
                    "resistance": float(resistance[i].item()),
                    "velocity": float(trait_velocity[i].item()) if i < len(trait_velocity) else 0.0
                }
            
            # Динамика драйвов (векторизовано)
            drive_dynamics_report = {}
            for i, name in enumerate(self.drive_names):
                drive_dynamics_report[name] = {
                    "value": float(drive_values[i].item()),
                    "net_change": float(drive_net_change[i].item()) if i < len(drive_net_change) else 0.0,
                    "velocity": float(drive_velocity[i].item()) if i < len(drive_velocity) else 0.0
                }
            
            return {
                "traits": {name: float(trait_values[i]) for i, name in enumerate(self.trait_names)},
                "drives": {name: float(drive_values[i]) for i, name in enumerate(self.drive_names)},
                "trait_dynamics": trait_dynamics,
                "drive_dynamics": drive_dynamics_report,
                "dynamic_coefficients": {
                    "plasticity": {name: float(plasticity[i]) for i, name in enumerate(self.trait_names)},
                    "resistance": {name: float(resistance[i]) for i, name in enumerate(self.trait_names)},
                    "drive_net_change": {name: float(drive_net_change[i]) for i, name in enumerate(self.drive_names)},
                },
                "mood": {
                    "state": scalars["mood_state"],
                    "velocity": scalars["mood_velocity"],
                    "trend": mood_trend,
                    "modifiers": mood_mods,
                },
                "controller": controller_report,
                "control_signals": control_signals,
                "symmetry_metrics": symmetry_metrics,
                "system_state": {
                    "radicalization": scalars["radicalization"],
                    "entropy": scalars["system_entropy"],
                    "energy": scalars["system_energy"],
                    "stability": scalars["system_stability"],
                    "trait_velocity_norm": float(torch.norm(trait_velocity).item()),
                    "drive_velocity_norm": float(torch.norm(drive_velocity).item()),
                    "learning_rate_factor": scalars["learning_rate_factor"],
                    "stability_counter": int(self._stability_counter.item()),
                    "extreme_corrections": int(self._extreme_correction_counter.item()),
                },
                "rl_state": {
                    "temperature": scalars["temperature"],
                    "last_reward": float(self.last_reward.item()),
                    "avg_reward": avg_reward,
                    "reward_history_size": int(torch.sum(reward_history != 0).item()),
                },
                "goals": goal_report,
                "actions": action_report,
                "experiences": int(self.experiences_count.item()),
                "ticks": int(self.total_ticks.item()),
            }
    
    def _get_fallback_report(self, device) -> Dict:
        return {
            "traits": {name: 0.5 for name in self.trait_names},
            "drives": {name: 0.5 for name in self.drive_names},
            "dynamic_coefficients": {
                "plasticity": {name: 0.5 for name in self.trait_names},
                "resistance": {name: 0.01 for name in self.trait_names},
                "drive_net_change": {name: 0.0 for name in self.drive_names},
            },
            "mood": {
                "state": 0.0,
                "velocity": 0.0,
                "trend": 0.0,
                "modifiers": {
                    "plasticity_mult": 1.0,
                    "resistance_mult": 1.0,
                    "exploration_bonus": 0.0,
                    "noise_mult": 1.0,
                    "learning_rate_mult": 1.0,
                    "overload_factor": 1.0,
                    "overload_recovery": 1.0,
                },
            },
            "controller": {
                "loss_history": {"mean": 0.0, "std": 0.0, "recent": []},
                "reward_history": {"mean": 0.0, "std": 0.0, "recent": []},
                "baseline": 0.0,
                "exploration_rate": 1.0,
                "training_steps": 0,
                "experience_buffer_size": 0
            },
            "control_signals": {
                "curiosity_threshold": 0.2,
                "recall_threshold_mod": 1.0,
                "write_intensity": 1.0,
                "noise_mod": 1.0,
                "negative_sensitivity": 1.0,
                "exploration_bonus": 0.0,
                "learning_rate_mult": 1.0,
                "fatigue_level": 0.5,
                "novelty_hunger": 0.5,
                "mood_influence": 0.0
            },
            "system_state": {
                "radicalization": 1.0,
                "entropy": 0.5,
                "energy": 1.0,
                "stability": 0.5,
                "trait_velocity_norm": 0.0,
                "drive_velocity_norm": 0.0,
                "learning_rate_factor": 1.0,
                "stability_counter": 0,
                "extreme_corrections": 0,
            },
            "rl_state": {
                "temperature": 1.0,
                "last_reward": 0.0,
                "avg_reward": 0.0,
                "reward_history_size": 0,
            },
            "goals": {
                "progress": {f"goal_{i}": 0.0 for i in range(5)},
                "priority": {f"goal_{i}": 1.0 for i in range(5)},
                "urgency": {f"goal_{i}": 0.0 for i in range(5)},
            },
            "actions": {
                "efficacy": {name: 1.0 for name in self.action_weights.action_names},
                "usage": {name: 0.0 for name in self.action_weights.action_names},
                "diversity": 1.0,
            },
            "experiences": int(self.experiences_count.item()),
            "ticks": int(self.total_ticks.item()),
        }
    
    def get_psyche_state(self) -> Dict:
        base_state = {
            "trait_raw": self.trait_raw.detach().cpu().clone(),
            "trait_momentum": self.trait_momentum.detach().cpu().clone(),
            "drive_values": self.drive_values.detach().cpu().clone(),
            "experiences_count": self.experiences_count.cpu().clone(),
            "total_ticks": self.total_ticks.cpu().clone(),
            "radicalization": self._radicalization.cpu().clone(),
            "learning_rate_factor": self._learning_rate_factor.cpu().clone(),
            "stability_counter": self._stability_counter.cpu().clone(),
            "extreme_correction_counter": self._extreme_correction_counter.cpu().clone(),
        }
        
        learnable_params = {
            "opposition_weights": self.opposition_weights.detach().cpu().clone(),
            "resonance_frequency": self.resonance_frequency.detach().cpu().clone(),
            "resonance_amplitude": self.resonance_amplitude.detach().cpu().clone(),
            # УБРАЛИ старый psyche_influence_weight
        }
        
        # 🔥 ДОБАВЛЯЕМ НОВЫЙ БЛОК СОСТОЯНИЙ ВЛИЯНИЯ (Кинематика)
        influence_state = {}
        if hasattr(self, 'psyche_influence_dynamic'):
            influence_state["psyche_influence_dynamic"] = self.psyche_influence_dynamic.cpu().clone()
        if hasattr(self, 'psyche_influence_target'):
            influence_state["psyche_influence_target"] = self.psyche_influence_target.cpu().clone()
        if hasattr(self, 'psyche_influence_velocity'):
            influence_state["psyche_influence_velocity"] = self.psyche_influence_velocity.cpu().clone()
        if hasattr(self, 'psyche_influence_min'):
            influence_state["psyche_influence_min"] = self.psyche_influence_min.cpu().clone()
        if hasattr(self, 'psyche_influence_max'):
            influence_state["psyche_influence_max"] = self.psyche_influence_max.cpu().clone()
        if hasattr(self, 'psyche_influence_inertia'):
            influence_state["psyche_influence_inertia"] = self.psyche_influence_inertia.cpu().clone()
        
        dynamics_state = {
            "base_plasticity": self.dynamics.base_plasticity.detach().cpu().clone(),
            "base_resistance": self.dynamics.base_resistance.detach().cpu().clone(),
            "base_drive_rate": self.dynamics.base_drive_rate.detach().cpu().clone(),
            "trait_influence": self.dynamics.trait_influence.detach().cpu().clone(),
            "trait_to_drive": self.dynamics.trait_to_drive.detach().cpu().clone(),
            "drive_to_trait": self.dynamics.drive_to_trait.detach().cpu().clone(),
            "trait_history": self.dynamics._trait_history.cpu().clone(),
            "drive_history": self.dynamics._drive_history.cpu().clone(),
            "history_idx": self.dynamics._history_idx.cpu().clone(),
            "trait_velocity": self.dynamics._trait_velocity.cpu().clone(),
            "trait_acceleration": self.dynamics._trait_acceleration.cpu().clone(),
            "drive_velocity": self.dynamics._drive_velocity.cpu().clone(),
            "system_entropy": self.dynamics._system_entropy.cpu().clone(),
            "system_energy": self.dynamics._system_energy.cpu().clone(),
            "system_stability": self.dynamics._system_stability.cpu().clone(),
        }
        
        action_weights_state = {
            "action_weights": self.action_weights.action_weights.detach().cpu().clone(),
            "action_efficacy": self.action_weights._action_efficacy.cpu().clone(),
            "action_usage": self.action_weights._action_usage.cpu().clone(),
            "last_trait_change": self.action_weights._last_trait_change.cpu().clone(),
            "action_names": self.action_weights.action_names,
        }
        
        mood_params = {}
        if hasattr(self, 'mood'):
            mood = self.mood
            mood_params = {
                "mood_state": mood.mood_state.detach().cpu().clone(),
                "mood_velocity": mood.mood_velocity.detach().cpu().clone(),
                "inertia": mood.inertia.detach().cpu().clone(),
                "recovery_rate": mood.recovery_rate.detach().cpu().clone(),
                "balance_sensitivity": mood.balance_sensitivity.detach().cpu().clone(),
                "tension_sensitivity": mood.tension_sensitivity.detach().cpu().clone(),
                "experience_sensitivity": mood.experience_sensitivity.detach().cpu().clone(),
                "mood_mass": mood.mood_mass.detach().cpu().clone(),
                "mood_damping": mood.mood_damping.detach().cpu().clone(),
                "mood_history": mood._mood_history.cpu().clone(),
                "mood_history_idx": mood._history_idx.cpu().clone(),
                "overload_factor": mood._overload_factor.cpu().clone(),
                "overload_recovery": mood._overload_recovery.cpu().clone(),
            }
        
        goal_state = {
            "goal_embeddings": self.goal_system.goal_embeddings.weight.detach().cpu().clone(),
            "goal_progress": self.goal_system.goal_progress.detach().cpu().clone(),
            "goal_priority": self.goal_system.goal_priority.detach().cpu().clone(),
            "goal_urgency": self.goal_system.goal_urgency.detach().cpu().clone(),
            "goal_history": self.goal_system._goal_history.cpu().clone(),
            "goal_history_idx": self.goal_system._history_idx.cpu().clone(),
        }
        
        rl_state = {
            "temperature": self.temperature.cpu().clone(),
            "last_reward": self.last_reward.cpu().clone(),
            "reward_history": self.reward_history.cpu().clone(),
            "reward_idx": self.reward_idx.cpu().clone(),
        }
        
        controller_state = {
            "controller_state_dict": self.controller.state_dict(),
            "controller_reward_baseline": self.controller_reward_baseline.cpu().clone(),
            "prev_controller_state": self._prev_controller_state.cpu().clone(),
            "prev_controller_action": self._prev_controller_action.cpu().clone(),
            "controller_loss_history": self.controller_loss_history.cpu().clone(),
            "controller_reward_history": self.controller_reward_history.cpu().clone(),
            "controller_history_idx": self.controller_history_idx.cpu().clone(),
            "controller_batch_buffer": list(self.controller.batch_buffer),
            "controller_training_steps": int(self.controller.training_steps.item()),
            "controller_version": "v9.2_gaussian_controller",
        }

        # 🔥 ДОБАВИТЬ ПРОВЕРКУ:
        if self.controller_optimizer is not None:
            controller_state["controller_optimizer_state_dict"] = self.controller_optimizer.state_dict()
        else:
            controller_state["controller_optimizer_state_dict"] = None
            logger.debug("⚠ controller_optimizer is None, сохраняю None")
        
        return {
            **base_state,
            **learnable_params,
            "influence_state": influence_state,  # 🔥 ВОТ ЭТО ДОБАВИТЬ
            "dynamics": dynamics_state,
            "action_weights": action_weights_state,
            "mood_params": mood_params,
            "goal_system": goal_state,
            "rl_state": rl_state,
            "controller_state": controller_state,
            "psyche_version": "6.0_neurocontroller",
        }
    
    def set_psyche_state(self, state: Dict):
        """Обновленный set_psyche_state с защитой от None и загрузкой оптимизатора"""
        if state is None:
            logger.warning("⚠ Попытка загрузить None state в психику")
            return
        
        with torch.no_grad():
            # ===========================================================
            # 🔥 БАЗОВЫЕ СОСТОЯНИЯ с проверкой на None
            # ===========================================================
            if "trait_raw" in state and state["trait_raw"] is not None:
                self.trait_raw.data.copy_(state["trait_raw"].to(self.trait_raw.device))
            
            if "trait_momentum" in state and state["trait_momentum"] is not None:
                self.trait_momentum.data.copy_(state["trait_momentum"].to(self.trait_momentum.device))
            
            if "drive_values" in state and state["drive_values"] is not None:
                self.drive_values.data.copy_(state["drive_values"].to(self.drive_values.device))
            
            if "experiences_count" in state and state["experiences_count"] is not None:
                self.experiences_count.copy_(state["experiences_count"].to(self.experiences_count.device))
            
            if "total_ticks" in state and state["total_ticks"] is not None:
                self.total_ticks.copy_(state["total_ticks"].to(self.total_ticks.device))
            
            # ===========================================================
            # 🔥 ПАРАМЕТРЫ ВЛИЯНИЯ (Кинематика)
            # ===========================================================
            if "influence_state" in state and state["influence_state"] is not None:
                inf = state["influence_state"]
                
                # Создаем буферы, если их еще нет
                if not hasattr(self, 'psyche_influence_dynamic'):
                    self.register_buffer('psyche_influence_dynamic', torch.tensor(0.25))
                if not hasattr(self, 'psyche_influence_target'):
                    self.register_buffer('psyche_influence_target', torch.tensor(0.25))
                if not hasattr(self, 'psyche_influence_velocity'):
                    self.register_buffer('psyche_influence_velocity', torch.tensor(0.0))
                if not hasattr(self, 'psyche_influence_min'):
                    self.register_buffer('psyche_influence_min', torch.tensor(0.10))
                if not hasattr(self, 'psyche_influence_max'):
                    self.register_buffer('psyche_influence_max', torch.tensor(0.45))
                if not hasattr(self, 'psyche_influence_inertia'):
                    self.register_buffer('psyche_influence_inertia', torch.tensor(0.85))
                
                if "psyche_influence_dynamic" in inf:
                    self.psyche_influence_dynamic.copy_(inf["psyche_influence_dynamic"].to(self.psyche_influence_dynamic.device))
                if "psyche_influence_target" in inf:
                    self.psyche_influence_target.copy_(inf["psyche_influence_target"].to(self.psyche_influence_target.device))
                if "psyche_influence_velocity" in inf:
                    self.psyche_influence_velocity.copy_(inf["psyche_influence_velocity"].to(self.psyche_influence_velocity.device))
                if "psyche_influence_min" in inf:
                    self.psyche_influence_min.copy_(inf["psyche_influence_min"].to(self.psyche_influence_min.device))
                if "psyche_influence_max" in inf:
                    self.psyche_influence_max.copy_(inf["psyche_influence_max"].to(self.psyche_influence_max.device))
                if "psyche_influence_inertia" in inf:
                    self.psyche_influence_inertia.copy_(inf["psyche_influence_inertia"].to(self.psyche_influence_inertia.device))
            
            # ===========================================================
            # 🔥 Восстанавливаем старые обучаемые параметры (без influence_weight)
            # ===========================================================
            if "opposition_weights" in state and state["opposition_weights"] is not None:
                self.opposition_weights.data.copy_(state["opposition_weights"].to(self.opposition_weights.device))
            if "resonance_frequency" in state and state["resonance_frequency"] is not None:
                self.resonance_frequency.data.copy_(state["resonance_frequency"].to(self.resonance_frequency.device))
            if "resonance_amplitude" in state and state["resonance_amplitude"] is not None:
                self.resonance_amplitude.data.copy_(state["resonance_amplitude"].to(self.resonance_amplitude.device))
            
            # ===========================================================
            # 🔥 MOOD STATE
            # ===========================================================
            if "mood_state" in state and state["mood_state"] is not None:
                self.mood.mood_state.data.copy_(state["mood_state"].to(self.mood.mood_state.device))
            
            if "mood_velocity" in state and state["mood_velocity"] is not None:
                self.mood.mood_velocity.data.copy_(state["mood_velocity"].to(self.mood.mood_velocity.device))
            
            # ===========================================================
            # 🔥 ДИНАМИЧЕСКИЕ ПАРАМЕТРЫ
            # ===========================================================
            if "radicalization" in state and state["radicalization"] is not None:
                self._radicalization.copy_(state["radicalization"].to(self._radicalization.device))
            
            if "learning_rate_factor" in state and state["learning_rate_factor"] is not None:
                self._learning_rate_factor.copy_(state["learning_rate_factor"].to(self._learning_rate_factor.device))
            
            if "stability_counter" in state and state["stability_counter"] is not None:
                self._stability_counter.copy_(state["stability_counter"].to(self._stability_counter.device))
            
            if "extreme_correction_counter" in state and state["extreme_correction_counter"] is not None:
                self._extreme_correction_counter.copy_(state.get("extreme_correction_counter", torch.tensor(0)).to(self._extreme_correction_counter.device))
            
            # ===========================================================
            # 🔥 RL STATE
            # ===========================================================
            if "rl_state" in state and state["rl_state"] is not None:
                rl = state["rl_state"]
                if "temperature" in rl and rl["temperature"] is not None:
                    self.temperature.copy_(rl["temperature"].to(self.temperature.device))
                if "last_reward" in rl and rl["last_reward"] is not None:
                    self.last_reward.copy_(rl["last_reward"].to(self.last_reward.device))
                if "reward_history" in rl and rl["reward_history"] is not None:
                    self.reward_history.copy_(rl["reward_history"].to(self.reward_history.device))
                if "reward_idx" in rl and rl["reward_idx"] is not None:
                    self.reward_idx.copy_(rl["reward_idx"].to(self.reward_idx.device))
            
            # ===========================================================
            # 🔥 ACTION WEIGHTS
            # ===========================================================
            if "action_weights" in state and state["action_weights"] is not None:
                aw = state["action_weights"]
                device = self.action_weights.action_weights.device
                if "action_weights" in aw and aw["action_weights"] is not None:
                    self.action_weights.action_weights.data.copy_(aw["action_weights"].to(device))
                if "action_efficacy" in aw and aw["action_efficacy"] is not None:
                    self.action_weights._action_efficacy.copy_(aw["action_efficacy"].to(device))
                if "action_usage" in aw and aw["action_usage"] is not None:
                    self.action_weights._action_usage.copy_(aw["action_usage"].to(device))
                if "last_trait_change" in aw and aw["last_trait_change"] is not None:
                    self.action_weights._last_trait_change.copy_(aw["last_trait_change"].to(device))
                if "action_names" in aw and aw["action_names"] is not None:
                    self.action_weights.action_names = aw["action_names"]
                    self.action_weights.num_actions = len(aw["action_names"])
                    self.action_weights.action_to_idx = {name: i for i, name in enumerate(aw["action_names"])}
            
            # ===========================================================
            # 🔥 MOOD PARAMS
            # ===========================================================
            if "mood_params" in state and state["mood_params"] is not None:
                mood = state["mood_params"]
                device = self.mood.inertia.device
                if "inertia" in mood and mood["inertia"] is not None:
                    self.mood.inertia.data.copy_(mood["inertia"].to(device))
                if "recovery_rate" in mood and mood["recovery_rate"] is not None:
                    self.mood.recovery_rate.data.copy_(mood["recovery_rate"].to(device))
                if "balance_sensitivity" in mood and mood["balance_sensitivity"] is not None:
                    self.mood.balance_sensitivity.data.copy_(mood["balance_sensitivity"].to(device))
                if "tension_sensitivity" in mood and mood["tension_sensitivity"] is not None:
                    self.mood.tension_sensitivity.data.copy_(mood["tension_sensitivity"].to(device))
                if "experience_sensitivity" in mood and mood["experience_sensitivity"] is not None:
                    self.mood.experience_sensitivity.data.copy_(mood["experience_sensitivity"].to(device))
                if "mood_mass" in mood and mood["mood_mass"] is not None:
                    self.mood.mood_mass.data.copy_(mood["mood_mass"].to(device))
                if "mood_damping" in mood and mood["mood_damping"] is not None:
                    self.mood.mood_damping.data.copy_(mood["mood_damping"].to(device))
                if "mood_history" in mood and mood["mood_history"] is not None:
                    self.mood._mood_history.copy_(mood["mood_history"].to(device))
                if "mood_history_idx" in mood and mood["mood_history_idx"] is not None:
                    self.mood._history_idx.copy_(mood["mood_history_idx"].to(device))
                if "overload_factor" in mood and mood["overload_factor"] is not None:
                    self.mood._overload_factor.copy_(mood["overload_factor"].to(device))
                if "overload_recovery" in mood and mood["overload_recovery"] is not None:
                    self.mood._overload_recovery.copy_(mood["overload_recovery"].to(device))
            
            # ===========================================================
            # 🔥 GOAL SYSTEM
            # ===========================================================
            if "goal_system" in state and state["goal_system"] is not None:
                goal = state["goal_system"]
                device = self.goal_system.goal_embeddings.weight.device
                if "goal_embeddings" in goal and goal["goal_embeddings"] is not None:
                    self.goal_system.goal_embeddings.weight.data.copy_(goal["goal_embeddings"].to(device))
                if "goal_progress" in goal and goal["goal_progress"] is not None:
                    self.goal_system.goal_progress.data.copy_(goal["goal_progress"].to(device))
                if "goal_priority" in goal and goal["goal_priority"] is not None:
                    self.goal_system.goal_priority.data.copy_(goal["goal_priority"].to(device))
                if "goal_urgency" in goal and goal["goal_urgency"] is not None:
                    self.goal_system.goal_urgency.data.copy_(goal["goal_urgency"].to(device))
                if "goal_history" in goal and goal["goal_history"] is not None:
                    self.goal_system._goal_history.copy_(goal["goal_history"].to(device))
                if "goal_history_idx" in goal and goal["goal_history_idx"] is not None:
                    self.goal_system._history_idx.copy_(goal["goal_history_idx"].to(device))
            
            # ===========================================================
            # 🔥 DYNAMICS
            # ===========================================================
            if "dynamics" in state and state["dynamics"] is not None:
                dyn = state["dynamics"]
                device = self.dynamics.base_plasticity.device
                if "base_plasticity" in dyn and dyn["base_plasticity"] is not None:
                    self.dynamics.base_plasticity.data.copy_(dyn["base_plasticity"].to(device))
                if "base_resistance" in dyn and dyn["base_resistance"] is not None:
                    self.dynamics.base_resistance.data.copy_(dyn["base_resistance"].to(device))
                if "base_drive_rate" in dyn and dyn["base_drive_rate"] is not None:
                    self.dynamics.base_drive_rate.data.copy_(dyn["base_drive_rate"].to(device))
                if "trait_influence" in dyn and dyn["trait_influence"] is not None:
                    self.dynamics.trait_influence.data.copy_(dyn["trait_influence"].to(device))
                if "trait_to_drive" in dyn and dyn["trait_to_drive"] is not None:
                    self.dynamics.trait_to_drive.data.copy_(dyn["trait_to_drive"].to(device))
                if "drive_to_trait" in dyn and dyn["drive_to_trait"] is not None:
                    self.dynamics.drive_to_trait.data.copy_(dyn["drive_to_trait"].to(device))
                if "trait_history" in dyn and dyn["trait_history"] is not None:
                    self.dynamics._trait_history.copy_(dyn["trait_history"].to(device))
                if "drive_history" in dyn and dyn["drive_history"] is not None:
                    self.dynamics._drive_history.copy_(dyn["drive_history"].to(device))
                if "history_idx" in dyn and dyn["history_idx"] is not None:
                    self.dynamics._history_idx.copy_(dyn["history_idx"].to(device))
                if "trait_velocity" in dyn and dyn["trait_velocity"] is not None:
                    self.dynamics._trait_velocity.copy_(dyn["trait_velocity"].to(device))
                if "trait_acceleration" in dyn and dyn["trait_acceleration"] is not None:
                    self.dynamics._trait_acceleration.copy_(dyn["trait_acceleration"].to(device))
                if "drive_velocity" in dyn and dyn["drive_velocity"] is not None:
                    self.dynamics._drive_velocity.copy_(dyn["drive_velocity"].to(device))
                if "system_entropy" in dyn and dyn["system_entropy"] is not None:
                    self.dynamics._system_entropy.copy_(dyn["system_entropy"].to(device))
                if "system_energy" in dyn and dyn["system_energy"] is not None:
                    self.dynamics._system_energy.copy_(dyn["system_energy"].to(device))
                if "system_stability" in dyn and dyn["system_stability"] is not None:
                    self.dynamics._system_stability.copy_(dyn["system_stability"].to(device))
            
            # ===========================================================
            # 🔥 CONTROLLER STATE (С ЗАГРУЗКОЙ ОПТИМИЗАТОРА)
            # ===========================================================
            if "controller_state" in state and state["controller_state"] is not None:
                ctrl = state["controller_state"]
                
                # Загружаем веса контроллера
                if "controller_state_dict" in ctrl and ctrl["controller_state_dict"] is not None:
                    controller_sd = ctrl["controller_state_dict"]
                    
                    if 'last_log_prob' not in controller_sd:
                        controller_sd['last_log_prob'] = self.controller.last_log_prob.cpu().clone()
                    
                    self.controller.load_state_dict(controller_sd, strict=False)
                
                # ===========================================================
                # 🔥 ИСПРАВЛЕНИЕ БАГА 3: Загружаем оптимизатор с созданием
                # ===========================================================
                if "controller_optimizer_state_dict" in ctrl and ctrl["controller_optimizer_state_dict"] is not None:
                    # 🔥 Создаем оптимизатор, если его еще нет
                    device = next(self.controller.parameters()).device
                    self._ensure_optimizers(device)
                    
                    try:
                        self.controller_optimizer.load_state_dict(ctrl["controller_optimizer_state_dict"])
                        # Синхронизация устройств внутри стейта
                        for param_group in self.controller_optimizer.param_groups:
                            for param in param_group['params']:
                                if param in self.controller_optimizer.state:
                                    state_dict = self.controller_optimizer.state[param]
                                    for key, value in state_dict.items():
                                        if isinstance(value, torch.Tensor) and value.device != device:
                                            state_dict[key] = value.to(device)
                        logger.info("✅ Состояние оптимизатора контроллера загружено")
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка загрузки оптимизатора: {e}")
                else:
                    logger.debug("⚠ controller_optimizer_state_dict is None, будет создан заново")
                
                # Загружаем остальные состояния контроллера
                if "controller_reward_baseline" in ctrl and ctrl["controller_reward_baseline"] is not None:
                    self.controller_reward_baseline.copy_(ctrl["controller_reward_baseline"].to(self.controller_reward_baseline.device))
                
                if "prev_controller_state" in ctrl and ctrl["prev_controller_state"] is not None:
                    self._prev_controller_state.copy_(ctrl["prev_controller_state"].to(self._prev_controller_state.device))
                
                if "prev_controller_action" in ctrl and ctrl["prev_controller_action"] is not None:
                    self._prev_controller_action.copy_(ctrl["prev_controller_action"].to(self._prev_controller_action.device))
                
                if "controller_loss_history" in ctrl and ctrl["controller_loss_history"] is not None:
                    self.controller_loss_history.copy_(ctrl["controller_loss_history"].to(self.controller_loss_history.device))
                
                if "controller_reward_history" in ctrl and ctrl["controller_reward_history"] is not None:
                    self.controller_reward_history.copy_(ctrl["controller_reward_history"].to(self.controller_reward_history.device))
                
                if "controller_history_idx" in ctrl and ctrl["controller_history_idx"] is not None:
                    self.controller_history_idx.copy_(ctrl["controller_history_idx"].to(self.controller_history_idx.device))
                
                # Загружаем буферы опыта
                if "controller_batch_buffer" in ctrl and ctrl["controller_batch_buffer"] is not None:
                    buffer_list = ctrl["controller_batch_buffer"]
                    self.controller.batch_buffer.clear()
                    for item in buffer_list:
                        self.controller.batch_buffer.append(item)
                
                if "controller_training_steps" in ctrl and ctrl["controller_training_steps"] is not None:
                    self.controller.training_steps.fill_(ctrl["controller_training_steps"])
            
            # Сбрасываем флаг инициализации оптимизаторов
            self._optimizers_initialized = False
            
            logger.info(f"✅ Динамичная психика v9.0 загружена успешно")
        
# ===================================================================
# ГЛАВНЫЙ КЛАСС THALIA v9.0 с НЕЙРОКОНТРОЛЛЕРОМ
# ===================================================================
class Thalia(GPT2LMHeadModel):
    config_class = ThaliaConfig
   
    def __init__(self, config):
        config._attn_implementation = "sdpa"  
        super().__init__(config)

        for module in self.modules():
            if isinstance(module, nn.GELU):
                module.approximate = 'tanh'
       
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            min_length=20,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.15,
            no_repeat_ngram_size=0,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )
       
        self.tokenizer = None
        
        # ===================================================================
        # 🔥 SEMANTIC ANCHORS — стабилизация семантического пространства
        # ===================================================================
        self.num_anchors = getattr(config, 'num_anchors', 24)           # 16–32 оптимально
        self.anchor_weight = getattr(config, 'anchor_weight', 0.0012)   # небольшой вес
        
        # Якоря — фиксированные опорные точки (не обучаются)
        self.register_buffer('semantic_anchors', torch.randn(self.num_anchors, config.n_embd))
        nn.init.orthogonal_(self.semantic_anchors)                     # ортогональные = максимальная независимость
        
        # Запоминаем исходную семантическую сигнатуру
        self.register_buffer('initial_anchor_sims', None)
        self._anchors_initialized = False
        
        logger.info(f"⚓ Semantic Anchors активированы: {self.num_anchors} ортогональных якорей")
        
        # Hebb-память
        self.use_hebb_layers = getattr(config, 'use_hebb_layers', True)
        if self.use_hebb_layers:
            hebb_slots = getattr(config, 'hebb_num_slots', 64)  
            self.hebb_layers = nn.ModuleList([
                TemporalHebbLayer(config, num_slots=hebb_slots, layer_idx=i, total_layers=config.n_layer) 
                for i in range(config.n_layer)
            ])
            if len(self.hebb_layers) > 0:
                self.hebb_layers[0].log_init(len(self.hebb_layers))
            logger.info(f"🧠 Hebb-память (TEMPORAL v2.0): {config.n_layer} слоев")
            
            # 🔥 ВНЕДРЯЕМ HEBB ПРЯМО В БЛОКИ GPT-2
            self._register_hebb_hooks()
        else:
            self.hebb_layers = None   
            
        # ===========================================================
        # 🔥 Создаем hebb_to_centroid 
        # ===========================================================
        slot_size = getattr(config, 'slot_size', config.n_embd // 2)
        self.hebb_to_centroid = nn.Linear(config.n_embd, slot_size)
        nn.init.zeros_(self.hebb_to_centroid.weight)
        if self.hebb_to_centroid.bias is not None:
            nn.init.zeros_(self.hebb_to_centroid.bias)
        logger.info(f"🔄 hebb_to_centroid создан: {config.n_embd} → {slot_size}")
       
        # Динамичная психика
        if getattr(config, 'use_psyche_core', True):
            self.personality_core = DynamicPsycheCoreV6(config)
        else:
            self.personality_core = None
       
        if getattr(config, 'use_living_layer', True):
            self.living_layer = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.n_embd * 2, config.n_embd),
                nn.Tanh(),
            )
            logger.info("🧠 Живой слой активирован")
        else:
            self.living_layer = None
       
        try:
            from memory_heads import AdaptiveMemoryHeads
            self.adaptive_memory = AdaptiveMemoryHeads(config)
            logger.info("🧠 Адаптивная память загружена")
        except ImportError as e:
            logger.warning(f"⚠ Не удалось импортировать AdaptiveMemoryHeads: {e}")
            self.adaptive_memory = None
       
        self.step_count = 0
       
        if getattr(config, 'use_psyche_core', True):
            self.register_buffer("_last_generation_embed", torch.zeros(config.n_embd), persistent=False)
            
        self.experience_exchange = BidirectionalExperienceExchange(config)
        logger.info("🔄 Двусторонний обмен опытом активирован")
        
        if self.adaptive_memory is not None:
            self.adaptive_memory._exchange_ref = [self.experience_exchange]
        
        self._event_callbacks = {
            'curiosity_state_change': deque(maxlen=10),
            'sleep_started': deque(maxlen=10),
            'mood_extreme': deque(maxlen=10),
            'experience_recorded': deque(maxlen=10),
            'generation_completed': deque(maxlen=10),
            'goal_updated': deque(maxlen=10),
            'system_stabilized': deque(maxlen=10),
            'rl_reward': deque(maxlen=10),
            'burnout_protection': deque(maxlen=10),
            'controller_updated': deque(maxlen=10),
        }
        self.register_buffer("last_surprise", torch.tensor(0.0))
                
        logger.info(f"🧠 Thalia v9.1 с Hebb-памятью и темпоральным циклом инициализирована")
    
    def compute_alignment_losses(self):
        """Стабилизация всех пространств представлений"""
        losses = {}
        device = next(self.parameters()).device
        
        # Hebb ↔ Centroid alignment
        if hasattr(self, 'hebb_layers') and hasattr(self, 'adaptive_memory'):
            if self.hebb_layers and self.adaptive_memory is not None:
                hebb_pattern = self.hebb_layers[-1].patterns.mean(dim=0)
                hebb_proj = self.hebb_to_centroid(hebb_pattern)
                centroid_context, _ = self.adaptive_memory.centroid_memory.query(hebb_pattern)
                if centroid_context is not None and centroid_context.numel() > 0:
                    centroid_target = centroid_context.detach()
                    losses['hebb_centroid'] = F.mse_loss(hebb_proj, centroid_target) * 0.1
        
        if hasattr(self, 'experience_exchange'):
            align_loss = self.experience_exchange.compute_direct_alignment_loss()
            if isinstance(align_loss, torch.Tensor) and align_loss.requires_grad:
                losses['alignment'] = align_loss * 0.1
                self.last_align_loss_value = align_loss.item()
                logger.debug(f"✅ Alignment loss добавлен: {align_loss.item():.6f}")
        
        return losses

    def get_semantic_signature(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Возвращает семантическую подпись эмбеддингов относительно якорей"""
        # embeddings: [batch, seq_len, dim] или [batch, dim]
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)          # усредняем по последовательности
        
        # Softmax по косинусной близости (температура 0.5 даёт чёткое распределение)
        sim = F.softmax(embeddings @ self.semantic_anchors.T / 0.5, dim=-1)
        return sim  # [batch, num_anchors]
    
    def compute_anchor_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Anchor Loss — стабилизирует семантическое пространство"""
        if input_ids is None:
            return torch.tensor(0.0, device=self.device)
        
        # Получаем эмбеддинги токенов
        embeddings = self.transformer.wte(input_ids)                    # [batch, seq, dim]
        
        # Текущая семантическая сигнатура
        current_sims = self.get_semantic_signature(embeddings)
        
        # Инициализация при первом вызове
        if not self._anchors_initialized:
            self.initial_anchor_sims = current_sims.detach().clone()
            self._anchors_initialized = True
            return torch.tensor(0.0, device=embeddings.device)
        
        # MSE между текущей и исходной сигнатурой
        anchor_loss = F.mse_loss(current_sims, self.initial_anchor_sims)
        
        return anchor_loss * self.anchor_weight

    @torch.no_grad()
    def sync_anchors_with_centroids(self):
        """Синхронизирует якоря с самыми важными слотами центроидной памяти"""
        if not hasattr(self, 'adaptive_memory') or self.adaptive_memory is None:
            return
        
        centroid = self.adaptive_memory.centroid_memory
        if not hasattr(centroid, 'centroids'):
            return
        
        # Берём топ-N слотов по utility
        if hasattr(centroid, 'slot_utility'):
            top_k = min(self.num_anchors, centroid.num_slots)
            _, top_indices = torch.topk(centroid.slot_utility, top_k)
            
            self.semantic_anchors.data = centroid.centroids[top_indices].clone()
            # 🔥 СБРАСЫВАЕМ ЭТАЛОН — критически важно!
            self.initial_anchor_sims = None
            self._anchors_initialized = False
            logger.info(f"⚓ Anchors синхронизированы с {top_k} топ-слотами центроидной памяти")
        else:
            logger.warning("⚠ slot_utility не найден в центроидной памяти")

    def _register_hebb_hooks(self):
        """🔥 Внедряет Hebb-слои прямо в блоки GPT-2 через forward hooks"""
        if not self.use_hebb_layers or self.hebb_layers is None:
            return
            
        total_layers = len(self.transformer.h)
        # Инициализируем хранилище для лоссов
        self._hebb_aux_losses = [torch.tensor(0.0, device=self.config.device) for _ in range(total_layers)]
            
        for layer_idx, hebb_layer in enumerate(self.hebb_layers):
            if layer_idx >= total_layers:
                break
                    
            target_block = self.transformer.h[layer_idx]
                    
            def make_hook(hebb, idx):
                def hook(module, args, output):
                    # output в GPT-2 — это кортеж: (hidden_states, presents, attentions)
                    hidden_states = output[0]
                    
                    # 🔥 Отсекаем градиенты для трансформера, 
                    # чтобы память училась автономно, не ломая wte/wpe
                    hebb_input = hidden_states.detach() 
                    
                    # Вызываем Hebb БЕЗ no_grad, чтобы его внутренний контроллер мог учиться!
                    modified_hidden, aux_loss = hebb(
                        hidden_states=hebb_input,
                        attention_mask=None,
                        # Динамические сигналы читаем из атрибутов модели
                        surprise=getattr(self, 'current_surprise', 0.0), 
                        signals=getattr(self, 'current_memory_controls', None)
                    )
                    
                    # Сохраняем aux_loss (detach для хранения, но requires_grad остаётся)
                    if aux_loss is not None and aux_loss.requires_grad:
                        self._hebb_aux_losses[idx] = aux_loss
                    else:
                        self._hebb_aux_losses[idx] = torch.tensor(0.0, device=hidden_states.device)
                    
                    # 🔥 Возвращаем кортеж с модифицированным hidden_states
                    return (modified_hidden,) + output[1:]
                    
                return hook
                    
            target_block.register_forward_hook(make_hook(hebb_layer, layer_idx))
            
        logger.info(f"🔗 Hebb-слои бесшовно интегрированы в {len(self.hebb_layers)} блоков GPT-2")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.step_count += 1
        
        if self.personality_core is not None:
            if self.training:
                self.personality_core.tick(training=True)
            else:
                self.personality_core.tick(training=False)
        
        # ===================================================================
        # 🔥🔥🔥 УРОВЕНЬ 1: Родной трансформер (с Hebb внутри через хуки)
        # ===================================================================
        
        # Подготавливаем динамические сигналы для хуков
        if self.personality_core is not None:
            self.current_memory_controls = self.personality_core.get_memory_control_signals()
        else:
            self.current_memory_controls = None
        
        self.current_surprise = getattr(self, '_last_surprise', 0.0)
        
        # Очищаем aux_losses перед forward
        if hasattr(self, '_hebb_aux_losses'):
            device = input_ids.device if input_ids is not None else self.config.device
            for i in range(len(self._hebb_aux_losses)):
                self._hebb_aux_losses[i] = torch.tensor(0.0, device=device)
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = transformer_outputs[0]
        attention_patterns = transformer_outputs.attentions
        all_hidden_states = list(transformer_outputs.hidden_states)
        
        # ===================================================================
        # 🔥 Собираем aux_loss из хуков Hebb
        # ===================================================================
        total_hebb_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if hasattr(self, '_hebb_aux_losses'):
            for aux_loss in self._hebb_aux_losses:
                if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                    total_hebb_aux_loss = total_hebb_aux_loss + torch.clamp(aux_loss, 0.0, 5.0)
        
        memory_controls = {}
        if self.personality_core is not None:
            memory_controls = self.personality_core.get_memory_control_signals()
        
        mamba_hidden_for_exchange = None
        surprise = 0.0
        base_surprise = 0.0
        memory_result = {}
        
        # ===================================================================
        # УРОВЕНЬ 2: ЭМОЦИОНАЛЬНАЯ МОДУЛЯЦИЯ
        # ===================================================================
        if self.personality_core is not None:
            hidden_states = self.personality_core.influence_hidden_states(hidden_states)
        
        if self.living_layer is not None:
            living_output = self.living_layer(hidden_states)
            hidden_states = hidden_states + 0.15 * living_output
        
        # ===================================================================
        # УРОВЕНЬ 3: ДОЛГОСРОЧНАЯ ПАМЯТЬ
        # ===================================================================
        transformer_experience = None
        
        if self.experience_exchange.bidirectional_enabled:
            generation_metadata = {
                'step': self.step_count,
                'input_length': input_ids.shape[1] if input_ids is not None else 0,
                'training': self.training,
                'labels_provided': labels is not None,
                'batch_size': hidden_states.shape[0]
            }
            
            last_layer_attention = attention_patterns[-1] if attention_patterns else None
            transformer_experience = None
        
        final_control_signals = memory_controls.copy() if memory_controls else {}
        
        if self.use_hebb_layers and self.hebb_layers is not None:
            for layer in reversed(self.hebb_layers):
                if hasattr(layer, '_last_hebb_query') and layer._last_hebb_query is not None:
                    final_control_signals['hebb_query_vector'] = layer._last_hebb_query
                    break
        
        if self.adaptive_memory is not None:
            # 🔥 Убеждаемся, что hidden_states имеет правильную размерность для Mamba
            hebb_hidden = hidden_states
            if hebb_hidden.dim() == 2:
                hebb_hidden = hebb_hidden.unsqueeze(1)
            elif hebb_hidden.dim() == 1:
                hebb_hidden = hebb_hidden.unsqueeze(0).unsqueeze(1)
            
            memory_result = self.adaptive_memory(
                hidden_states=hebb_hidden,
                slots=None,
                transformer_experience=transformer_experience,
                control_signals=final_control_signals
            )
            
            # 🔥 Берём мета-лосс напрямую
            if 'meta_loss' in memory_result and memory_result['meta_loss'] is not None:
                meta_loss_tensor = memory_result['meta_loss']
                if isinstance(meta_loss_tensor, torch.Tensor) and meta_loss_tensor.requires_grad:
                    self._current_meta_loss = meta_loss_tensor
                    if self.step_count % 50 == 0:
                        logger.info(f"🧠 Meta loss получен: {meta_loss_tensor.item():.6f} (requires_grad=True)")
                else:
                    self._current_meta_loss = None
            else:
                self._current_meta_loss = None
            
            # Извлекаем мета-когнитивную информацию
            if 'meta_cognitive' in memory_result:
                self._last_meta = memory_result['meta_cognitive']
            
            base_surprise = memory_result.get('surprise', 0.0) if memory_result else 0.0
            surprise = base_surprise
            surprise = min(1.0, surprise)
            
            self.last_surprise.fill_(surprise)
            
            if 'sedimentary_context' in memory_result:
                sedimentary_context = memory_result['sedimentary_context']
                if sedimentary_context is not None and sedimentary_context.numel() > 0:
                    context_expanded = sedimentary_context.unsqueeze(0).unsqueeze(0)
                    if context_expanded.shape[-1] == hidden_states.shape[-1]:
                        stability = memory_controls.get('stability_factor', 0.2)
                        hidden_states = hidden_states + context_expanded * stability
            
            if memory_result and 'insights' in memory_result and memory_result['insights']:
                mamba_insight = memory_result['insights'][0]
                if mamba_insight.dim() == 1:
                    mamba_insight = mamba_insight.unsqueeze(0)
                if mamba_insight.dim() == 2:
                    mamba_hidden_for_exchange = mamba_insight.unsqueeze(1)
                else:
                    mamba_hidden_for_exchange = mamba_insight
            
            if self.personality_core is not None:
                try:
                    effort = memory_result.get('writer_delta', 0.0)
                    recall = memory_result.get('reader_improvement', 0.0)
                    if hasattr(self.personality_core, 'ingest_memory_experience'):
                        self.personality_core.ingest_memory_experience(
                            surprise_value=min(1.0, max(0.0, surprise)),
                            effort_value=min(1.0, max(0.0, effort)),
                            recall_success=min(1.0, max(0.0, recall))
                        )
                except Exception as e:
                    logger.debug(f"⚠ ingest_memory_experience: {e}")
        
        # ===================================================================
        # Синхронизированная запись пар для alignment
        # ===================================================================
        if self.experience_exchange.bidirectional_enabled and mamba_hidden_for_exchange is not None:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                t_raw = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                t_raw = hidden_states.mean(dim=1)
            
            if mamba_hidden_for_exchange.dim() == 3:
                m_raw = mamba_hidden_for_exchange[:, -1, :]
            elif mamba_hidden_for_exchange.dim() == 2:
                m_raw = mamba_hidden_for_exchange
            else:
                m_raw = mamba_hidden_for_exchange.view(-1, self.config.slot_size)
            
            if t_raw.shape[0] != m_raw.shape[0]:
                min_batch = min(t_raw.shape[0], m_raw.shape[0])
                if min_batch > 0:
                    t_raw_aligned = t_raw[:min_batch]
                    m_raw_aligned = m_raw[:min_batch]
                    
                    if not (torch.isnan(t_raw_aligned).any() or torch.isinf(t_raw_aligned).any() or 
                            torch.isnan(m_raw_aligned).any() or torch.isinf(m_raw_aligned).any()):
                        t_raw_norm = F.normalize(t_raw_aligned, dim=-1)
                        m_raw_norm = F.normalize(m_raw_aligned, dim=-1)
                        self.experience_exchange.capture_pair(t_raw_norm, m_raw_norm)
            else:
                if not (torch.isnan(t_raw).any() or torch.isinf(t_raw).any() or 
                        torch.isnan(m_raw).any() or torch.isinf(m_raw).any()):
                    t_raw_norm = F.normalize(t_raw, dim=-1)
                    m_raw_norm = F.normalize(m_raw, dim=-1)
                    self.experience_exchange.capture_pair(t_raw_norm, m_raw_norm)
        
        # ===================================================================
        # УРОВЕНЬ 4: ИНТЕГРАЦИЯ И СИМБИОЗ
        # ===================================================================
        if self.experience_exchange.bidirectional_enabled:
            exchange_mode = 'mutual_enhancement'
            
            if not self.training:
                if memory_result and memory_result.get('surprise', 0) > 0.3:
                    exchange_mode = 'mamba_filtering'
                elif memory_result and memory_result.get('writer_delta', 0) < 0.05:
                    exchange_mode = 'transformer_guidance'
            
            hidden_states, mamba_hidden_for_exchange = self.experience_exchange.exchange_experiences(
                transformer_hidden=hidden_states,
                mamba_hidden=mamba_hidden_for_exchange,
                mode=exchange_mode
            )
        
        # ===================================================================
        # УРОВЕНЬ 5: ВЫХОД
        # ===================================================================
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if shift_logits.numel() > 0 and shift_labels.numel() > 0:
                main_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = main_loss
        
        # ===================================================================
        # 🔥 ДОПОЛНИТЕЛЬНЫЕ LOSS КОМПОНЕНТЫ
        # ===================================================================
        
        if self.training and loss is not None:
            # 1. Hebb aux loss
            if total_hebb_aux_loss.numel() > 0 and total_hebb_aux_loss.requires_grad:
                hebb_loss = total_hebb_aux_loss.sum() if total_hebb_aux_loss.dim() > 0 else total_hebb_aux_loss
                loss = loss + 0.01 * torch.clamp(hebb_loss, 0.0, 50.0)
            
            # 2. Alignment losses
            try:
                align_losses = self.compute_alignment_losses()
                for loss_name, loss_value in align_losses.items():
                    if isinstance(loss_value, torch.Tensor) and loss_value.requires_grad:
                        loss = loss + loss_value * 0.02
            except Exception as e:
                logger.debug(f"⚠ Alignment losses error: {e}")
            
            # 3. Controller losses
            controller_loss_total = torch.tensor(0.0, device=loss.device)
            controller_count = 0
            for layer in self.hebb_layers:
                if hasattr(layer, '_controller_loss_for_backward') and layer._controller_loss_for_backward is not None:
                    if isinstance(layer._controller_loss_for_backward, torch.Tensor) and layer._controller_loss_for_backward.requires_grad:
                        controller_loss_total = controller_loss_total + layer._controller_loss_for_backward
                        controller_count += 1
                    layer._controller_loss_for_backward = None
            
            if controller_count > 0:
                controller_loss_total = torch.clamp(controller_loss_total, min=0.0)
                loss = loss + 0.01 * controller_loss_total
            
            # 4. МЕТА-LOSS (от предиктора)
            if hasattr(self, '_current_meta_loss') and self._current_meta_loss is not None:
                meta_loss_tensor = self._current_meta_loss
                if isinstance(meta_loss_tensor, torch.Tensor) and meta_loss_tensor.requires_grad:
                    meta_coef = getattr(self.config, 'meta_loss_coef', 0.05)
                    loss = loss + meta_coef * meta_loss_tensor
                self._current_meta_loss = None
            
            # 5. Repulsion loss из психики
            if self.personality_core is not None and hasattr(self.personality_core, 'get_repulsion_loss'):
                rep_loss = self.personality_core.get_repulsion_loss()
                if isinstance(rep_loss, torch.Tensor) and rep_loss.requires_grad:
                    rep_coef = getattr(self.config, 'repulsion_coef', 0.001)
                    loss = loss + rep_coef * rep_loss
            
            # 6. Psyche regularization (ИСПРАВЛЕНО)
            if self.personality_core is not None:
                try:
                    if hasattr(self.personality_core, 'compute_regularization_loss'):
                        psyche_loss = self.personality_core.compute_regularization_loss()
                        if isinstance(psyche_loss, torch.Tensor) and psyche_loss.requires_grad:
                            psy_coef = getattr(self.config, 'psyche_coef', 0.05)
                            loss = loss + psy_coef * psyche_loss
                except Exception as e:
                    logger.debug(f"⚠ Psyche loss error: {e}")
            
            # 7. Memory Gate aux-loss
            if memory_result:
                if 'gate_aux_loss' in memory_result and memory_result['gate_aux_loss'] is not None:
                    gate_loss = memory_result['gate_aux_loss']
                    if isinstance(gate_loss, torch.Tensor) and gate_loss.requires_grad:
                        loss = loss + 0.01 * torch.clamp(gate_loss, min=0.0)
                
                if 'gate_weights_aux_loss' in memory_result and memory_result['gate_weights_aux_loss'] is not None:
                    weights_loss = memory_result['gate_weights_aux_loss']
                    if isinstance(weights_loss, torch.Tensor) and weights_loss.requires_grad:
                        loss = loss + 0.01 * weights_loss
        
        # ===================================================================
        # 🔥 SEMANTIC ANCHORS LOSS
        # ===================================================================
        if self.training and input_ids is not None and loss is not None and hasattr(self, 'semantic_anchors'):
            try:
                token_embeds = self.transformer.wte(input_ids)
                anchor_loss = self.compute_anchor_loss(token_embeds)
                if isinstance(anchor_loss, torch.Tensor) and anchor_loss.requires_grad:
                    loss = loss + anchor_loss
            except Exception as e:
                if self.step_count % 500 == 0:
                    logger.debug(f"⚠ Anchor loss error: {e}")
        
        if self.step_count % 200 == 0 and loss is not None:
            logger.info(f"📊 TOTAL LOSS: {loss.item():.6f}")
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=all_hidden_states,
            attentions=attention_patterns,
        )
    
    def get_meta_cognitive_state(self) -> Dict:
        """Возвращает текущее мета-когнитивное состояние модели (обновлённое для v3.0)"""
        if hasattr(self, '_last_meta') and self._last_meta is not None:
            return self._last_meta
        
        if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
            if hasattr(self.adaptive_memory, 'meta_predictor'):
                meta = self.adaptive_memory.meta_predictor._last_meta
                if meta is not None:
                    return {
                        'confidence': meta['confidence'].mean().item(),
                        'doubt': meta['doubt'].mean().item(),
                        'curiosity': meta['curiosity'].mean().item(),
                        'readiness': meta['readiness'].mean().item(),
                        'need_recheck': meta['need_recheck'].mean().item(),
                        'decision': meta.get('decision', ['act'])[0] if meta.get('decision') else 'act',
                        'novelty': meta['novelty'].mean().item(),
                        'self_doubt': meta['self_doubt'].mean().item(),
                        'meta_loss': meta['meta_loss'].item() if hasattr(meta['meta_loss'], 'item') else 0
                    }
        
        return {
            'confidence': 0.5,
            'doubt': 0.0,
            'curiosity': 0.0,
            'readiness': 0.5,
            'need_recheck': 0.0,
            'decision': 'act',
            'novelty': 0.0,
            'self_doubt': 0.0,
            'message': 'Meta-cognitive state not available'
        }
    
    def _log_cognitive_state(self, memory_controls, memory_result, surprise, hebb_loss):
        """Логирование состояния когнитивной системы"""
        try:
            if self.personality_core is not None:
                was_training = self.training
                if was_training:
                    self.eval()
                    self.personality_core.eval()
                
                try:
                    report = self.personality_core.get_detailed_report()
                    mood_val = report["mood"]["state"]
                    mood_emoji = "😊" if mood_val > 0.5 else "🙂" if mood_val > 0.1 else "😐" if mood_val > -0.1 else "😕" if mood_val > -0.5 else "😔"
                    
                    current_step = self.step_count
                    if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
                        current_step = self.adaptive_memory.step_count
                    
                    logger.info(f"\n🧠 КОГНИТИВНАЯ СИСТЕМА [шаг {current_step}]")
                    logger.info(f"🎭 Настроение: {mood_emoji} {mood_val:+.3f}")
                    logger.info(f"🧠 Контроллер: write={memory_controls.get('hebb_write_gate', 0):.2f}, "
                               f"lr={memory_controls.get('hebb_lr_mult', 0):.2f}, "
                               f"stability={memory_controls.get('stability_factor', 0):.2f}")
                except Exception as e:
                    logger.debug(f"⚠ Ошибка логирования психики: {e}")
                finally:
                    if was_training:
                        self.train()
                        self.personality_core.train()
            
            if self.use_hebb_layers and self.hebb_layers is not None:
                total_active = 0
                total_initialized = 0
                total_updates = 0
                total_reads = 0
                total_utility = 0.0
                n_blocks = 0
                
                total_accum_pending = 0
                total_new_slots_pending = 0
                max_accum_weight = 0.0
                
                for layer in self.hebb_layers:
                    if layer._is_meta():
                        continue
                    stats = layer.get_stats()
                    
                    total_active += stats.get("active_slots", 0)
                    total_initialized += stats.get("initialized", 0)
                    total_updates += stats.get("total_updates", 0)
                    total_reads += stats.get("total_reads", 0)
                    total_utility += stats.get("utility_mean", 0.0)
                    
                    total_accum_pending += stats.get("accum_pending", 0)
                    total_new_slots_pending += stats.get("new_slot_present", 0)
                    
                    n_blocks += 1
                
                if n_blocks > 0:
                    logger.info(
                        f"🧠 HEBB: active={total_active}/{total_initialized} "
                        f"(init={total_initialized}/{n_blocks*64}), "
                        f"updates={total_updates}, reads={total_reads}, "
                        f"utility={total_utility/n_blocks:.3f}, "
                        f"surprise={surprise:.3f}, aux_loss={hebb_loss:.4f}"
                    )
                    
                    if total_accum_pending > 0 or total_new_slots_pending > 0:
                        logger.info(
                            f"📊 НАКОПЛЕНИЕ: слотов с данными={total_accum_pending}, "
                            f"новых слотов готово={total_new_slots_pending}, "
                            f"порог={self.hebb_layers[0].accum_min_weight if self.hebb_layers else 0.05:.2f}"
                        )
                    else:
                        logger.info("📊 НАКОПЛЕНИЕ: 0 (ждем накопления)")
                    
                    if self.adaptive_memory is not None:
                        thoughts_count = len(self.adaptive_memory.data_manager.thought_chains)
                        logger.info(f"📓 РЕАЛЬНЫЙ Блокнот: {thoughts_count} мыслей")   
                        
        except Exception as e:
            logger.debug(f"⚠ Ошибка логирования: {e}")
 
    def get_symbiosis_control(self) -> Dict:
        """Управление симбиозом"""
        if not hasattr(self, 'experience_exchange'):
            return {"error": "Experience exchange not initialized"}
        
        report = self.experience_exchange.get_exchange_report()
        
        controls = {
            "exchange_strength": self.experience_exchange.exchange_strength,
            "alignment_score": float(self.experience_exchange.alignment_score.item()),
            "mutual_reward": float(self.experience_exchange.mutual_reward.item()),
            "transformer_experiences": len(self.experience_exchange.transformer_experience_bank),
            "mamba_experiences": len(self.experience_exchange.mamba_experience_bank),
            "exchange_counter": self.experience_exchange.exchange_counter,
        }
        
        return {**report, **controls}

    def adjust_symbiosis_strength(self, strength: float):
        """Ручная настройка силы симбиоза"""
        if hasattr(self, 'experience_exchange'):
            self.experience_exchange.exchange_strength = max(0.1, min(0.5, strength))
            logger.info(f"🔄 Сила симбиоза установлена: {strength:.3f}")
    
    def clear_experience_banks(self, keep_last: int = 20):
        """Очистка банков опыта"""
        if hasattr(self, 'experience_exchange'):
            self.experience_exchange.clear_banks(keep_last)
    
    def trigger_manual_exchange(self, mode: str = 'mutual_enhancement'):
        """Ручной запуск обмена опытом"""
        if not hasattr(self, 'experience_exchange'):
            logger.warning("⚠ Опытный обмен не инициализирован")
            return
        
        device = next(self.parameters()).device
        dummy_transformer = torch.randn(1, 5, self.config.n_embd, device=device)
        dummy_mamba = torch.randn(1, 3, self.config.slot_size, device=device)
        
        with torch.no_grad():
            self.experience_exchange.exchange_experiences(
                dummy_transformer, dummy_mamba, mode=mode
            )
        
        report = self.experience_exchange.get_exchange_report()
        
        logger.info(f"🔄 РУЧНОЙ ОБМЕН [{mode}]:")
        logger.info(f"   Выравнивание: {report['alignment']:.3f}")
        logger.info(f"   Взаимная награда: {report['mutual_reward']:.3f}")
        logger.info(f"   Опыт T/M: {report['transformer_experiences']['count']}/{report['mamba_experiences']['count']}")
        
        return report
 
    def on_event(self, event_type: str, callback):
        """Добавляем callback в deque"""
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = deque(maxlen=10)
        self._event_callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """Безопасный вызов callback'ов"""
        for callback in self._event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.debug(f"Ошибка callback {event_type}: {e}")
  
    def get_psyche_state(self) -> Optional[Dict]:
        if self.personality_core is not None:
            return self.personality_core.get_psyche_state()
        return None
    
    def set_psyche_state(self, state: Dict):
        if self.personality_core is not None and state is not None:
            self.personality_core.set_psyche_state(state)
    
    def trigger_psyche_experience(self, action_type: str = "explore", intensity: float = 0.5):
        if self.personality_core is None:
            logger.warning("⚠ Психика не активирована")
            return
        
        self.personality_core.process_experience(action_type, intensity)
        logger.info(f"🧠 Искусственный опыт: {action_type} (интенсивность: {intensity})")
    
    def prepare_generation_params(self, input_ids=None, **kwargs):
        """
        🔥 УЛУЧШЕННЫЙ: Интеграция с мета-сетью v3.2
        Параметры генерации теперь учитывают внутренний голос модели
        """
        # Базовые параметры от психики
        if self.personality_core is None:
            return self._default_params(kwargs)
        
        self.personality_core.tick(training=False)
        report = self.personality_core.get_detailed_report()
        
        mood = report.get('mood', {}).get('state', 0.0)
        entropy = report.get('system_state', {}).get('entropy', 0.5)
        stability = report.get('system_state', {}).get('stability', 0.5)
        creativity = report.get('traits', {}).get('creativity', 0.5)
        
        drives = report.get('drives', {})
        curiosity_level = drives.get('novelty', 0.5)
        fatigue_level = drives.get('fatigue', 0.5)
        
        # ===========================================================
        # 🔥 ИНТЕГРАЦИЯ С МЕТА-СЕТЬЮ v3.2
        # ===========================================================
        meta_state = self.get_meta_cognitive_state() if hasattr(self, 'get_meta_cognitive_state') else {}
        
        # Извлекаем мета-показатели
        meta_confidence = meta_state.get('confidence', 0.5)
        meta_doubt = meta_state.get('doubt', 0.3)
        meta_curiosity = meta_state.get('curiosity', 0.5)
        meta_decision = meta_state.get('decision', 'act')
        
        # ===========================================================
        # 🔥 АДАПТАЦИЯ ПАРАМЕТРОВ НА ОСНОВЕ МЕТА-СОСТОЯНИЯ
        # ===========================================================
        
        # 1. Температура: 
        #    - Высокое любопытство → выше температура (больше экспериментов)
        #    - Высокая уверенность → ниже температура (точнее)
        temp = 0.70 + (curiosity_level * 0.4) + (creativity * 0.3) - (stability * 0.1)
        temp = temp * (1.0 + meta_curiosity * 0.2)  # мета-любопытство увеличивает
        temp = temp * (1.0 - meta_confidence * 0.15)  # мета-уверенность уменьшает
        temp = max(0.55, min(1.6, temp))
        
        # 2. Top_p (разнообразие):
        #    - Высокое любопытство → шире выбор
        #    - Высокая уверенность → уже выбор
        top_p = 0.92 + ((curiosity_level - 0.5) * 0.15) - (fatigue_level * 0.12) + ((entropy - 0.5) * 0.1)
        top_p = top_p * (1.0 + meta_curiosity * 0.1)  # любопытство расширяет
        top_p = top_p * (1.0 - meta_confidence * 0.1)  # уверенность сужает
        top_p = max(0.75, min(0.99, top_p))
        
        # 3. Top_k
        top_k = int(40 + (curiosity_level * 30) - (fatigue_level * 20) + (entropy * 10))
        top_k = int(top_k * (1.0 + meta_curiosity * 0.15))
        top_k = max(20, min(100, top_k))
        
        # 4. Repetition penalty
        rep_penalty = 1.05 + (fatigue_level * 0.2) + ((1 - creativity) * 0.1)
        # Мета-сомнение увеличивает penalty (меньше повторений)
        rep_penalty = rep_penalty * (1.0 + meta_doubt * 0.2)
        rep_penalty = max(1.02, min(1.35, rep_penalty))
        
        # 5. Минимальное количество токенов для keep
        creativity_norm = (creativity + 1) / 2 
        min_keep = int(2 + (6 - 2) * creativity_norm)
        # Если решение "think" или "deep_rethink" — увеличиваем
        if meta_decision in ["think", "deep_rethink"]:
            min_keep = min_keep * 2
        min_keep = max(2, min(12, min_keep))
        
        # 6. Максимальное количество токенов
        base_max = kwargs.get('max_new_tokens', 1024)
        stability_factor = max(0.4, stability)
        curiosity_factor = 1.0 + (curiosity_level * 0.25)
        max_new_tokens_clamped = int(base_max * stability_factor * curiosity_factor)
        
        # 🔥 МЕТА-КОРРЕКЦИЯ: если решение "explore" — генерируем больше
        if meta_decision == "explore":
            max_new_tokens_clamped = int(max_new_tokens_clamped * 1.3)
        # Если "act" — можно меньше
        elif meta_decision == "act":
            max_new_tokens_clamped = int(max_new_tokens_clamped * 0.9)
        
        max_new_tokens_clamped = max(60, min(base_max, max_new_tokens_clamped))
        
        # 7. CoT множитель
        cot_multiplier = 1.1 + (entropy - 0.5) * 0.6
        # Если решение "deep_rethink" — увеличиваем CoT
        if meta_decision == "deep_rethink":
            cot_multiplier = cot_multiplier * 1.3
        cot_multiplier = max(1.1, min(1.6, cot_multiplier))
        
        # 8. No repeat ngram size
        no_repeat = 2 + int(stability * 1.5)
        no_repeat = min(4, no_repeat)
        
        # 9. Early stopping
        # Если уверенность высокая — можно останавливаться раньше
        early_stopping = meta_confidence > 0.7
        
        # 10. Добавляем параметры для борьбы с короткими генерациями
        min_new_tokens = kwargs.get('min_new_tokens', 30)
        # Если модель слишком быстро останавливается, увеличиваем min_new_tokens
        if meta_decision in ["explore", "deep_rethink"]:
            min_new_tokens = max(50, min_new_tokens)
        
        return {
            'temperature': temp,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': rep_penalty,
            'min_tokens_to_keep': min_keep,
            'max_new_tokens_clamped': max_new_tokens_clamped,
            'min_new_tokens': min_new_tokens,
            'cot_temperature_multiplier': cot_multiplier,
            'no_repeat_ngram_size': no_repeat,
            'early_stopping': early_stopping,
            'burnout_active': False,
            # Мета-показатели для логирования
            'meta_confidence': meta_confidence,
            'meta_doubt': meta_doubt,
            'meta_curiosity': meta_curiosity,
            'meta_decision': meta_decision,
            # Остальные
            'mood': mood,
            'entropy': entropy,
            'creativity': creativity,
            'stability': stability,
            'curiosity_level': curiosity_level,
            'fatigue_level': fatigue_level
        }
    
    def _default_params(self, kwargs):
        """Параметры по умолчанию, если психика отключена"""
        return {
            'temperature': getattr(self.generation_config, 'temperature', 0.85),
            'top_p': getattr(self.generation_config, 'top_p', 0.92),
            'top_k': 50,
            'repetition_penalty': getattr(self.generation_config, 'repetition_penalty', 1.15),
            'min_tokens_to_keep': 5,
            'max_new_tokens_clamped': kwargs.get('max_new_tokens', 1024),
            'cot_temperature_multiplier': 1.3,
            'no_repeat_ngram_size': 0,
            'burnout_active': False,
            'mood': 0.0,
            'entropy': 0.5,
            'creativity': 0.5,
            'stability': 0.5,
            'curiosity_level': 0.5,
            'fatigue_level': 0.5
        }
    
# ===================================================================
# 🧠 ГЛАВНЫЙ МЕТОД ГЕНЕРАЦИИ (ЕДИНСТВЕННАЯ ТОЧКА ВХОДА)
# ===================================================================
    def generate(self, inputs=None, **kwargs):
        """
        ⚡ ЕДИНСТВЕННАЯ ТОЧКА ВХОДА ДЛЯ ГЕНЕРАЦИИ
        Исправлена рекурсия, хуки, CoT и обработка ошибок
        """
        # 🔥 Защита от сна во время генерации
        if hasattr(self, 'adaptive_memory'):
            self.adaptive_memory.pre_generation_hook()

        try:
            input_ids = self._prepare_input_ids(inputs, kwargs)
            attention_mask = kwargs.pop('attention_mask', None)
            max_new_tokens = kwargs.pop('max_new_tokens', 512)

            # Основной путь — через психику
            if hasattr(self, 'generate_with_psyche'):
                result, _ = self.generate_with_psyche(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    record_experience=True,
                    **kwargs
                )
                return result
            else:
                # Fallback
                return self._standard_generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs)

        finally:
            # 🔥 Гарантированно снимаем блокировку даже при ошибке
            if hasattr(self, 'adaptive_memory'):
                self.adaptive_memory.post_generation_hook()

    def _prepare_input_ids(self, inputs, kwargs):
        """
        Унифицирует входные данные в тензор input_ids.
        Поддерживает:
        - inputs как строку
        - inputs как тензор
        - inputs как список
        - keyword аргумент 'input_ids'
        - пустой ввод (bos_token)
        """
        tokenizer = kwargs.get('tokenizer')
        
        if inputs is not None:
            if isinstance(inputs, torch.Tensor):
                return inputs.to(self.device)
            elif isinstance(inputs, str):
                if tokenizer is None:
                    raise ValueError("Для строкового ввода нужен tokenizer")
                return tokenizer.encode(inputs, return_tensors="pt").to(self.device)
            else:
                # список чисел
                return torch.tensor([inputs], dtype=torch.long, device=self.device)
        
        elif 'input_ids' in kwargs:
            input_ids = kwargs.pop('input_ids')
            if isinstance(input_ids, torch.Tensor):
                return input_ids.to(self.device)
            else:
                return torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        else:
            # Пустой ввод — bos_token
            return torch.tensor([[self.config.bos_token_id]], device=self.device)

    def generate_with_psyche(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        use_chain_of_thought: bool = False,
        record_experience: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """Генерация с психикой — ИНТЕГРАЦИЯ С МЕТА-СЕТЬЮ v3.2"""
        self.eval()
        device = input_ids.device

        params = self.prepare_generation_params(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # 🔥 ЛОГИРУЕМ МЕТА-СОСТОЯНИЕ ПЕРЕД ГЕНЕРАЦИЕЙ
        if params.get('meta_decision'):
            logger.info(f"🧠 Мета-решение: {params['meta_decision']} "
                       f"(увер={params['meta_confidence']:.3f}, "
                       f"сомнение={params['meta_doubt']:.3f}, "
                       f"любопытство={params['meta_curiosity']:.3f})")

        # CoT (цепочка мыслей)
        if use_chain_of_thought and self.personality_core:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            # 🔥 Учитываем мета-решение для длины CoT
            cot_max_tokens = 512 if params.get('meta_decision') == 'deep_rethink' else 256
            cot_ids = self._generate_internal_thought(
                input_ids, attention_mask,
                temperature=params['temperature'] * params.get('cot_temperature_multiplier', 1.3),
                top_p=params['top_p'],
                max_new_tokens=cot_max_tokens
            )
            input_ids = torch.cat([input_ids, cot_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(cot_ids)], dim=-1)

        # Запоминаем состояние для RL
        prev_drives = None
        if record_experience and self.personality_core is not None:
            prev_drives = self.personality_core.get_drive_values().detach().clone()

        # Генерация
        with torch.no_grad():
            output = super(Thalia, self).generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=params['max_new_tokens_clamped'],
                    min_new_tokens=params.get('min_new_tokens', 30),  # 🔥 минимальная длина
                    do_sample=True,
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params.get('top_k', 50),
                    repetition_penalty=params['repetition_penalty'],
                    no_repeat_ngram_size=params.get('no_repeat_ngram_size', 0),
                    min_tokens_to_keep=params['min_tokens_to_keep'],
                    eos_token_id=self.config.eos_token_id,
                    pad_token_id=self.config.pad_token_id,
                    early_stopping=params.get('early_stopping', True),
                    return_dict_in_generate=True,
                ),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['max_new_tokens', 'temperature', 'top_p', 'top_k', 
                               'repetition_penalty', 'min_tokens_to_keep', 'min_new_tokens']}
            )

        generated = self._extract_sequences(output)
        
        # 🔥 ПРОВЕРКА ДЛИНЫ ГЕНЕРАЦИИ
        generated_len = generated.shape[1] - input_ids.shape[1]
        if generated_len < 30:
            logger.warning(f"⚠️ Короткая генерация: {generated_len} токенов. "
                          f"Мета-решение: {params.get('meta_decision', '?')}")

        # RL-обновление
        if record_experience and self.personality_core is not None and prev_drives is not None:
            current_drives = self.personality_core.get_drive_values().detach()
            reward = self.personality_core.calculate_intrinsic_reward(prev_drives, current_drives, user_feedback=0.5)
            lr = 0.012 if reward > 0.1 else 0.006 if reward > 0 else 0.003
            if reward < -0.15:
                logger.warning(f"⚠ Негативная награда ({reward:.3f}) → откат")
                self.personality_core.drive_values.data.copy_(prev_drives)
                self.personality_core.trait_raw.data.mul_(0.95)
                lr = 0.001
            self.personality_core.apply_reinforcement(reward, learning_rate=lr)

        metadata = {
            'gen_params': params,
            'rl_data': {'reward': reward if 'reward' in locals() else 0.0},
            'generated_len': generated_len
        }
        return generated, metadata
 
    def _generate_internal_thought(self, input_ids, attention_mask, temperature=1.4, top_p=0.97, max_new_tokens=256):
        """Внутренняя мысль (CoT) — с параметром max_new_tokens"""
        with torch.no_grad():
            output = super(Thalia, self).generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,  # 🔥 теперь параметр
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    eos_token_id=self.config.eos_token_id,
                    pad_token_id=self.config.pad_token_id,
                    return_dict_in_generate=True,
                )
            )

        thought_ids = self._extract_sequences(output)
        thought_ids = thought_ids[:, input_ids.shape[-1]:]

        if self.personality_core is not None:
            self.personality_core.process_experience("think", 0.3)

        return thought_ids
 
# ===================================================================
# 📝 СТАНДАРТНАЯ ГЕНЕРАЦИЯ (с учётом психики)
# ===================================================================
    def _standard_generate(self, input_ids: torch.Tensor, attention_mask=None, max_new_tokens=512, **kwargs):
        """Стандартная генерация (fallback)"""
        params = self.prepare_generation_params(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)

        gen_config = GenerationConfig(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            max_new_tokens=params['max_new_tokens_clamped'],   # ← ИСПРАВЛЕНО
            do_sample=True,
            temperature=params['temperature'],
            top_p=params['top_p'],
            top_k=params.get('top_k', 50),
            repetition_penalty=params['repetition_penalty'],
            no_repeat_ngram_size=params.get('no_repeat_ngram_size', 0),
            min_tokens_to_keep=params['min_tokens_to_keep'],
            early_stopping=True,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )

        safe_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'temperature', 'top_p', 'top_k', 'repetition_penalty',
            'max_new_tokens', 'min_tokens_to_keep'
        ]}

        output = super().generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            **safe_kwargs
        )

        return self._extract_sequences(output)

    def _extract_sequences(self, output):
        """Извлекает тензор последовательности из результата generate()"""
        if hasattr(output, 'sequences'):
            return output.sequences
        elif isinstance(output, tuple):
            return output[0]
        else:
            return output
 
    def bind_tokenizer(self, tokenizer):
        """Привязывает токенизатор к модели"""
        self.tokenizer = tokenizer
        logger.info("🔗 Токенизатор привязан к ThaliaModel") 
 
    def save_pretrained(self, save_directory, **kwargs):
        """
        🔥 ИСПРАВЛЕННЫЙ save_pretrained для Thalia v9.2
        Сохраняет токенизатор, мета-предиктор v3.0 и все компоненты модели
        """
        current_device = next(self.parameters()).device
        logger.info(f"💾 Сохранение чекпоинта, текущее устройство: {current_device}")
        
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
            logger.info(f"📊 transformer.wte.weight: {self.transformer.wte.weight.shape}, device: {self.transformer.wte.weight.device}")
        else:
            logger.warning("⚠ transformer.wte.weight не найден в модели!")
        
        if hasattr(self, 'experience_exchange'):
            logger.info("🧪 Проверка буферов перед сохранением...")
            self.experience_exchange.test_buffer_integrity()
        
        # Сохраняем основную модель
        super().save_pretrained(save_directory, **kwargs)
        
        # Сохраняем generation_config
        if hasattr(self, 'generation_config') and self.generation_config:
            try:
                self.generation_config.save_pretrained(save_directory)
            except Exception as e:
                logger.warning(f"⚠ Не удалось сохранить generation_config: {e}")
        
        try:
            state = {
                'step_count': self.step_count,
                'thalia_version': '9.2_hebb_centroid_memory_v3',
            }
            
            # ===========================================================
            # 🔥 СОХРАНЯЕМ МЕТА-ПРЕДИКТОР v3.0 (упрощённая версия)
            # ===========================================================
            if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
                if hasattr(self.adaptive_memory, 'meta_predictor') and self.adaptive_memory.meta_predictor is not None:
                    try:
                        # Используем метод get_state()
                        meta_state = self.adaptive_memory.meta_predictor.get_state()
                        state['meta_predictor_state'] = meta_state
                        
                        # Логируем статистику
                        if 'stats' in meta_state:
                            stats = meta_state['stats']
                            logger.info(f"🧠 Сохранён мета-предиктор v3.0: "
                                       f"conf={stats.get('avg_confidence', 0):.3f}, "
                                       f"doubt={stats.get('avg_doubt', 0):.3f}, "
                                       f"curiosity={stats.get('avg_curiosity', 0):.3f}")
                        else:
                            logger.info(f"🧠 Сохранён мета-предиктор v3.0")
                        
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка сохранения мета-предиктора: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
            
            # ===========================================================
            # 🔥 ОСТАЛЬНЫЕ КОМПОНЕНТЫ
            # ===========================================================
            
            if hasattr(self, 'experience_exchange'):
                try:
                    exchange_state = self.experience_exchange.get_state()
                    state['exchange_state'] = exchange_state
                    
                    if 'pair_count' in exchange_state:
                        logger.info(f"📦 Сохранены буферы обмена опытом: "
                                   f"pair_count={exchange_state['pair_count'].item()}, "
                                   f"pair_ptr={exchange_state['pair_ptr'].item()}")
                    else:
                        logger.warning("⚠ exchange_state не содержит pair_count!")
                        
                except Exception as e:
                    logger.warning(f"⚠ Ошибка сохранения состояния обмена: {e}")
            
            if hasattr(self, '_last_generation_embed'):
                embed = self._last_generation_embed
                if embed.dim() == 2 and embed.size(0) == 1:
                    embed = embed.squeeze(0)
                state['_last_generation_embed'] = embed.cpu().clone()
            
            if self.adaptive_memory is not None:
                try:
                    # Сохраняем только веса, не буферы (они сохраняются отдельно)
                    adaptive_state = {}
                    for name, param in self.adaptive_memory.state_dict().items():
                        if not name.startswith('meta_predictor'):  # мета-предиктор уже сохранён
                            adaptive_state[name] = param.cpu().clone()
                    state['adaptive_memory_state'] = adaptive_state
                    
                    if hasattr(self.adaptive_memory, 'sedimentary_slots'):
                        sedimentary_data = {
                            'sedimentary_slots': self.adaptive_memory.sedimentary_slots.data.cpu().clone(),
                            'hebb_matrix': self.adaptive_memory.hebb_matrix.data.cpu().clone(),
                            'slot_activation': self.adaptive_memory.slot_activation.cpu().clone(),
                            'slot_age': self.adaptive_memory.slot_age.cpu().clone(),
                            'slot_frozen': self.adaptive_memory.slot_frozen.cpu().clone(),
                        }
                        state['sedimentary_memory'] = sedimentary_data
                    
                    if hasattr(self.adaptive_memory, 'data_manager'):
                        state['thought_chains'] = getattr(
                            self.adaptive_memory.data_manager, 
                            'thought_chains', 
                            []
                        )
                    
                    logger.info(f"🧠 Сохранено состояние адаптивной памяти v9.2")
                    
                except Exception as mem_e:
                    logger.warning(f"⚠ Ошибка сохранения адаптивной памяти: {mem_e}")
                    state['adaptive_memory_state'] = None
            
            if self.use_hebb_layers and hasattr(self, 'hebb_layers') and self.hebb_layers is not None:
                try:
                    hebb_states = []
                    for layer in self.hebb_layers:
                        hebb_states.append(layer.state_dict())
                    state['hebb_states'] = hebb_states
                    
                    hebb_meta = []
                    for i, layer in enumerate(self.hebb_layers):
                        stats = layer.get_stats()
                        hebb_meta.append({
                            'layer': i,
                            'active_slots': stats['active_slots'],
                            'total_updates': stats['total_updates'],
                            'total_reads': stats['total_reads'],
                            'avg_utility': stats['utility_mean'],
                            'aux_loss': stats['aux_loss'],
                            'temperature': stats['temperature'],
                        })
                    state['hebb_metadata'] = hebb_meta
                    
                    logger.info(f"🧠 Сохранено Hebb-памяти: {len(hebb_states)} слоев, "
                               f"активных слотов: {sum(m['active_slots'] for m in hebb_meta)}")
                    
                except Exception as hebb_e:
                    logger.warning(f"⚠ Ошибка сохранения Hebb-памяти: {hebb_e}")
            
            centroid_was_on_cpu = False
            if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
                if hasattr(self.adaptive_memory, 'centroid_memory'):
                    try:
                        original_centroid_device = self.adaptive_memory.centroid_memory.centroids.device
                        
                        centroid_state = self.adaptive_memory.centroid_memory.cpu().state_dict_custom()
                        
                        for key, value in centroid_state.items():
                            if isinstance(value, torch.Tensor):
                                centroid_state[key] = value.cpu()
                            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                                centroid_state[key] = [v.cpu() for v in value]
                        
                        state['centroid_memory_state'] = centroid_state
                        
                        stats = self.adaptive_memory.centroid_memory.get_stats()
                        logger.info(f"🧠 Сохранено центроидной памяти: {stats['active_slots']}/{stats['total_slots']} слотов")
                        
                        if hasattr(self.adaptive_memory.centroid_memory, 'linker_status'):
                            linker_info = self.adaptive_memory.centroid_memory.linker_status()
                            if linker_info and linker_info.get('updates', 0) > 0:
                                logger.info(f"   • Neural Linker: обновлений={linker_info['updates']}, "
                                           f"история={linker_info.get('history_size', 0)}")
                        
                        if original_centroid_device != torch.device('cpu'):
                            logger.info(f"🔄 Возвращаю центроидную память на {original_centroid_device}")
                            self.adaptive_memory.centroid_memory = self.adaptive_memory.centroid_memory.to(original_centroid_device)
                            self.adaptive_memory.centroid_memory.reset_cache()
                            centroid_was_on_cpu = True
                            
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка сохранения центроидной памяти: {e}")
            
            # ===========================================================
            # 🔥 СОХРАНЯЕМ СОСТОЯНИЕ ПСИХИКИ (исправлено)
            # ===========================================================
            if self.personality_core is not None:
                try:
                    psyche_state = self.personality_core.get_psyche_state()
                    state['psyche_state'] = psyche_state
                    
                    # 🔥 УДАЛЯЕМ НЕСУЩЕСТВУЮЩИЕ АТРИБУТЫ
                    # slot_curiosity и recovery_level отсутствуют в DynamicPsycheCoreV6
                    
                    controller_config = {
                        "current_controls": self.personality_core.get_memory_control_signals(),
                        "exploration_rate": float(self.personality_core.controller.exploration_rate.item()) 
                            if hasattr(self.personality_core.controller, 'exploration_rate') else 1.0,
                        "training_steps": int(self.personality_core.controller.training_steps.item()) 
                            if hasattr(self.personality_core.controller, 'training_steps') else 0,
                        "baseline": float(self.personality_core.controller_reward_baseline.item()) 
                            if hasattr(self.personality_core, 'controller_reward_baseline') else 0.0,
                        "version": "v9.2_gaussian_controller"
                    }
                    state['controller_debug'] = controller_config
                    
                    logger.info(f"🧠 Сохранено состояние психики v9.2")
                    
                except Exception as psyche_e:
                    logger.warning(f"⚠ Ошибка сохранения психики: {psyche_e}")
                    state['psyche_state'] = None
            
            torch.save(state, f"{save_directory}/thalia_enhanced_state.pt", pickle_protocol=4)
            
            logger.info(f"✅ Сохранено улучшенное состояние Thalia v9.2")
            logger.info(f"📊 Версия: {state.get('thalia_version', 'unknown')}")
            logger.info(f"🧠 Шаг: {self.step_count}")
            
            if 'meta_predictor_state' in state:
                logger.info(f"🧠 Мета-предиктор v3.0: сохранён")
            
            if 'adaptive_memory_state' in state and state['adaptive_memory_state']:
                logger.info(f"💾 Адаптивная память: {len(state['adaptive_memory_state'])} параметров")
            
            if 'hebb_states' in state:
                logger.info(f"🧠 Hebb-память: {len(state['hebb_states'])} слоев")
            
            if 'centroid_memory_state' in state:
                logger.info(f"🧠 Центроидная память: сохранена")
            
            if 'exchange_state' in state:
                exchange_state = state['exchange_state']
                if 'pair_count' in exchange_state:
                    logger.info(f"🔄 Состояние обмена: {exchange_state['pair_count'].item()} пар")
            
            if centroid_was_on_cpu:
                logger.info(f"✅ Центроидная память возвращена на исходное устройство")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка сохранения: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"🔄 Финальная синхронизация: возвращаю модель на {current_device}")
        self = self.to(current_device)
        
        if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
            if hasattr(self.adaptive_memory, 'centroid_memory'):
                self.adaptive_memory.centroid_memory = self.adaptive_memory.centroid_memory.to(current_device)
                if hasattr(self.adaptive_memory.centroid_memory, 'reset_cache'):
                    self.adaptive_memory.centroid_memory.reset_cache()
                self.adaptive_memory.centroid_memory._cache_dirty = True
        
        logger.info(f"✅ Модель полностью возвращена на {current_device}")


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        🔥 ИСПРАВЛЕННЫЙ from_pretrained с загрузкой мета-предиктора
        """
        import os
        
        try:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            
            if not os.path.exists(config_path):
                logger.warning(f"⚠ config.json не найден в {pretrained_model_name_or_path}")
                return super(Thalia, cls).from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    **kwargs
                )
            
            config = ThaliaConfig.from_json_file(config_path)
            config.model_type = "thalia"
            logger.info(f"📊 Конфиг из чекпоинта: n_embd={config.n_embd}, n_layer={config.n_layer}")
            
            safe_kwargs = kwargs.copy()
            safe_kwargs.pop('state_dict', None)
            safe_kwargs.pop('strict', None)
            
            model = super(Thalia, cls).from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                *model_args,
                **safe_kwargs
            )
            
            device = kwargs.get('device', model.device)
            
            # ===========================================================
            # 🔥 АВТОМАТИЧЕСКАЯ ЗАГРУЗКА ТОКЕНИЗАТОРА
            # ===========================================================
            try:
                from transformers import AutoTokenizer
                logger.info(f"🔄 Попытка автозагрузки токенизатора из {pretrained_model_name_or_path}")
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
                model.bind_tokenizer(tokenizer)
                logger.info("✅ Токенизатор успешно привязан к ThaliaModel")
            except Exception as e:
                logger.warning(f"⚠️ Автозагрузка токенизатора не удалась: {e}")
            
            # ===========================================================
            # 🔥 ЗАГРУЖАЕМ УЛУЧШЕННОЕ СОСТОЯНИЕ
            # ===========================================================
            state_path = os.path.join(pretrained_model_name_or_path, "thalia_enhanced_state.pt")
            if os.path.exists(state_path):
                logger.info(f"🧠 Найден файл улучшенного состояния: {state_path}")
                try:
                    enhanced_state = torch.load(state_path, map_location='cpu', weights_only=False)
                    
                    # ===========================================================
                    # 🔥 ЗАГРУЖАЕМ МЕТА-ПРЕДИКТОР (упрощённая версия)
                    # ===========================================================
                    if 'meta_predictor_state' in enhanced_state and hasattr(model, 'adaptive_memory'):
                        if model.adaptive_memory and hasattr(model.adaptive_memory, 'meta_predictor'):
                            meta_state = enhanced_state['meta_predictor_state']
                            try:
                                # Используем метод load_state() если есть
                                if hasattr(model.adaptive_memory.meta_predictor, 'load_state'):
                                    model.adaptive_memory.meta_predictor.load_state(meta_state, device)
                                else:
                                    # Fallback: загружаем напрямую
                                    mp = model.adaptive_memory.meta_predictor
                                    
                                    if 'predictor_state_dict' in meta_state:
                                        mp.predictor.load_state_dict(meta_state['predictor_state_dict'], strict=False)
                                    
                                    if 'meta_network_state_dict' in meta_state:
                                        try:
                                            mp.meta_network.load_state_dict(meta_state['meta_network_state_dict'], strict=False)
                                        except Exception:
                                            # Миграция 3→5
                                            old_sd = meta_state['meta_network_state_dict']
                                            new_sd = mp.meta_network.state_dict()
                                            for key in new_sd.keys():
                                                if key in old_sd and old_sd[key].shape == new_sd[key].shape:
                                                    new_sd[key] = old_sd[key]
                                                elif key in old_sd and key.endswith('weight') and old_sd[key].shape[0] == 3 and new_sd[key].shape[0] == 5:
                                                    new_sd[key][:3] = old_sd[key]
                                                elif key in old_sd and key.endswith('bias') and old_sd[key].shape[0] == 3 and new_sd[key].shape[0] == 5:
                                                    new_sd[key][:3] = old_sd[key]
                                            mp.meta_network.load_state_dict(new_sd, strict=False)
                                            logger.info(f"🔄 Мета-сеть: миграция 3→5 выходов выполнена")
                                    
                                    if 'buffers' in meta_state:
                                        for name, value in meta_state['buffers'].items():
                                            if hasattr(mp, name):
                                                buf = getattr(mp, name)
                                                if isinstance(buf, torch.Tensor) and isinstance(value, torch.Tensor):
                                                    if buf.shape == value.shape:
                                                        buf.data.copy_(value.to(device))
                                                    else:
                                                        logger.warning(f"⚠ Буфер {name}: shape mismatch {buf.shape} vs {value.shape}")
                                
                                logger.info(f"✅ Мета-предиктор v3.0 загружен")
                                
                            except Exception as meta_e:
                                logger.warning(f"⚠ Ошибка загрузки мета-предиктора: {meta_e}")
                                logger.info("🔄 Инициализируем мета-предиктор заново")
                                if hasattr(model.adaptive_memory.meta_predictor, '_init_weights'):
                                    model.adaptive_memory.meta_predictor._init_weights()
                    
                    # ===========================================================
                    # 🔥 ЗАГРУЖАЕМ ОСТАЛЬНЫЕ КОМПОНЕНТЫ
                    # ===========================================================
                    if 'psyche_state' in enhanced_state and model.personality_core is not None:
                        psyche_state = enhanced_state['psyche_state']
                        if psyche_state is not None:
                            model.personality_core.set_psyche_state(psyche_state)
                            logger.info(f"✅ Состояние психики загружено")
                    
                    if 'hebb_states' in enhanced_state and hasattr(model, 'hebb_layers') and model.hebb_layers:
                        hebb_states = enhanced_state['hebb_states']
                        if hebb_states:
                            for i, layer in enumerate(model.hebb_layers):
                                if i < len(hebb_states) and hebb_states[i] is not None:
                                    old_sd = hebb_states[i].copy()
                                    
                                    if 'slot_utility_ema' in old_sd:
                                        old_shape = old_sd['slot_utility_ema'].shape
                                        expected = (layer.num_slots,)
                                        
                                        if old_shape != expected:
                                            logger.warning(f"⚠️ Исправляем старый shape slot_utility_ema в слое {i}: {old_shape} → {expected}")
                                            if len(old_shape) == 2 and old_shape[0] == 10 and old_shape[1] == layer.num_slots:
                                                old_sd['slot_utility_ema'] = old_sd['slot_utility_ema'].mean(dim=0)
                                            else:
                                                old_sd['slot_utility_ema'] = torch.ones(layer.num_slots, device=old_sd['slot_utility_ema'].device) * 0.5
                                    
                                    if 'slot_stats' in old_sd and old_sd['slot_stats'].shape[0] != layer.num_slots:
                                        old_sd['slot_stats'] = torch.zeros(layer.num_slots, 4, device=old_sd['slot_stats'].device)
                                    
                                    layer.load_state_dict(old_sd, strict=False)
                            logger.info(f"✅ Hebb-слои загружены")
                    
                    if 'centroid_memory_state' in enhanced_state and hasattr(model, 'adaptive_memory'):
                        if model.adaptive_memory and hasattr(model.adaptive_memory, 'centroid_memory'):
                            centroid_state = enhanced_state['centroid_memory_state']
                            if centroid_state:
                                model.adaptive_memory.centroid_memory.load_state_dict_custom(centroid_state)
                                model.adaptive_memory.centroid_memory = model.adaptive_memory.centroid_memory.to(device)
                                stats = model.adaptive_memory.centroid_memory.get_stats()
                                logger.info(f"✅ Центроидная память загружена: {stats['active_slots']}/{stats['total_slots']} слотов")
                    
                    if 'exchange_state' in enhanced_state and hasattr(model, 'experience_exchange'):
                        try:
                            model.experience_exchange.set_state(enhanced_state['exchange_state'], device)
                            logger.info("✅ Буферы обмена опытом восстановлены")
                        except Exception as e:
                            logger.warning(f"⚠ Ошибка восстановления буферов обмена: {e}")
                    
                    if 'step_count' in enhanced_state:
                        model.step_count = enhanced_state['step_count']
                    
                    logger.info(f"✅ Улучшенное состояние загружено успешно")
                    
                except Exception as e:
                    logger.warning(f"⚠ Ошибка загрузки улучшенного состояния: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            model = model.to(device)
            return model
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка загрузки Thalia: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            try:
                logger.warning(f"⚠ Пробую загрузить как обычную GPT-2 модель...")
                from transformers import GPT2LMHeadModel
                return GPT2LMHeadModel.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    **kwargs
                )
            except Exception as e2:
                logger.error(f"❌ И загрузка как GPT-2 тоже не удалась: {e2}")
                raise e
