# -*- coding: utf-8 -*-
# bidirectional_exchange.py - ИСПРАВЛЕННАЯ ВЕРСИЯ v3.9
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# ===================================================================
# КЛАСС ДВУСТОРОННЕГО ОБМЕНА (ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ===================================================================
class BidirectionalExperienceExchange(nn.Module):
    """
    🔄 ИСПРАВЛЕННЫЙ v3.9: ДВУСТОРОННИЙ обмен опытом между Transformer и Mamba
    (Только pair-буферы, обратная совместимость со старыми чекпоинтами)
    """   
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ИСПРАВЛЕНИЕ: сохраняем размерности как атрибуты
        self.n_embd = config.n_embd
        self.slot_size = getattr(config, 'slot_size', config.n_embd // 2)
        self.shared_exchange_dim = getattr(config, 'shared_exchange_dim', config.n_embd)
        
        # ФЛАГИ
        self.bidirectional_enabled = getattr(config, 'bidirectional_exchange', True)
        self.exchange_strength = getattr(config, 'exchange_strength', 0.35)
        self.feedback_loop_active = False
        self.max_experience_bank_size = getattr(config, 'max_experience_bank_size', 100)
        
        if not self.bidirectional_enabled:
            logger.info("🔄 Двусторонний обмен отключен")
            return
        
        logger.info(f"🔄 Shared exchange dim: {self.shared_exchange_dim} "
                   f"(n_embd={config.n_embd}, slot_size={self.slot_size})")
        
        # Transformer → Shared space
        self.transformer_to_shared = nn.Sequential(
            nn.Linear(config.n_embd, self.shared_exchange_dim),
            nn.LayerNorm(self.shared_exchange_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.shared_exchange_dim, self.shared_exchange_dim),
            nn.LayerNorm(self.shared_exchange_dim)
        )
        
        # Mamba → Shared space
        self.mamba_to_shared = nn.Sequential(
            nn.Linear(self.slot_size, self.shared_exchange_dim),
            nn.LayerNorm(self.shared_exchange_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.shared_exchange_dim, self.shared_exchange_dim),
            nn.LayerNorm(self.shared_exchange_dim)
        )
        
        # Shared space → Transformer
        self.shared_to_transformer = nn.Sequential(
            nn.Linear(self.shared_exchange_dim, self.shared_exchange_dim),
            nn.GELU(),
            nn.Linear(self.shared_exchange_dim, config.n_embd)
        )
        
        # Shared space → Mamba
        self.shared_to_mamba = nn.Sequential(
            nn.Linear(self.shared_exchange_dim, self.shared_exchange_dim),
            nn.GELU(),
            nn.Linear(self.shared_exchange_dim, self.slot_size)
        )
        
        # ГЕЙТЫ (ШЛЮЗЫ)
        self.transformer_gate = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.Sigmoid()
        )
        
        self.mamba_gate = nn.Sequential(
            nn.Linear(self.slot_size * 2, self.slot_size),
            nn.GELU(),
            nn.Linear(self.slot_size, self.slot_size),
            nn.Sigmoid()
        )
        
        # КРОСС-МОДАЛЬНОЕ ВНИМАНИЕ
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.shared_exchange_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
            kdim=self.shared_exchange_dim,
            vdim=self.shared_exchange_dim
        )
        
        # Инициализация attention
        def _init_attention_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        self.cross_modal_attention.apply(_init_attention_weights)
        
        # Температурный параметр для Attention
        self.attention_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.attention_scale = math.sqrt(self.shared_exchange_dim)
        
        # ПРОЕКЦИЯ ATTENTION ENTROPY
        self.attention_entropy_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, config.n_embd)
        )
        
        # ===========================================================
        # 🔥 ТОЛЬКО БУФЕРЫ ДЛЯ СИНХРОНИЗИРОВАННЫХ ПАР
        # ===========================================================
        self._init_buffers()
        
        # Метаданные
        self.transformer_metadata = []
        self.mamba_metadata = []
        
        # ОБУЧАЕМЫЕ ЯКОРЯ ДЛЯ ALIGNMENT (если буферы пусты)
        self.alignment_anchors = nn.Parameter(
            torch.randn(16, self.shared_exchange_dim) * 0.02
        )
        
        self.exchange_counter = 0
        
        # МЕТРИКИ СИМБИОЗА
        self.register_buffer("exchange_efficiency", torch.tensor(0.5))
        self.register_buffer("alignment_score", torch.tensor(0.3))
        self.register_buffer("mutual_reward", torch.tensor(0.0))
        
        # 🔥 ДЛЯ ОТСЛЕЖИВАНИЯ ALIGNMENT LOSS
        self._last_align_loss = 0.0
        
        # Счетчики для мониторинга заполнения буфера
        self._logged_50_percent = False
        self._logged_75_percent = False
        self._logged_90_percent = False
        
        logger.info(f"🔄 ДВУСТОРОННИЙ ОБМЕН v3.9: Только pair-буферы, max_size={self.max_experience_bank_size}")

    def _init_buffers(self):
        """Инициализация только pair-буферов"""
        # 🔥 ТОЛЬКО БУФЕРЫ ДЛЯ СИНХРОНИЗИРОВАННЫХ ПАР
        self.register_buffer("pair_t_buffer", 
                           torch.zeros(self.max_experience_bank_size, self.n_embd))
        self.register_buffer("pair_m_buffer", 
                           torch.zeros(self.max_experience_bank_size, self.slot_size))
        self.register_buffer("pair_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("pair_count", torch.tensor(0, dtype=torch.long))
        
        # Счетчик шагов (для log_gradient_stats)
        self.step_count = 0
        
        # Сбрасываем несуществующий кэш (пустышка)
        self.invalidate_cache()

    def capture_pair(self, t_raw: torch.Tensor, m_raw: torch.Tensor):
        """
        🔥 Записывает синхронизированную пару (Transformer, Mamba) для alignment.
        
        Args:
            t_raw: [batch, n_embd] или [batch, seq, n_embd] - сырой вектор от Transformer
            m_raw: [batch, slot_size] или [batch, seq, slot_size] - сырой вектор от Mamba
        """
        if not self.bidirectional_enabled:
            return
        
        with torch.no_grad():
            # 🔧 ИСПРАВЛЕНИЕ 1: Проверяем размерность t_raw
            if t_raw.dim() == 3:
                # Если пришел 3D [batch, seq, dim] - берем среднее
                t_raw = t_raw.mean(dim=1)
            elif t_raw.dim() == 1:
                # Если пришел 1D [dim] - добавляем batch dimension
                t_raw = t_raw.unsqueeze(0)
            
            # 🔧 ИСПРАВЛЕНИЕ 2: Проверяем размерность m_raw
            if m_raw.dim() == 3:
                # Если пришел 3D [batch, seq, dim] - берем последний токен
                m_raw = m_raw[:, -1, :]
            elif m_raw.dim() == 1:
                # Если пришел 1D [dim] - добавляем batch dimension
                m_raw = m_raw.unsqueeze(0)
            
            # 🔧 ИСПРАВЛЕНИЕ 3: Убеждаемся, что у нас [batch, dim]
            assert t_raw.dim() == 2, f"t_raw должен быть 2D, но {t_raw.dim()}D"
            assert m_raw.dim() == 2, f"m_raw должен быть 2D, но {m_raw.dim()}D"
            
            batch_size = t_raw.shape[0]
            
            # 🔧 ИСПРАВЛЕНИЕ 4: Проверяем, что batch_size совпадает
            if m_raw.shape[0] != batch_size:
                logger.debug(f"⚠️ Размер батча не совпадает: t={batch_size}, m={m_raw.shape[0]}, выравниваю...")
                min_batch = min(batch_size, m_raw.shape[0])
                t_raw = t_raw[:min_batch]
                m_raw = m_raw[:min_batch]
                batch_size = min_batch
            
            # 🔥 ИСПРАВЛЕНИЕ: Явная синхронизация устройств
            buffer_device = self.pair_t_buffer.device
            t_raw = t_raw.to(buffer_device)
            m_raw = m_raw.to(buffer_device)
            
            # Защита от NaN
            t_raw = torch.nan_to_num(t_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            m_raw = torch.nan_to_num(m_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # ===========================================================
            # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Защита от коллизий индексов
            # ===========================================================
            if batch_size >= self.max_experience_bank_size:
                logger.warning(
                    f"⚠️ batch_size ({batch_size}) >= max_experience_bank_size ({self.max_experience_bank_size}). "
                    f"Буфер будет перезаписан полностью, беру последние {self.max_experience_bank_size} элементов."
                )
                # Берем последние max_size элементов из батча
                t_raw = t_raw[-self.max_experience_bank_size:]
                m_raw = m_raw[-self.max_experience_bank_size:]
                batch_size = self.max_experience_bank_size
                
                # Прямая запись с нуля (перезаписываем весь буфер)
                self.pair_t_buffer.data.copy_(t_raw)
                self.pair_m_buffer.data.copy_(m_raw)
                self.pair_count.fill_(batch_size)
                self.pair_ptr.fill_(batch_size % self.max_experience_bank_size)
                
                # Логируем заполнение буфера
                fill_ratio = batch_size / self.max_experience_bank_size
                if fill_ratio >= 0.9 and not self._logged_90_percent:
                    logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({batch_size}/{self.max_experience_bank_size})")
                    self._logged_90_percent = True
                elif fill_ratio >= 0.75 and not self._logged_75_percent:
                    logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({batch_size}/{self.max_experience_bank_size})")
                    self._logged_75_percent = True
                elif fill_ratio >= 0.5 and not self._logged_50_percent:
                    logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({batch_size}/{self.max_experience_bank_size})")
                    self._logged_50_percent = True
                
                return
            
            # ===========================================================
            # 🔥 Векторизованная вставка (без цикла с item/fill)
            # ===========================================================
            
            # Получаем текущий указатель (ОДИН раз за вызов)
            current_ptr = self.pair_ptr.item()
            
            # Вычисляем индексы для всего батча сразу
            indices = torch.arange(batch_size, device=self.pair_ptr.device)
            indices = (current_ptr + indices) % self.max_experience_bank_size
            
            # Векторизованная вставка (одна операция на весь батч)
            self.pair_t_buffer[indices] = t_raw
            self.pair_m_buffer[indices] = m_raw
            
            # Обновляем указатель (ОДИН раз за вызов)
            new_ptr = (current_ptr + batch_size) % self.max_experience_bank_size
            self.pair_ptr.fill_(new_ptr)
            
            # Обновляем счетчик (ОДИН раз за вызов)
            new_count = min(self.max_experience_bank_size, self.pair_count.item() + batch_size)
            self.pair_count.fill_(new_count)
            
            # Логируем при заполнении буфера
            fill_ratio = new_count / self.max_experience_bank_size
            if fill_ratio >= 0.9 and not self._logged_90_percent:
                logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({new_count}/{self.max_experience_bank_size})")
                self._logged_90_percent = True
            elif fill_ratio >= 0.75 and not self._logged_75_percent:
                logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({new_count}/{self.max_experience_bank_size})")
                self._logged_75_percent = True
            elif fill_ratio >= 0.5 and not self._logged_50_percent:
                logger.info(f"📊 Pair буфер заполнен на {fill_ratio:.1%} ({new_count}/{self.max_experience_bank_size})")
                self._logged_50_percent = True

    # ===========================================================
    # 🔥 ALIGNMENT LOSS НА ОСНОВЕ СИНХРОНИЗИРОВАННЫХ ПАР
    # ===========================================================
    
    def compute_direct_alignment_loss(self):
        """
        🔥 ALIGNMENT LOSS - MSE между представлениями одной мысли
        """
        count = self.pair_count.item()
        if count < 32:
            return torch.tensor(0.0, device=self.pair_t_buffer.device, requires_grad=True)
        
        # 🔥 ИСПРАВЛЕНИЕ: используем устройство из буфера, а не из alignment_score
        device = self.pair_t_buffer.device
        n = min(64, count)
        ptr = self.pair_ptr.item()
        
        # Берем последние n пар из кольцевого буфера
        if count >= self.max_experience_bank_size:
            indices = [(ptr - i - 1) % self.max_experience_bank_size for i in range(n)]
        else:
            indices = list(range(count - n, count))
        
        t_raw = self.pair_t_buffer[indices].to(device)
        m_raw = self.pair_m_buffer[indices].to(device)
        
        # Проецируем в общее пространство
        t_shared = self.transformer_to_shared(t_raw)
        m_shared = self.mamba_to_shared(m_raw)
        
        # Нормализация для стабильности (опционально)
        t_norm = F.normalize(t_shared, dim=-1, eps=1e-8)
        m_norm = F.normalize(m_shared, dim=-1, eps=1e-8)
        
        # 🔥 MSE LOSS между соответствующими парами
        alignment_loss = F.mse_loss(t_norm, m_norm)
        
        # Добавляем небольшую регуляризацию для разнообразия
        diversity_loss = -torch.std(t_shared) * 0.001 - torch.std(m_shared) * 0.001
        loss = alignment_loss + diversity_loss
        
        # 🔥 Проверка на валидность лосса
        if not torch.isfinite(loss):
            logger.warning(f"⚠ Alignment loss невалиден: {loss.item()}, возвращаю 0")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Обновляем alignment score (чем меньше loss, тем лучше)
        with torch.no_grad():
            # Преобразуем loss в score от 0 до 1 (меньше loss = больше score)
            current_score = 1.0 / (1.0 + alignment_loss.item() * 10)
            self.alignment_score.mul_(0.95).add_(current_score * 0.05)
        
        if self.exchange_counter % 200 == 0:
            logger.info(f"🎯 ALIGNMENT LOSS: {loss.item():.4f} | "
                       f"MSE={alignment_loss.item():.4f} | "
                       f"score={self.alignment_score.item():.3f} | "
                       f"m_std={torch.std(m_shared).item():.3f}")
        
        return loss

    # ===========================================================
    # 🔥 ИСПРАВЛЕННЫЙ МЕТОД ОБМЕНА ОПЫТОМ
    # ===========================================================
    
    def exchange_experiences(self, 
                             transformer_hidden: torch.Tensor,
                             mamba_hidden: torch.Tensor,
                             mode: str = 'mutual_enhancement'):
        """
        🔄 ОБМЕН ОПЫТОМ (исправлен - прямой доступ к pair-буферу)
        """
        if not self.bidirectional_enabled:
            return transformer_hidden, mamba_hidden
        
        # 🔥 РЕГУЛЯРНАЯ ОЧИСТКА БУФЕРА ОТ NaN
        if self.exchange_counter % 50 == 0:
            with torch.no_grad():
                if torch.isnan(self.pair_t_buffer).any():
                    self.pair_t_buffer.data = torch.nan_to_num(self.pair_t_buffer, nan=0.0)
                if torch.isnan(self.pair_m_buffer).any():
                    self.pair_m_buffer.data = torch.nan_to_num(self.pair_m_buffer, nan=0.0)
        
        # Коррекция размерностей Mamba
        if mamba_hidden is not None:
            # 🔥 ИСПРАВЛЕНИЕ: проверка совместимости форм перед expand
            if mamba_hidden.shape[0] != transformer_hidden.shape[0]:
                batch_size = transformer_hidden.shape[0]
                try:
                    mamba_hidden = mamba_hidden.expand(batch_size, -1, -1)
                except RuntimeError as e:
                    logger.warning(f"⚠ Не удалось expand mamba_hidden: {e}, использую повторение")
                    # Fallback: повторяем первый элемент
                    if mamba_hidden.shape[0] == 1:
                        mamba_hidden = mamba_hidden.repeat(batch_size, 1, 1)
                    else:
                        # Обрезаем или дополняем
                        if mamba_hidden.shape[0] > batch_size:
                            mamba_hidden = mamba_hidden[:batch_size]
                        else:
                            # Дополняем нулями
                            pad = torch.zeros(batch_size - mamba_hidden.shape[0], 
                                            mamba_hidden.shape[1], 
                                            mamba_hidden.shape[2],
                                            device=mamba_hidden.device)
                            mamba_hidden = torch.cat([mamba_hidden, pad], dim=0)
            
            if mamba_hidden.shape[1] == 1 and transformer_hidden.shape[1] > 1:
                seq_len = transformer_hidden.shape[1]
                try:
                    mamba_hidden = mamba_hidden.expand(-1, seq_len, -1)
                except RuntimeError as e:
                    logger.warning(f"⚠ Не удалось expand mamba_hidden по seq: {e}")
                    # Fallback: повторяем
                    mamba_hidden = mamba_hidden.repeat(1, seq_len, 1)
        
        self.exchange_counter += 1
        
        # Определяем устройство из входных данных
        device = transformer_hidden.device
        
        try:
            # 🔥 Берем сырые векторы прямо из актуального буфера пар
            count = self.pair_count.item()
            if count == 0:
                return transformer_hidden, mamba_hidden
            
            t_raw = self.pair_t_buffer[:count].to(device)
            m_raw = self.pair_m_buffer[:count].to(device)
            
            # 🔥 Проецируем их в shared пространство на лету
            trans_shared = self.transformer_to_shared(t_raw)
            mamba_shared = self.mamba_to_shared(m_raw)
            
            # Адаптивная сила обмена
            current_alignment = self.alignment_score.item()
            adaptive_strength = self.exchange_strength
            
            if current_alignment < 0.3:
                adaptive_strength *= 1.2
            elif current_alignment > 0.5:
                adaptive_strength *= 0.8
            
            adaptive_strength = max(0.1, min(0.5, adaptive_strength))
            
            batch_size = transformer_hidden.shape[0]
            
            if mode == 'mutual_enhancement':
                # Простая интеграция через центроиды
                trans_center = trans_shared.mean(dim=0, keepdim=True)  # [1, shared_dim]
                mamba_center = mamba_shared.mean(dim=0, keepdim=True)  # [1, shared_dim]
                
                # Нормализуем
                trans_center = F.normalize(trans_center, dim=-1, eps=1e-8)
                mamba_center = F.normalize(mamba_center, dim=-1, eps=1e-8)
                
                # Смешиваем
                transformer_enrichment = 0.7 * trans_center + 0.3 * mamba_center
                mamba_enrichment = 0.3 * trans_center + 0.7 * mamba_center
                
                # Восстанавливаем норму
                trans_norm = trans_shared.norm(dim=-1).mean().item()
                mamba_norm = mamba_shared.norm(dim=-1).mean().item()
                target_norm = (trans_norm + mamba_norm) / 2.0
                
                transformer_enrichment = transformer_enrichment * target_norm
                mamba_enrichment = mamba_enrichment * target_norm
                
                # Расширяем для батча
                transformer_enrichment = transformer_enrichment.expand(batch_size, -1)
                mamba_enrichment = mamba_enrichment.expand(batch_size, -1)
                
                # Проецируем обратно (с градиентами)
                trans_to_orig = self.shared_to_transformer(transformer_enrichment)
                mamba_to_orig = self.shared_to_mamba(mamba_enrichment)
                
                # Применяем к Transformer
                memory_advice = trans_to_orig.unsqueeze(1).expand(-1, transformer_hidden.shape[1], -1)
                gate_input = torch.cat([transformer_hidden, memory_advice], dim=-1)
                gate_values = self.transformer_gate(gate_input)
                gate_values = torch.clamp(gate_values, 0.05, 0.95)
                
                transformer_hidden = transformer_hidden + memory_advice * gate_values * adaptive_strength
                
                # Применяем к Mamba
                if mamba_hidden is not None and mamba_to_orig is not None:
                    mamba_memory = mamba_to_orig.unsqueeze(1).expand(-1, mamba_hidden.shape[1], -1)
                    mamba_gate_input = torch.cat([mamba_hidden, mamba_memory], dim=-1)
                    mamba_gate_values = self.mamba_gate(mamba_gate_input)
                    mamba_gate_values = torch.clamp(mamba_gate_values, 0.05, 0.95)
                    mamba_hidden = mamba_hidden + mamba_memory * mamba_gate_values * adaptive_strength
            
            # Обновляем метрики
            buffer_usage = count / self.max_experience_bank_size
            self.exchange_efficiency = 0.95 * self.exchange_efficiency + 0.05 * buffer_usage
            
            if self.exchange_counter % 10 == 0:
                try:
                    self.log_gradient_stats(self.exchange_counter)
                except Exception as e:
                    logger.debug(f"⚠ Ошибка при логировании градиентов: {e}")
            
            return transformer_hidden, mamba_hidden
            
        except Exception as e:
            logger.debug(f"⚠ Ошибка обмена опытом: {e}")
            return transformer_hidden, mamba_hidden
    
    def invalidate_cache(self):
        """Пустышка для совместимости"""
        pass
    
    def log_gradient_stats(self, step_count):
        """🔥 Логирует статистику градиентов"""
        self.step_count = step_count
        
        if step_count % 10 != 0:
            return
        
        total_grad_norm = 0.0
        param_count = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                # 🔥 Проверка на валидность градиента
                if not math.isfinite(grad_norm):
                    logger.warning(f"⚠ Невалидный градиент для {name}: {grad_norm}")
                    continue
                total_grad_norm += grad_norm
                param_count += 1
                
                if grad_norm > 0.1:
                    logger.debug(f"📈 {name}: grad_norm = {grad_norm:.6f}")
        
        if param_count > 0:
            avg_grad = total_grad_norm / param_count
            logger.info(f"📊 Gradient stats at step {step_count}:")
            logger.info(f"   Average norm: {avg_grad:.6f}, Total: {total_grad_norm:.6f}")
    
    def resize_buffers(self, new_n_embd=None, new_slot_size=None, new_shared_dim=None):
        """Вызывается при загрузке модели с другими размерностями"""
        if new_n_embd:
            self.n_embd = new_n_embd
        if new_slot_size:
            self.slot_size = new_slot_size
        if new_shared_dim:
            self.shared_exchange_dim = new_shared_dim
        
        self._init_buffers()
        logger.info(f"🔄 Буферы изменены: n_embd={self.n_embd}, "
                   f"slot_size={self.slot_size}, shared_dim={self.shared_exchange_dim}")
    
    def get_exchange_report(self) -> Dict:
        """Отчет о состоянии обмена"""
        if not self.bidirectional_enabled:
            return {"status": "disabled"}
        
        quality_stats = self.get_alignment_quality_stats()
        
        return {
            "status": "active",
            "exchange_counter": self.exchange_counter,
            "efficiency": float(self.exchange_efficiency.item()),
            "alignment": float(self.alignment_score.item()),
            "mutual_reward": float(self.mutual_reward.item()),
            "exchange_strength": float(self.exchange_strength),
            "quality_stats": quality_stats,
            "pair_buffer": {
                "count": self.pair_count.item(),
                "max_size": self.max_experience_bank_size,
                "pointer": self.pair_ptr.item(),
                "usage_percent": self.pair_count.item() / self.max_experience_bank_size * 100
            }
        }
    
    def detach_buffers(self):
        """Для совместимости с существующим кодом"""
        pass
    
    def get_alignment_quality_stats(self) -> Dict:
        """Статистика качества alignment"""
        if self.pair_count.item() < 5:
            return {'quality': 0.0, 'samples': 0}
        
        # 🔥 ИСПРАВЛЕНИЕ: используем устройство из буфера
        device = self.pair_t_buffer.device
        
        try:
            count = self.pair_count.item()
            n = min(20, count)
            ptr = self.pair_ptr.item()
            
            if count >= self.max_experience_bank_size:
                indices = [(ptr - i - 1) % self.max_experience_bank_size for i in range(n)]
            else:
                indices = list(range(count - n, count))
            
            t_raw = self.pair_t_buffer[indices].to(device)
            m_raw = self.pair_m_buffer[indices].to(device)
            
            with torch.no_grad():
                t_shared = self.transformer_to_shared(t_raw)
                m_shared = self.mamba_to_shared(m_raw)
                
                t_norm = F.normalize(t_shared, dim=-1)
                m_norm = F.normalize(m_shared, dim=-1)
                
                pos_sim = (t_norm * m_norm).sum(dim=-1).mean().item()
            
            return {
                'quality': pos_sim,
                'samples': n,
                'pair_count': count,
                'alignment_score': self.alignment_score.item()
            }
            
        except Exception as e:
            logger.debug(f"⚠ Ошибка получения статистики: {e}")
            return {'quality': 0.0, 'samples': 0}
    
    def test_buffer_integrity(self):
        """
        🧪 Тест целостности с проверкой pair-буферов
        """
        logger.info("🧪 Проверка целостности буферов обмена опытом...")
        
        try:
            # Проверка pair-буферов
            pair_real = (self.pair_t_buffer.abs().sum(dim=-1) > 1e-6).sum().item()
            pair_count = self.pair_count.item()
            
            # 🔥 Дополнительная проверка: count не должен превышать max
            if pair_count > self.max_experience_bank_size:
                logger.error(f"❌ pair_count ({pair_count}) > max_experience_bank_size ({self.max_experience_bank_size})! Исправляю...")
                self.pair_count.fill_(self.max_experience_bank_size)
                pair_count = self.max_experience_bank_size
            
            if pair_real != pair_count:
                logger.warning(f"⚠️ Расхождение в pair буфере: счетчик={pair_count}, реально={pair_real}")
                self.pair_count.fill_(pair_real)
                if pair_real == 0:
                    self.pair_ptr.fill_(0)
            
            # Проверка на NaN/Inf
            for buffer_name in ['pair_t_buffer', 'pair_m_buffer']:
                buffer = getattr(self, buffer_name)
                if torch.isnan(buffer).any() or torch.isinf(buffer).any():
                    logger.warning(f"⚠️ В {buffer_name} обнаружены NaN/Inf! Очищаем...")
                    buffer.data = torch.nan_to_num(buffer, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"📊 Pair buffer: {pair_real}/{self.max_experience_bank_size}, ptr={self.pair_ptr.item()}")
            
            if pair_real > 0:
                quality_stats = self.get_alignment_quality_stats()
                logger.info(f"   Alignment quality: {quality_stats['quality']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке целостности: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_alignment_gradients(self):
        """
        🔧 Тестовая функция для проверки потока градиентов
        """
        print("\n🧪 ТЕСТ ГРАДИЕНТОВ ALIGNMENT LOSS:")
        print("=" * 60)
        
        loss = self.compute_direct_alignment_loss()
        
        # 🔥 Проверка на валидность
        if not torch.isfinite(loss):
            print("❌ Loss невалиден (NaN/Inf)")
            return False
        
        if loss.item() > 0:
            print(f"✅ Alignment loss требует градиентов: {loss.requires_grad}")
            print(f"✅ Значение loss: {loss.item():.6f}")
            
            t_proj = self.transformer_to_shared[0].weight
            m_proj = self.mamba_to_shared[0].weight
            
            if t_proj.grad is not None:
                grad_norm = t_proj.grad.norm().item()
                if math.isfinite(grad_norm):
                    print(f"✅ Transformer projector уже имеет градиенты (norm={grad_norm:.6f})")
                else:
                    print(f"⚠️ Transformer projector имеет невалидные градиенты")
            else:
                print("ℹ️ Transformer projector пока без градиентов (нужен backward)")
            
            if m_proj.grad is not None:
                grad_norm = m_proj.grad.norm().item()
                if math.isfinite(grad_norm):
                    print(f"✅ Mamba projector уже имеет градиенты (norm={grad_norm:.6f})")
                else:
                    print(f"⚠️ Mamba projector имеет невалидные градиенты")
            else:
                print("ℹ️ Mamba projector пока без градиентов (нужен backward)")
            
        else:
            print("⚠️ Alignment loss = 0, буферы пусты")
        
        print("=" * 60)
        return loss.item() > 0 and torch.isfinite(loss)
    
    # ===========================================================
    # 🔥 ОБРАТНАЯ СОВМЕСТИМОСТЬ ДЛЯ СТАРЫХ ЧЕКПОИНТОВ
    # ===========================================================
    
    def get_state(self):
        """
        Возвращает состояние буферов для сохранения.
        Сохраняем в новом формате, но добавляем пустые заглушки для старых ключей,
        чтобы не ломать код, который ожидает их наличие.
        """
        device = self.pair_t_buffer.device
        
        # Базовое состояние с новыми буферами
        state = {
            # Новые ключи
            'pair_ptr': self.pair_ptr.cpu().clone(),
            'pair_count': self.pair_count.cpu().clone(),
            'pair_t_buffer': self.pair_t_buffer.cpu().clone(),
            'pair_m_buffer': self.pair_m_buffer.cpu().clone(),
            'exchange_counter': self.exchange_counter,
            'alignment_score': self.alignment_score.cpu().clone(),
            'mutual_reward': self.mutual_reward.cpu().clone(),
            
            # Заглушки для обратной совместимости (пустые буферы)
            'trans_ptr': torch.tensor(0, dtype=torch.long),
            'mamba_ptr': torch.tensor(0, dtype=torch.long),
            'trans_count': torch.tensor(0, dtype=torch.long),
            'mamba_count': torch.tensor(0, dtype=torch.long),
            'transformer_raw_buffer': torch.zeros(0, self.n_embd),
            'mamba_raw_buffer': torch.zeros(0, self.slot_size),
            'transformer_shared_buffer': torch.zeros(0, self.shared_exchange_dim),
            'mamba_shared_buffer': torch.zeros(0, self.shared_exchange_dim),
        }
        
        return state

    def set_state(self, state, device):
        """
        Восстанавливает состояние буферов из словаря.
        Поддерживает как новый формат (только pair-буферы), так и старый.
        """
        # Проверяем, есть ли новые ключи
        has_new_format = 'pair_ptr' in state and 'pair_count' in state and \
                         'pair_t_buffer' in state and 'pair_m_buffer' in state
        
        if has_new_format:
            # Загружаем в новом формате
            self.pair_ptr.data.copy_(state['pair_ptr'].to(device))
            self.pair_count.data.copy_(state['pair_count'].to(device))
            self.pair_t_buffer.data.copy_(state['pair_t_buffer'].to(device))
            self.pair_m_buffer.data.copy_(state['pair_m_buffer'].to(device))
        else:
            # Старый формат - пытаемся восстановить данные из старых буферов
            logger.info("🔄 Обнаружен старый формат чекпоинта, конвертирую в новый...")
            
            # Пытаемся извлечь данные из старых буферов
            old_t_raw = state.get('transformer_raw_buffer')
            old_m_raw = state.get('mamba_raw_buffer')
            old_t_count = state.get('trans_count', torch.tensor(0)).item()
            old_m_count = state.get('mamba_count', torch.tensor(0)).item()
            
            if old_t_raw is not None and old_m_raw is not None and old_t_count > 0 and old_m_count > 0:
                # Берем минимальное количество пар
                n_pairs = min(old_t_count, old_m_count, self.max_experience_bank_size)
                
                if n_pairs > 0:
                    # Копируем данные из старых буферов
                    t_data = old_t_raw[:n_pairs].to(device)
                    m_data = old_m_raw[:n_pairs].to(device)
                    
                    # Проверяем размерности
                    if t_data.shape[-1] != self.n_embd:
                        logger.warning(f"⚠️ Размерность t_raw не совпадает: {t_data.shape[-1]} vs {self.n_embd}")
                        t_data = F.adaptive_avg_pool1d(t_data.unsqueeze(0), self.n_embd).squeeze(0)
                    
                    if m_data.shape[-1] != self.slot_size:
                        logger.warning(f"⚠️ Размерность m_raw не совпадает: {m_data.shape[-1]} vs {self.slot_size}")
                        m_data = F.adaptive_avg_pool1d(m_data.unsqueeze(0), self.slot_size).squeeze(0)
                    
                    # Заполняем pair-буферы
                    self.pair_t_buffer[:n_pairs] = t_data
                    self.pair_m_buffer[:n_pairs] = m_data
                    self.pair_count.fill_(n_pairs)
                    self.pair_ptr.fill_(n_pairs % self.max_experience_bank_size)
                    
                    logger.info(f"✅ Конвертировано {n_pairs} пар из старого формата")
            
            # Загружаем остальные метрики
            if 'alignment_score' in state:
                self.alignment_score.data.copy_(state['alignment_score'].to(device))
            if 'mutual_reward' in state:
                self.mutual_reward.data.copy_(state['mutual_reward'].to(device))
        
        # Загружаем общие метрики (если есть)
        if 'exchange_counter' in state:
            self.exchange_counter = state['exchange_counter']
        
        # 🔥 Дополнительная проверка: count не должен превышать max
        if self.pair_count.item() > self.max_experience_bank_size:
            logger.warning(f"⚠️ pair_count ({self.pair_count.item()}) > max, исправляю...")
            self.pair_count.fill_(self.max_experience_bank_size)
        
        self.invalidate_cache()
        
        # Сбрасываем флаги логирования заполнения
        self._logged_50_percent = False
        self._logged_75_percent = False
        self._logged_90_percent = False
        
        logger.info(f"✅ Состояние обмена опытом восстановлено, пар в буфере: {self.pair_count.item()}")
