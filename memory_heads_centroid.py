# -*- coding: utf-8 -*-
# memory_heads_centroid.py
"""
memory_heads_centroid.py — ЦЕНТРОИДНАЯ ПАМЯТЬ v1.4.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import random
import numpy as np
import copy
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

MAX_MEMBERSHIP_PER_SLOT = 100
UTILITY_BOOST_STRONG = 0.018


# ===================================================================
# СТРУКТУРА СЛОТА 
# ===================================================================
@dataclass
class ConceptSlot:
    centroid: torch.Tensor
    membership_ids: List[int] = field(default_factory=list)
    utility: float = 0.0
    hit_count: int = 0
    write_count: int = 0
    variance: float = 0.0
    age: int = 0
    frozen: bool = False
    last_activation: int = 0

    learning_rate: float = 0.05
    momentum: torch.Tensor = None

    def to_dict(self) -> Dict:
        return {
            'centroid': self.centroid.tolist(),
            'membership_ids': self.membership_ids,
            'utility': self.utility,
            'hit_count': self.hit_count,
            'write_count': self.write_count,
            'variance': self.variance,
            'age': self.age,
            'frozen': self.frozen,
            'last_activation': self.last_activation,
            'learning_rate': self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict, device: torch.device) -> 'ConceptSlot':
        slot = cls(
            centroid=torch.tensor(data['centroid'], dtype=torch.float32, device=device),
            membership_ids=data.get('membership_ids', []),
            utility=data.get('utility', 0.0),
            hit_count=data.get('hit_count', 0),
            write_count=data.get('write_count', 0),
            variance=data.get('variance', 0.0),
            age=data.get('age', 0),
            frozen=data.get('frozen', False),
            last_activation=data.get('last_activation', 0),
            learning_rate=data.get('learning_rate', 0.05),
        )
        return slot

# ===================================================================
# NEURAL LINKER
# ===================================================================

class NeuralLinker(nn.Module):
    """
    🧠 УЛУЧШЕННЫЙ Linker со стабильными embeddings и LayerNorm
    """
    def __init__(self, slot_dim: int, num_slots: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # ✅ Добавляем размер для embeddings (slot_dim//4)
        embed_dim = slot_dim // 4
        input_dim = slot_dim + embed_dim
        hidden = hidden_dim or (input_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),  # ✅ LayerNorm для стабильности
            nn.ReLU(),
            nn.Linear(hidden, num_slots),
        )
        
        # ✅ Опциональный GRU для последовательностей (пока не используется)
        # self.temporal_encoder = nn.GRU(
        #     input_size=input_dim,
        #     hidden_size=hidden,
        #     num_layers=1,
        #     batch_first=True
        # )
        
        self.apply(self._init_weights)
        
        # Регистрация устройства
        self.register_buffer('_dummy', torch.tensor(0), persistent=False)
        
        logger.info(f"🧠 Linker архитектура: {slot_dim} + {embed_dim} → {hidden} → {num_slots}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, slot_embed=None, sequence=False):
        """
        Args:
            x: центроид [batch, dim] или [batch, seq, dim]
            slot_embed: embedding слота [batch, embed_dim] или None
            sequence: использовать GRU для последовательностей (пока не реализовано)
        """
        device = self._dummy.device
        x = x.to(device)
        
        if slot_embed is not None:
            slot_embed = slot_embed.to(device)
            if x.dim() == 3:
                # [batch, seq, dim] + [batch, embed] -> расширяем embed
                slot_embed = slot_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, slot_embed], dim=-1)
        
        # if sequence and x.dim() == 3:
        #     x, _ = self.temporal_encoder(x)
        #     x = x[:, -1, :]  # берем последний
        
        return self.net(x)

# ===================================================================
# ЦЕНТРОИДНЫЙ МЕНЕДЖЕР — v1.4.3 
# ===================================================================
class CentroidMemoryManager(nn.Module):
    def __init__(
        self,
        num_slots: int = 512,
        slot_dim: int = 384,
        core_slots: int = 8,
        similarity_threshold: float = 0.85,
        merge_threshold: float = 0.96,
        split_variance_threshold: float = 0.3,
        eviction_utility_threshold: float = 0.01,
        buffer_size: int = 2000,  # ← ДОБАВЛЕННЫЙ ПАРАМЕТР
        device: torch.device = None,
        enable_linker: bool = True,
        linker_lr: float = 0.001,
        linker_weight_decay: float = 0.01,
        linker_hidden_dim: Optional[int] = None,
        linker_batch_size: int = 32,
        linker_train_frequency: int = 5,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.core_slots = core_slots
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.split_variance_threshold = split_variance_threshold
        self.eviction_utility_threshold = eviction_utility_threshold
        self.buffer_size = buffer_size  # ← СОХРАНЯЕМ ЗНАЧЕНИЕ
        self.device = device or torch.device('cpu')
        self._is_meta = False
        
        # Параметры Linker
        self.enable_linker = enable_linker
        self.linker_lr = linker_lr
        self.linker_weight_decay = linker_weight_decay
        self.linker_batch_size = linker_batch_size
        self.linker_train_frequency = linker_train_frequency
        # ✅ ИСПРАВЛЕНИЕ 3: сохраняем linker_hidden_dim
        self.linker_hidden_dim = linker_hidden_dim or (slot_dim // 2)

        self.register_buffer("centroids", torch.zeros(num_slots, slot_dim, device=self.device))
        nn.init.normal_(self.centroids, mean=0.0, std=0.02)

        self.centroids.requires_grad_(False)

        # ✅ Стабильные embeddings для Linker
        embed_dim = slot_dim // 4
        self.slot_embeddings = nn.Embedding(num_slots, embed_dim)
        nn.init.normal_(self.slot_embeddings.weight, mean=0.0, std=0.1)

        self.transition_matrix = nn.Parameter(torch.zeros(num_slots, num_slots, device=self.device))

        # Метаданные - сразу на устройстве
        self.register_buffer('slot_utility', torch.zeros(num_slots, device=self.device))
        self.register_buffer('slot_hit_count', torch.zeros(num_slots, dtype=torch.long, device=self.device))
        self.register_buffer('slot_write_count', torch.zeros(num_slots, dtype=torch.long, device=self.device))
        self.register_buffer('slot_variance', torch.zeros(num_slots, device=self.device))
        self.register_buffer('slot_age', torch.zeros(num_slots, dtype=torch.long, device=self.device))
        self.register_buffer('slot_frozen', torch.zeros(num_slots, dtype=torch.bool, device=self.device))
        self.register_buffer('slot_last_activation', torch.zeros(num_slots, dtype=torch.long, device=self.device))
        self.register_buffer('slot_active', torch.zeros(num_slots, dtype=torch.bool, device=self.device))

        self.membership: List[List[int]] = [[] for _ in range(num_slots)]
        self.register_buffer('momentum', torch.zeros(num_slots, slot_dim, device=self.device))

        # ✅ Система любопытства (curiosity)
        self.register_buffer(
            "slot_curiosity",
            torch.zeros(num_slots, device=self.device)
        )
        self.curiosity_decay = 0.99
        self.prediction_errors = deque(maxlen=100)

        # ===========================================================
        # 🔥 Кэш для быстрого поиска - создаём сразу на device
        # ===========================================================
        self.register_buffer("_cached_active_centroids", 
                            torch.zeros(0, slot_dim, device=self.device), 
                            persistent=False)
        self.register_buffer("_cached_active_indices", 
                            torch.zeros(0, dtype=torch.long, device=self.device), 
                            persistent=False)
        self._cache_dirty = True

        # Динамическая стратегия
        self.semantic_thresholds = [0.40, 0.55, 0.70]
        self.fill_stages = [256, 450, 512]
        self.new_slot_initial_utility = 0.03

        # Атрибуты для Linker
        self._activation_history: List[int] = []
        self.step_count = 0
        
        # ИСПОЛЬЗУЕМ buffer_size ВМЕСТО ХАРДКОДА
        self.transition_history = deque(maxlen=buffer_size) 
        self._transition_similarities = deque(maxlen=buffer_size)
        
        # Для статистики usage_rate
        self._prev_hit_count = torch.zeros(num_slots, dtype=torch.long, device=self.device)
        self._last_maintenance_step = 0
        
        self._stats = {
            'total_writes': 0, 'total_merges': 0, 'total_splits': 0, 'total_evictions': 0,
            'avg_slot_utility': 0.0,
        }
        
        # ===========================================================
        # 🔥🔥🔥 Neural Linker - СОЗДАЕМ ТОЛЬКО linker, НЕ оптимизатор!
        # ===========================================================
        if enable_linker:
            linker_hidden = linker_hidden_dim or (slot_dim // 2)
            self.linker = NeuralLinker(slot_dim, num_slots, hidden_dim=linker_hidden)
            
            # 🔥🔥🔥 ВАЖНО: оптимизатор НЕ создаём здесь!
            self.linker_optimizer = None  # Будет создан позже при перемещении на устройство
            
            # ✅ Добавляем scheduler (пока None, создадим позже)
            self.linker_scheduler = None
            
            # Счетчики для Linker
            self.linker_updates = 0
            self.linker_losses = []
            
            self.linker.eval()
            
            self._update_lock = threading.Lock()
            
            logger.info(f"🧠 Neural Linker инициализирован: lr={linker_lr}, hidden={linker_hidden}")
            logger.info(f"   • updates={self.linker_updates}, train_freq={linker_train_frequency}")
            logger.info(f"   • buffer_size={buffer_size}")  # ← ДОБАВЛЕН ЛОГ
        else:
            self.linker = None
            self.linker_optimizer = None
            self.linker_updates = 0
            self.linker_losses = []
            logger.info("⚠️ Neural Linker отключён в конфигурации")

        logger.info(f"🧠 CentroidMemoryManager v1.4.3 | слотов: {num_slots}, ядро: {core_slots}, linker: {enable_linker}, buffer_size: {buffer_size}")  # ← ОБНОВЛЕН ЛОГ

# ===================================================================
# Repulsion loss против semantic collapse
# ===================================================================

    def compute_repulsion_loss(self, strength: float = 0.001) -> torch.Tensor:
        """
        Предотвращает semantic collapse - поощряет ортогональность центроидов
        """
        if not self.slot_active.any():
            return torch.tensor(0.0, device=self.centroids.device)
        
        active_mask = self.slot_active
        active_centroids = self.centroids[active_mask]
        active_centroids = F.normalize(active_centroids, dim=-1, eps=1e-8)
        
        # Матрица сходств
        sim_matrix = torch.mm(active_centroids, active_centroids.t())
        
        # Убираем диагональ
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        
        # Поощряем ортогональность (минимизируем квадраты сходств)
        repulsion = (sim_matrix[mask] ** 2).mean()
        
        return repulsion * strength

# ===================================================================
# Curiosity signal (безопасная версия)
# ===================================================================

    def update_curiosity(self, predicted_slot: int, actual_slot: int, confidence: float):
        """
        Обновляет любопытство - ВЕКТОРИЗОВАННАЯ версия
        """
        if predicted_slot != actual_slot:
            error_strength = 1.0 - confidence
            self.slot_curiosity[actual_slot] += error_strength
            self.prediction_errors.append(error_strength)
        
        # Векторизованная проверка и обновление
        high_curiosity_mask = self.slot_curiosity > 0.5
        if high_curiosity_mask.any():
            # Обновляем все слоты с высоким curiosity сразу
            self.slot_utility[high_curiosity_mask] = torch.min(
                torch.ones_like(self.slot_utility[high_curiosity_mask]),
                self.slot_utility[high_curiosity_mask] + 0.05
            )

# ===================================================================
# ОСНОВНОЙ МЕТОД ЗАПИСИ 
# ===================================================================

    def update_slot_centroid(self, thought_vector: torch.Tensor, thought_id: int, 
                             thought_delta: float = 0.5, force_slot=None):
        """
        🔥 ОПТИМИЗИРОВАННАЯ версия с ГАРАНТИРОВАННОЙ синхронизацией устройств
        """
        # 1. ГАРАНТИРУЕМ, ЧТО ВСЁ НА ПРАВИЛЬНОМ УСТРОЙСТВЕ
        device = self.centroids.device
        
        if thought_vector.device != device:
            thought_vector = thought_vector.to(device)
        
        self.step_count += 1
        thought_vector = F.normalize(thought_vector.view(-1), dim=-1)

        result = {'slot_idx': -1, 'action': 'none', 'similarity': 0.0, 'links_created': 0}

        # Динамический порог
        active_count = self.slot_active.sum().item()
        if active_count <= 256:
            semantic_threshold = self.semantic_thresholds[0]
        elif active_count <= 450:
            semantic_threshold = self.semantic_thresholds[1]
        else:
            semantic_threshold = self.semantic_thresholds[2]

        # 2. Поиск ближайшего слота - С ПРИНУДИТЕЛЬНОЙ СИНХРОНИЗАЦИЕЙ
        if self.slot_active.any():
            # 🔥 КРИТИЧЕСКИ ВАЖНО: все индексы и тензоры на device!
            active_indices = torch.where(self.slot_active)[0].to(device)
            
            # Явно загружаем центроиды и нормализуем
            active_centroids = self.centroids[active_indices].to(device)
            active_centroids = F.normalize(active_centroids, dim=-1)
            
            # Векторизованное вычисление сходств
            similarities = F.cosine_similarity(
                thought_vector.unsqueeze(0), 
                active_centroids, 
                dim=-1
            )
            max_sim, best_local = similarities.max(dim=0)
            max_similarity = max_sim.item()
            best_slot = active_indices[best_local].item()
        else:
            max_similarity = 0.0
            best_slot = self.core_slots

        result['similarity'] = max_similarity

        # 3. Решение
        if max_similarity >= semantic_threshold:
            result['action'] = 'semantic_merge'
            result['slot_idx'] = best_slot
            self._oja_update(best_slot, thought_vector, thought_delta * 0.8)
        else:
            empty_slot = self._find_empty_slot()
            if empty_slot is not None:
                result['action'] = 'new_slot'
                result['slot_idx'] = empty_slot
                self._initialize_slot(empty_slot, thought_vector)
            else:
                slot_idx = self._find_least_useful_slot()
                result['action'] = 'replace_weakest'
                result['slot_idx'] = slot_idx
                self._oja_update(slot_idx, thought_vector, thought_delta)

        slot_idx = result['slot_idx']

        # 4. Метаданные
        if thought_id not in self.membership[slot_idx]:
            self.membership[slot_idx].append(thought_id)
            if len(self.membership[slot_idx]) > MAX_MEMBERSHIP_PER_SLOT:
                self.membership[slot_idx] = self.membership[slot_idx][-MAX_MEMBERSHIP_PER_SLOT:]

        self.slot_write_count[slot_idx] += 1
        self.slot_last_activation[slot_idx] = self.step_count

        if thought_delta > 0.7:
            self.slot_utility[slot_idx] = min(1.0, self.slot_utility[slot_idx].item() + UTILITY_BOOST_STRONG)

        # 5. Обновление variance - передаем thought_vector который уже на правильном устройстве
        self._update_variance(slot_idx, thought_vector)
        
        # 6. Обновление переходов
        self._update_transition_matrix(slot_idx)
        
        # 7. Refinement через Linker
        if self.enable_linker and self.linker is not None and self.step_count % 5 == 0:
            links_created = self._neural_transition_refinement(slot_idx)
            result['links_created'] = links_created

        self._stats['total_writes'] += 1

        return result
    
    def _neural_transition_refinement(self, current_slot_idx: int):
        """
        Создает семантические связи между слотами - ОПТИМИЗИРОВАННАЯ версия
        """
        if not self.enable_linker or self.linker is None:
            return 0
        
        if not self.slot_active[current_slot_idx]:
            return 0
        
        try:
            with torch.no_grad():
                # Вектор текущего слота + его embedding
                current_vec = self.centroids[current_slot_idx].unsqueeze(0)
                current_embed = self.slot_embeddings(
                    torch.tensor([current_slot_idx], device=self.device)
                )
                
                # Предсказание связей через улучшенный linker
                raw_links = self.linker(current_vec, slot_embed=current_embed).squeeze(0)
                
                # Маска (только активные слоты, не себя, не ядро)
                mask = self.slot_active.clone()
                mask[current_slot_idx] = False
                mask[:self.core_slots] = False
                
                # Сигмоида -> вероятности
                links = torch.sigmoid(raw_links) * mask.float()
                
                # Нормируем
                if links.sum() > 0:
                    links = links / links.sum() * 0.2
                
                # Обновляем прямые связи (векторизованно)
                self.transition_matrix.data[current_slot_idx] = torch.clamp(
                    0.9 * self.transition_matrix[current_slot_idx] + 0.1 * links,
                    0, 1
                )
                
                # Обновляем обратные связи (векторизованно)
                reverse = links * 0.2
                self.transition_matrix.data[:, current_slot_idx] = torch.clamp(
                    0.9 * self.transition_matrix[:, current_slot_idx] + 0.1 * reverse,
                    0, 1
                )
                
                # Подсчет количества связей (один .item() в конце)
                return (links > 0.01).sum().item()
        except Exception as e:
            logger.debug(f"⚠ Linker refinement error: {e}")
            return 0

# ===================================================================
# 🔥 МЕТОД ОБУЧЕНИЯ LINKER (с embeddings)
# ===================================================================

    def _ensure_linker_optimizer(self, device=None):
        """ИСПРАВЛЕНИЕ 6.1: Создает оптимизатор после перемещения linker на правильное устройство"""
        if not self.enable_linker or self.linker is None:
            return
        
        if device is None:
            device = self.centroids.device
        
        # Убеждаемся, что linker на правильном устройстве
        linker_device = next(self.linker.parameters()).device
        if linker_device != device:
            logger.info(f"🔄 Перемещаю linker с {linker_device} на {device}")
            self.linker = self.linker.to(device)
            # 🔥🔥🔥 ВАЖНО: после перемещения linker, старый оптимизатор больше недействителен!
            self.linker_optimizer = None
            self.linker_scheduler = None
        
        # Создаём оптимизатор только если его нет
        if self.linker_optimizer is None:
            self.linker_optimizer = torch.optim.AdamW(
                self.linker.parameters(),  # теперь параметры точно на device
                lr=self.linker_lr,
                weight_decay=self.linker_weight_decay
            )
            
            # ✅ LINKER-2: Создаём scheduler с правильным устройством
            self.linker_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.linker_optimizer, 
                mode='min', 
                factor=0.5, 
                patience=20, 
                min_lr=1e-5
            )
            logger.info(f"✅ Оптимизатор и scheduler linker созданы на {device}")
        
        return self.linker_optimizer

    def _get_valid_transition_pairs(self, min_confidence: float = 0.05) -> List[Tuple[int, int]]:
        """
        Получает только надежные пары переходов для обучения
        """
        valid_pairs = []
        device = self.centroids.device  # Получаем устройство
        
        for prev, nxt in self.transition_history:
            # Проверяем границы
            if not (0 <= prev < self.num_slots and 0 <= nxt < self.num_slots):
                continue
            
            # Проверяем активность слотов (используем .item() для сравнения)
            if not (self.slot_active[prev].item() and self.slot_active[nxt].item()):
                continue
            
            # Проверяем силу связи в transition_matrix
            transition_strength = self.transition_matrix[prev, nxt].item()
            if transition_strength < min_confidence:
                continue
            
            valid_pairs.append((prev, nxt))
        
        return valid_pairs

    def _train_linker(self):
        """Обучение linker с фильтрацией по качеству переходов - ИСПРАВЛЕНО 2025"""
        
        if not self.enable_linker or self.linker is None:
            return None
        
        # 🔥🔥🔥 КЛЮЧЕВАЯ ПРОВЕРКА: пропускаем обучение если нет градиентов
        if not torch.is_grad_enabled():
            if self.step_count % 200 == 0:
                logger.debug("🔇 Обучение linker'а пропущено (нет градиентов)")
            return None
        
        # 🔥 Проверка устройства и создание оптимизатора при необходимости
        device = self.centroids.device
        self._ensure_linker_optimizer(device)
        
        if self.linker_optimizer is None:
            logger.warning("⚠ Оптимизатор linker не создан, пропускаю обучение")
            return None
        
        if len(self.transition_history) < 32:
            if self.linker_updates == 0 and len(self.transition_history) > 0:
                logger.debug(f"⏳ Neural Linker: накоплено {len(self.transition_history)}/32 пар")
            return None
        
        # Фильтрация по качеству
        high_quality_pairs = []
        
        if len(self._transition_similarities) >= len(self.transition_history):
            sim_list = list(self._transition_similarities)
            history_list = list(self.transition_history)
            
            if sim_list:
                median_sim = float(np.median(sim_list))
                threshold = max(0.15, median_sim * 0.8)
            else:
                threshold = 0.15
            
            for i, (prev, nxt) in enumerate(history_list):
                if i < len(sim_list) and sim_list[i] > threshold:
                    high_quality_pairs.append((prev, nxt))
            
            logger.debug(f"🔍 Linker: порог={threshold:.3f}, качественных={len(high_quality_pairs)}/{len(history_list)}")
        
        if len(high_quality_pairs) < 24:
            logger.debug(f"⚠ Недостаточно качественных пар: {len(high_quality_pairs)} < 16")
            return None
        
        # Группируем и балансируем
        pairs_by_target = defaultdict(list)
        
        for prev, nxt in high_quality_pairs:
            if not (0 <= prev < self.num_slots and 0 <= nxt < self.num_slots):
                continue
            if not (self.slot_active[prev] and self.slot_active[nxt]):
                continue
            
            pairs_by_target[nxt].append((prev, nxt))
        
        # Балансировка
        balanced_pairs = []
        target_counts = {}
        
        for target, pairs in pairs_by_target.items():
            target_counts[target] = len(pairs)
            sample_size = min(20, len(pairs))
            balanced_pairs.extend(random.sample(pairs, sample_size))
        
        logger.debug(f"📊 Распределение после балансировки: {target_counts}")
        
        if len(balanced_pairs) < 24:
            logger.debug(f"⚠ После балансировки только {len(balanced_pairs)} пар")
            return None
        
        # Сэмплируем батч
        batch_size = min(32, len(balanced_pairs))
        sampled_pairs = random.sample(balanced_pairs, batch_size)
        
        device = self.centroids.device
        
        # Сбор данных
        prev_vecs = []
        prev_embeds = []
        target_slots = []
        
        for prev, nxt in sampled_pairs:
            prev_vecs.append(self.centroids[prev].detach())
            prev_embeds.append(self.slot_embeddings.weight[prev].detach())
            target_slots.append(nxt)
        
        X = torch.stack(prev_vecs)
        
        if torch.isnan(X).any() or torch.isinf(X).any():
            logger.warning("⚠ NaN в X, пропускаем батч")
            return None
        
        if X.norm(dim=-1).min() < 1e-6:
            logger.warning("⚠ Нулевые векторы в X, пропускаем")
            return None
        
        X = F.normalize(X, dim=-1, eps=1e-8)
        
        E = torch.stack(prev_embeds)
        E = F.normalize(E, dim=-1, eps=1e-8)
        
        y = torch.tensor(target_slots, device=device, dtype=torch.long)
        
        # Улучшенные веса классов
        class_counts = torch.bincount(y, minlength=self.num_slots)
        class_weights = 1.0 / (class_counts.float().clamp(min=1.0))
        class_weights = class_weights / class_weights.sum() * class_counts[class_counts > 0].numel()
        
        # Обучение
        self.linker.train()
        self.linker = self.linker.to(device)
        
        pred = self.linker(X, slot_embed=E)
        
        loss = F.cross_entropy(pred, y, weight=class_weights.to(device))
        
        # Добавляем repulsion loss
        repulsion = self.compute_repulsion_loss(strength=0.001)
        loss = loss + repulsion
        
        loss_val = loss.item()
        
        if math.isnan(loss_val) or math.isinf(loss_val):
            logger.error(f"❌ loss={loss_val}, пропускаю обновление (NaN/Inf)")
            return None
        
        if loss_val > 10.0:
            logger.warning(f"⚠ Огромный loss={loss_val:.2f}, пропускаю обновление")
            return None
        
        if loss_val < 0.0:
            logger.warning(f"⚠ Отрицательный loss={loss_val:.2f}, пропускаю обновление")
            return None
        
        self.linker_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.linker.parameters(), 0.5)
        self.linker_optimizer.step()
        
        self.linker_updates += 1
        self.linker_losses.append(loss_val)
        
        # Шаг scheduler
        if hasattr(self, 'linker_scheduler') and self.linker_scheduler is not None:
            old_lr = self.linker_optimizer.param_groups[0]['lr']
            self.linker_scheduler.step(loss_val)
            new_lr = self.linker_optimizer.param_groups[0]['lr']
            
            if abs(new_lr - old_lr) > 1e-6:
                logger.info(f"📉 Learning rate уменьшен: {old_lr:.6f} → {new_lr:.6f}")
            
            if self.linker_updates % 10 == 0:
                logger.debug(f"📉 Текущий LR: {new_lr:.6f}")
        
        self.linker.eval()
        
        # Вычисление точности и обновление curiosity (ВЕКТОРИЗОВАННО)
        with torch.no_grad():
            pred_probs = F.softmax(pred, dim=-1)
            pred_labels = pred_probs.argmax(dim=1)
            
            correct = (pred_labels == y).sum().item()
            accuracy_top1 = correct / len(y)
            
            top3_pred = pred_probs.topk(3, dim=1).indices
            correct_top3 = (top3_pred == y.unsqueeze(-1)).any(dim=1).float().mean().item()
            
            confidence_in_correct = pred_probs[torch.arange(len(y)), y].mean().item()
            entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=1).mean().item()
            
            # 🔥 ВЕКТОРИЗОВАННОЕ обновление curiosity
            errors = (pred_labels != y)
            if errors.any():
                error_slots = y[errors]
                error_confs = pred_probs[errors, y[errors]]
                self.slot_curiosity[error_slots] += (1.0 - error_confs)
        
        # Логгирование
        if self.linker_updates == 1:
            logger.info(f"🧠 Neural Linker: ПЕРВОЕ ОБУЧЕНИЕ! loss={loss_val:.4f}")
            logger.info(f"   📊 Метрики: топ-1={accuracy_top1:.2%}, топ-3={correct_top3:.2%}, "
                        f"уверенность={confidence_in_correct:.3f}, энтропия={entropy:.3f}")
        elif self.linker_updates % 10 == 0:
            avg_loss = sum(self.linker_losses[-10:]) / min(10, len(self.linker_losses))
            logger.info(f"🧠 Neural Linker: обновление #{self.linker_updates}, "
                        f"loss={loss_val:.4f}, avg_loss={avg_loss:.4f}")
            logger.info(f"   📊 Метрики: топ-1={accuracy_top1:.2%}, топ-3={correct_top3:.2%}, "
                        f"уверенность={confidence_in_correct:.3f}, энтропия={entropy:.3f}")
            
            unique_targets = len(set(target_slots))
            most_common = max(set(target_slots), key=target_slots.count) if target_slots else "None"
            logger.info(f"   Батч: {unique_targets} уникальных target, "
                        f"мажоритарный: {most_common}")
        
        return loss_val

    def _ensure_linker_device(self):
        """
        Гарантирует, что все компоненты linker на правильном устройстве
        """
        if self.linker is None:
            return
        
        device = self.centroids.device
        linker_device = next(self.linker.parameters()).device
        
        if linker_device != device:
            logger.debug(f"🔄 Перемещаю linker с {linker_device} на {device}")
            self.linker = self.linker.to(device)
            
            # Также перемещаем состояния оптимизатора
            if self.linker_optimizer is not None:
                for state in self.linker_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            state[k] = v.to(device)

# ===================================================================
# Улучшенное обновление variance (EMA)
# ===================================================================
    def _update_variance(self, slot_idx: int, new_vector: torch.Tensor):
        """🔥 ВЕКТОРИЗОВАННОЕ обновление variance с синхронизацией устройств"""
        # 🔥 ИСПРАВЛЕНИЕ: тензорное сравнение вместо .item()
        if self.slot_write_count[slot_idx].item() < 2:
            return
        
        # 🔥 КРИТИЧЕСКИ ВАЖНО: ГАРАНТИРУЕМ, ЧТО ВСЁ НА ОДНОМ УСТРОЙСТВЕ
        device = self.centroids.device
        
        # Убеждаемся, что centroid на правильном устройстве
        centroid = self.centroids[slot_idx].to(device)
        
        # Убеждаемся, что new_vector на правильном устройстве
        new_vector = new_vector.to(device)
        
        # Векторизованное вычисление расстояния
        distance = 1.0 - F.cosine_similarity(
            new_vector.unsqueeze(0), 
            centroid.unsqueeze(0)
        )
        
        alpha = 0.1
        self.slot_variance[slot_idx] = (1 - alpha) * self.slot_variance[slot_idx] + alpha * (distance ** 2)

# ===================================================================
# Oja update
# ===================================================================
    def _oja_update(self, slot_idx: int, thought_vector: torch.Tensor, delta: float):
        """
        🔥 ИСПРАВЛЕНО: slot_idx - Python int, всё остальное на устройстве
        """
        with torch.no_grad():
            # slot_idx - уже Python int, не требует устройства
            w = self.centroids[slot_idx]
            momentum_val = self.momentum[slot_idx]
            
            # thought_vector уже должен быть на правильном устройстве
            x = thought_vector.to(w.device)
            x = F.normalize(x, dim=-1, eps=1e-8)
            
            y = (w * x).sum()
            
            eta = 0.05 * (0.3 + 0.7 * delta)
            
            gradient = y * (x - y * w)
            gradient = torch.clamp(gradient, -1.0, 1.0)
            
            new_momentum = 0.9 * momentum_val + 0.1 * gradient
            self.momentum.data[slot_idx] = new_momentum
            
            w_new = w + eta * new_momentum
            self.centroids.data[slot_idx] = F.normalize(w_new, dim=-1, eps=1e-8)
            
            # Убеждаемся, что слот активен
            if not self.slot_active[slot_idx]:
                self.slot_active[slot_idx] = True
            
            self._cache_dirty = True

    def _initialize_slot(self, slot_idx: int, thought_vector: torch.Tensor):
        """Инициализация нового слота с проверкой устройства"""
        with torch.no_grad():
            # 🔥 ВАЖНО: убеждаемся, что вектор на правильном устройстве
            vec = thought_vector.to(self.centroids.device)
            self.centroids.data[slot_idx] = F.normalize(vec, dim=-1, eps=1e-8)
            self.slot_active[slot_idx] = True
            self.slot_write_count[slot_idx] = 1
            self.slot_utility[slot_idx] = self.new_slot_initial_utility
            self.slot_variance[slot_idx] = 0.0
            self.slot_age[slot_idx] = 0
            self.momentum[slot_idx].zero_()
            self.membership[slot_idx] = []
            self._cache_dirty = True

    def _find_empty_slot(self) -> Optional[int]:
        """Находит пустой слот"""
        self._ensure_device_consistency()  # гарантия
        
        inactive_mask = ~self.slot_active
        inactive_mask[:self.core_slots] = False
        
        if inactive_mask.any():
            indices = torch.where(inactive_mask)[0]
            return int(indices[0].item())
        return None

    def _find_least_useful_slot(self) -> int:
        """🔥 ИСПРАВЛЕНИЕ BUG-2: защита от пустой маски"""
        utilities = self.slot_utility.clone()
        
        # Маски одним векторизованным действием
        mask = torch.ones_like(utilities, dtype=torch.bool)
        mask[:self.core_slots] = False
        mask[self.slot_frozen] = False
        mask[~self.slot_active] = False
        
        # 🔧 ИСПРАВЛЕНИЕ BUG-2: проверяем, есть ли доступные слоты
        if not mask.any():
            # Fallback: берём любой не-ядерный слот
            mask[:self.core_slots] = False
            if not mask.any():
                return self.core_slots  # крайний случай
        
        utilities[~mask] = float('inf')
        return utilities.argmin().item()

    def _update_transition_matrix(self, current_slot: int):
        """ИСПРАВЛЕНИЕ 6.2: Обновление матрицы переходов с синхронизацией истории"""
        if len(self._activation_history) == 0:
            self._activation_history.append(current_slot)
            return
        
        prev_slot = self._activation_history[-1]
        if prev_slot == current_slot:
            return
        
        # Вычисляем семантическую близость
        prev_centroid = self.centroids[prev_slot]
        curr_centroid = self.centroids[current_slot]
        
        semantic_sim = F.cosine_similarity(
            prev_centroid.unsqueeze(0),
            curr_centroid.unsqueeze(0)
        ).item()
        
        # ИСПРАВЛЕНИЕ 6.2: Всегда добавляем в историю, но с меткой качества
        with torch.no_grad():
            self.transition_history.append((prev_slot, current_slot))
            self._transition_similarities.append(semantic_sim)
            
            # Обновляем матрицу только для качественных переходов
            if semantic_sim > 0.3:
                self.transition_matrix.data[prev_slot, current_slot] = torch.clamp(
                    self.transition_matrix[prev_slot, current_slot] + 0.2,
                    max=1.0
                )
                self.transition_matrix.data[current_slot, prev_slot] = torch.clamp(
                    self.transition_matrix[current_slot, prev_slot] + 0.06,  # 0.2 * 0.3 = 0.06
                    max=1.0
                )

        self._activation_history.append(current_slot)
        if len(self._activation_history) > 100:
            self._activation_history = self._activation_history[-100:]
        
        # Ограничиваем размер истории
        if len(self.transition_history) > 2000:
            self.transition_history = deque(list(self.transition_history)[-2000:], maxlen=2000)
            self._transition_similarities = deque(list(self._transition_similarities)[-2000:], maxlen=2000)

# ===================================================================
# ЗАПРОС К ПАМЯТИ
# ===================================================================
    def _update_cache(self):
        """Обновляет кэш активных центроидов"""
        if not self._cache_dirty:
            return
        
        device = self.centroids.device
        
        active_mask = self.slot_active
        if active_mask.any():
            indices = torch.where(active_mask)[0]
            self._cached_active_indices = indices.clone().to(device)
            self._cached_active_centroids = self.centroids[indices].clone().to(device)
        else:
            self._cached_active_indices = torch.zeros(0, dtype=torch.long, device=device)
            self._cached_active_centroids = torch.zeros(0, self.slot_dim, device=device)
        
        self._cache_dirty = False
        # Логируем только при значительных изменениях
        if len(self._cached_active_indices) > 10:
            logger.debug(f"🔄 Кэш обновлён: {len(self._cached_active_indices)} слотов")

    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        use_transitions: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        🔥 ПОЛНОСТЬЮ ВЕКТОРИЗОВАННЫЙ запрос к памяти (без циклов, без .item())
        """
        # 1. Приводим запрос к устройству модели
        input_device = query_vector.device
        model_device = self.centroids.device
        
        query_vector = query_vector.to(model_device)
        query_vector = F.normalize(query_vector.view(-1), dim=-1)
        
        # 2. Проверяем кэш
        if self._cache_dirty:
            self._update_cache()
        
        # 3. Проверяем наличие активных слотов
        if len(self._cached_active_indices) == 0:
            return torch.zeros(self.slot_dim, device=input_device), {
                'found': 0,
                'top_slots': [],
                'top_sims': [],
                'weights': []
            }
        
        # 4. Используем кэш
        active_indices = self._cached_active_indices
        active_centroids = self._cached_active_centroids
        
        # 5. Считаем схожесть
        similarities = F.cosine_similarity(
            query_vector.unsqueeze(0), 
            active_centroids, 
            dim=-1
        )
        
        # ===========================================================
        # 🔥 ВЕКТОРИЗОВАННЫЕ ШТРАФЫ/БОНУСЫ (БЕЗ ЦИКЛА!)
        # ===========================================================
        hits = self.slot_hit_count[active_indices].float()
        
        # Штраф для очень частых (hits > 100000)
        high_hits_mask = hits > 100000
        if high_hits_mask.any():
            penalty = torch.min(
                torch.full_like(hits[high_hits_mask], 0.5),
                (hits[high_hits_mask] - 100000) / 200000
            )
            similarities[high_hits_mask] -= penalty
        
        # Бонус для редких (hits < 1000)
        low_hits_mask = hits < 1000
        if low_hits_mask.any():
            bonus = 0.1 * (1.0 - hits[low_hits_mask] / 1000)
            similarities[low_hits_mask] += bonus
        
        # 6. Добавляем влияние переходов (векторизованно)
        if use_transitions and len(self._activation_history) > 0:
            recent_slot = self._activation_history[-1]
            boost = self.transition_matrix[recent_slot][active_indices]
            similarities = similarities + 0.2 * boost
        
        # 7. Выбираем top-k
        k = min(top_k, len(active_indices))
        top_sims, top_local = similarities.topk(k)
        top_slots = active_indices[top_local]
        
        # ===========================================================
        # 🔥 ВЕКТОРИЗОВАННОЕ ОБНОВЛЕНИЕ СТАТИСТИКИ
        # ===========================================================
        # Обновляем hit_count (векторизованно)
        self.slot_hit_count[top_slots] += 1
        
        # Обновляем utility (векторизованно, без .item())
        self.slot_utility[top_slots] = torch.min(
            torch.ones_like(self.slot_utility[top_slots]),
            self.slot_utility[top_slots] + 0.035
        )
        
        # 8. Обновляем историю переходов
        if use_transitions:
            current_slot = top_slots[0].item()  # только один .item() в конце
            
            if len(self._activation_history) > 0:
                prev_slot = self._activation_history[-1]
                if prev_slot != current_slot:
                    # Можно добавить логику если нужно
                    pass
            
            self._activation_history.append(current_slot)
            if len(self._activation_history) > 100:
                self._activation_history = self._activation_history[-100:]
        
        # 9. Формируем контекст
        top_centroids = self.centroids[top_slots]
        weights = F.softmax(top_sims, dim=0)
        context = torch.sum(top_centroids * weights.unsqueeze(-1), dim=0)
        context = F.normalize(context, dim=-1)
        context = context.to(input_device)
        
        return context, {
            'found': k,
            'top_slots': top_slots.cpu().tolist(),
            'top_sims': top_sims.cpu().tolist(),
            'weights': weights.cpu().tolist(),
        }

    def to(self, device):
        """🔧 ИСПРАВЛЕНИЕ 1: Перемещает все компоненты на указанное устройство с правильным созданием оптимизатора"""
        if device == self.device and not self._is_meta:
            return self
        
        logger.debug(f"📱 Перемещение CentroidMemoryManager с {self.device} на {device}")
        
        # 🔥 ВАЖНО: Сначала перемещаем все параметры через super()
        result = super().to(device)
        
        # Дополнительная обработка
        self.device = device
        self._is_meta = False
        
        # Сбрасываем кэш
        self._cached_active_indices = torch.zeros(0, dtype=torch.long, device=device)
        self._cached_active_centroids = torch.zeros(0, self.slot_dim, device=device)
        
        # 🔥🔥🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Перемещаем linker и ПЕРЕСОЗДАЁМ оптимизатор
        if hasattr(self, 'linker') and self.linker is not None:
            # Сначала перемещаем linker
            self.linker = self.linker.to(device)
            
            # ✅ LINKER-2: ВСЕГДА пересоздаём оптимизатор после перемещения!
            self.linker_optimizer = torch.optim.AdamW(
                self.linker.parameters(),
                lr=self.linker_lr,
                weight_decay=self.linker_weight_decay
            )
            
            # ✅ LINKER-3: Создаём scheduler
            self.linker_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.linker_optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=1e-5
            )
            
            logger.info(f"✅ Оптимизатор и scheduler linker'а пересозданы на {device}")
        
        self._cache_dirty = True
        
        logger.debug(f"✅ CentroidMemoryManager перемещён на {device}")
        return result

    def _ensure_device_consistency(self):
        """
        🔥 ГАРАНТИРУЕТ, что все тензоры на одном устройстве
        """
        if self._is_meta:
            return
        
        main_device = self.centroids.device
        any_mismatch = False
        
        # Список всех буферов для проверки
        buffers_to_check = [
            ('centroids', self.centroids),
            ('transition_matrix', self.transition_matrix),
            ('slot_utility', self.slot_utility),
            ('slot_hit_count', self.slot_hit_count),
            ('slot_write_count', self.slot_write_count),
            ('slot_variance', self.slot_variance),
            ('slot_age', self.slot_age),
            ('slot_frozen', self.slot_frozen),
            ('slot_last_activation', self.slot_last_activation),
            ('slot_active', self.slot_active),
            ('momentum', self.momentum),
            ('slot_curiosity', self.slot_curiosity),
            ('slot_embeddings.weight', self.slot_embeddings.weight),
        ]
        
        # Проверяем каждый буфер
        for name, tensor in buffers_to_check:
            if tensor.device != main_device:
                logger.warning(f"⚠ {name} на {tensor.device}, перемещаю на {main_device}")
                any_mismatch = True
                if isinstance(tensor, nn.Parameter):
                    tensor.data = tensor.data.to(main_device)
                else:
                    # Для register_buffer и других тензоров
                    if '.' in name:
                        parent_name, child_name = name.split('.')
                        parent = getattr(self, parent_name)
                        if hasattr(parent, child_name):
                            setattr(parent, child_name, tensor.to(main_device))
                    else:
                        setattr(self, name, tensor.to(main_device))
        
        # 🔥🔥🔥 ВАЖНО: проверяем кэш
        if hasattr(self, '_cached_active_indices'):
            if self._cached_active_indices.device != main_device:
                logger.warning(f"⚠ _cached_active_indices на {self._cached_active_indices.device}, ПЕРЕСОЗДАЮ")
                self._cached_active_indices = torch.zeros(0, dtype=torch.long, device=main_device)
                any_mismatch = True
        
        if hasattr(self, '_cached_active_centroids'):
            if self._cached_active_centroids.device != main_device:
                logger.warning(f"⚠ _cached_active_centroids на {self._cached_active_centroids.device}, ПЕРЕСОЗДАЮ")
                self._cached_active_centroids = torch.zeros(0, self.slot_dim, device=main_device)
                any_mismatch = True
        
        # Если были несоответствия, помечаем кэш как грязный
        if any_mismatch:
            self._cache_dirty = True
            logger.info(f"✅ Исправлены несоответствия устройств, все на {main_device}")
 
    def reset_cache(self):
        """
        🔥 Принудительный сброс кэша
        """
        device = self.centroids.device
        self._cached_active_indices = torch.zeros(0, dtype=torch.long, device=device)
        self._cached_active_centroids = torch.zeros(0, self.slot_dim, device=device)
        self._cache_dirty = True
        logger.debug(f"🔄 Кэш принудительно сброшен на {device}")
        return self
 
# ===================================================================
# ОБСЛУЖИВАНИЕ ПАМЯТИ (с adaptive utility decay)
# ===================================================================
    def run_maintenance(self) -> Dict[str, int]:
        """🚀 ОПТИМИЗИРОВАННОЕ обслуживание памяти с синхронизацией embeddings"""
        stats = {'merged': 0, 'split': 0, 'evicted': 0}
        
        # ===========================================================
        # 🔥 ОПТИМИЗИРОВАННАЯ проверка устройств (реже)
        # ===========================================================
        if not hasattr(self, '_maintenance_counter'):
            self._maintenance_counter = 0
        
        self._maintenance_counter += 1
        
        # Проверяем устройства только если кэш грязный или раз в 5 вызовов
        if self._cache_dirty or self._maintenance_counter % 5 == 0:
            self._ensure_device_consistency()
            if self._maintenance_counter % 5 == 0:
                logger.debug(f"🧹 Плановая проверка устройств (вызов {self._maintenance_counter})")
        
        # 🔧 ИСПРАВЛЕНИЕ #9: Применяем decay к curiosity здесь (раз за обслуживание)
        self.slot_curiosity *= self.curiosity_decay
        
        # Получаем активные индексы и сразу перемещаем на правильное устройство
        device = self.centroids.device
        active_mask = self.slot_active.clone()
        active_mask[:self.core_slots] = False
        active_indices = torch.where(active_mask)[0].to(device)  # 🔥 Явно на device
        
        if len(active_indices) < 2:
            return stats
        
        # ===================================================================
        # MERGE
        # ===================================================================
        active_centroids = self.centroids[active_indices]
        active_centroids_norm = F.normalize(active_centroids, dim=-1)
        
        sim_matrix = torch.mm(active_centroids_norm, active_centroids_norm.t())
        
        triu_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        merge_candidates = (sim_matrix > self.merge_threshold) & triu_mask
        
        if merge_candidates.any():
            pairs = torch.nonzero(merge_candidates)
            
            sim_values = sim_matrix[merge_candidates]
            sorted_order = torch.argsort(sim_values, descending=True)
            pairs = pairs[sorted_order]
            
            merged_set = set()
            for pair in pairs:
                # 🔥 ВАЖНО: конвертируем в Python int для использования в индексах
                i, j = pair.tolist()
                slot_a = active_indices[i].item()  # .item() даёт Python int
                slot_b = active_indices[j].item()
                
                if slot_a in merged_set or slot_b in merged_set:
                    continue
                if self.slot_frozen[slot_a].item() or self.slot_frozen[slot_b].item():  # .item() для bool
                    continue
                
                self._merge_slots(slot_a, slot_b)
                merged_set.add(slot_b)
                stats['merged'] += 1
                
                if stats['merged'] >= 3:
                    break
        
        # ===================================================================
        # SPLIT
        # ===================================================================
        # Обновляем активные индексы (могли измениться после merge)
        active_mask = self.slot_active.clone()
        active_mask[:self.core_slots] = False
        active_indices = torch.where(active_mask)[0].to(device)
        
        if len(active_indices) > 0:
            variances = self.slot_variance[active_indices]
            high_var_mask = variances > self.split_variance_threshold
            
            if high_var_mask.any():
                split_candidates = active_indices[high_var_mask]
                
                max_splits = min(2, len(split_candidates))
                if max_splits > 0:
                    # Сортируем и получаем индексы
                    sorted_idx = torch.argsort(variances[high_var_mask], descending=True)
                    for idx in sorted_idx[:max_splits]:
                        slot_idx = split_candidates[idx].item()  # .item() для Python int
                        
                        empty_slot = self._find_empty_slot()
                        if empty_slot is not None:
                            self._split_slot(slot_idx, empty_slot)
                            stats['split'] += 1
        
        # ===================================================================
        # EVICTION
        # ===================================================================
        # Обновляем активные индексы
        active_mask = self.slot_active.clone()
        active_mask[:self.core_slots] = False
        active_indices = torch.where(active_mask)[0].to(device)
        
        if len(active_indices) > 0:
            low_utility = self.slot_utility[active_indices] < self.eviction_utility_threshold
            old_enough = self.slot_age[active_indices] > 5
            low_hits = self.slot_hit_count[active_indices] < 3
            
            evict_mask = low_utility & old_enough & low_hits
            
            if evict_mask.any():
                evict_candidates = active_indices[evict_mask]
                
                max_evictions = min(3, len(evict_candidates))
                if max_evictions > 0:
                    utilities = self.slot_utility[evict_candidates]
                    _, sorted_idx = torch.sort(utilities)
                    
                    for idx in sorted_idx[:max_evictions]:
                        slot_idx = evict_candidates[idx].item()  # .item() для Python int
                        self._evict_slot(slot_idx)
                        stats['evicted'] += 1
        
        # ===================================================================
        # ✅ СИНХРОНИЗАЦИЯ EMBEDDINGS С ЦЕНТРОИДАМИ (МЯГКОЕ ОБНОВЛЕНИЕ)
        # ===================================================================
        with torch.no_grad():
            embed_dim = self.slot_embeddings.embedding_dim
            if embed_dim <= self.slot_dim:
                centroid_proj = self.centroids[:, :embed_dim]
            else:
                indices = torch.linspace(0, self.slot_dim - 1, embed_dim).long().to(device)
                centroid_proj = self.centroids[:, indices]
            
            # 🔧 ИСПРАВЛЕНИЕ #2: Мягкое EMA-обновление, а не полная перезапись
            alpha = 0.05
            current_embeds = self.slot_embeddings.weight.data
            new_embeds = F.normalize(centroid_proj, dim=-1, eps=1e-8)
            self.slot_embeddings.weight.data = (1 - alpha) * current_embeds + alpha * new_embeds
        
        # ===================================================================
        # ✅ АДАПТИВНЫЙ DECAY И НОРМАЛИЗАЦИЯ TRANSITION MATRIX
        # ===================================================================
        with torch.no_grad():
            # ===========================================================
            # 🔥 ОПТИМИЗИРОВАННОЕ вычисление usage_rate
            # ===========================================================
            steps_since_last = self.step_count - self._last_maintenance_step
            if steps_since_last > 0 and self.slot_active.any():
                # Гарантируем одинаковое устройство
                if self._prev_hit_count.device != self.slot_hit_count.device:
                    self._prev_hit_count = self._prev_hit_count.to(self.slot_hit_count.device)
                
                # Векторизованное вычисление прироста
                hit_increase = (self.slot_hit_count - self._prev_hit_count).float()
                usage_rate = hit_increase[self.slot_active].mean().item() / steps_since_last
            else:
                usage_rate = 0.0
            
            # Сохраняем текущие хиты для следующего раза
            self._prev_hit_count = self.slot_hit_count.clone()
            self._last_maintenance_step = self.step_count
            
            base_decay = 0.995
            decay = base_decay + 0.004 * min(1.0, usage_rate)
            
            # 🔧 ИСПРАВЛЕНИЕ LOGIC-2: Применяем decay ДО нормализации
            self.transition_matrix.data *= decay
            self.slot_utility *= decay
            
            # 🔧 ИСПРАВЛЕНИЕ LOGIC-6: Безопасное копирование для нормализации
            row_sums = self.transition_matrix.sum(dim=1, keepdim=True) + 1e-8
            self.transition_matrix.data.copy_(self.transition_matrix.data / row_sums)
            
            self.slot_age[self.slot_active] += 1
            
            # Обрезаем membership (используем Python int для индексов)
            active_indices_list = torch.where(self.slot_active)[0].cpu().tolist()
            for slot_idx in active_indices_list:
                if len(self.membership[slot_idx]) > 10:
                    self.membership[slot_idx] = self.membership[slot_idx][-10:]
        
        # Обновляем статистику
        self._stats['total_merges'] += stats['merged']
        self._stats['total_splits'] += stats['split']
        self._stats['total_evictions'] += stats['evicted']
        
        if self.slot_active.any():
            self._stats['avg_slot_utility'] = self.slot_utility[self.slot_active].mean().item()
        
        # ===================================================================
        # 🔥 ОБУЧЕНИЕ NEURAL LINKER (ИСПРАВЛЕНО)
        # ===================================================================
        if (self.enable_linker and len(self.transition_history) >= 32):
            # Убедимся, что оптимизатор создан
            if self.linker_optimizer is None:
                self._ensure_linker_optimizer()
            
            self._ensure_linker_device()
            loss = self._train_linker()
            if loss is not None:
                logger.info(f"🧠 Linker: обучен 1 раз (loss={loss:.4f}, всего={self.linker_updates})")
        
        return stats

    def _merge_slots(self, target_slot: int, source_slot: int):
        """ИСПРАВЛЕНИЕ 6.3: ОПТИМИЗИРОВАННЫЙ merge с нормализацией"""
        with torch.no_grad():
            # target_slot и source_slot уже Python int
            
            n_target = len(self.membership[target_slot]) + 1
            n_source = len(self.membership[source_slot]) + 1
            total = n_target + n_source
            
            target_centroid = self.centroids[target_slot]
            source_centroid = self.centroids[source_slot]
            
            merged = (target_centroid * (n_target / total) + 
                      source_centroid * (n_source / total))
            
            self.centroids.data[target_slot] = F.normalize(merged, dim=-1, eps=1e-8)
            
            self.membership[target_slot].extend(self.membership[source_slot])
            if len(self.membership[target_slot]) > MAX_MEMBERSHIP_PER_SLOT:
                self.membership[target_slot] = self.membership[target_slot][-MAX_MEMBERSHIP_PER_SLOT:]
            
            self.membership[source_slot] = []
            
            # Используем .item() для скаляров
            self.slot_write_count[target_slot] += self.slot_write_count[source_slot].item()
            self.slot_hit_count[target_slot] += self.slot_hit_count[source_slot].item()
            self.slot_utility[target_slot] = max(
                self.slot_utility[target_slot].item(),
                self.slot_utility[source_slot].item()
            )
            
            # ИСПРАВЛЕНИЕ 6.3: Обновляем transition matrix с нормализацией
            self.transition_matrix.data[target_slot] += self.transition_matrix[source_slot]
            self.transition_matrix.data[:, target_slot] += self.transition_matrix[:, source_slot]
            
            # ИСПРАВЛЕНИЕ 6.3: нормализуем, а не просто clamp
            row_sum = self.transition_matrix[target_slot].sum()
            if row_sum > 0:
                self.transition_matrix.data[target_slot] = self.transition_matrix[target_slot] / row_sum
            
            col_sum = self.transition_matrix[:, target_slot].sum()
            if col_sum > 0:
                self.transition_matrix.data[:, target_slot] = self.transition_matrix[:, target_slot] / col_sum
            
            self._evict_slot(source_slot)

    def _split_slot(self, slot_idx: int, new_slot_idx: int):
        """ИСПРАВЛЕНИЕ 6.4: УЛУЧШЕННЫЙ split с более разнообразными центроидами"""
        members = self.membership[slot_idx]
        if len(members) < 2:
            return
        
        # Разделяем membership
        mid = len(members) // 2
        new_members = members[mid:]
        old_members = members[:mid]
        
        self.membership[slot_idx] = old_members
        self.membership[new_slot_idx] = new_members
        
        with torch.no_grad():
            centroid = self.centroids[slot_idx]
            
            # Улучшенный split: разносим в ортогональные направления
            noise1 = F.normalize(torch.randn_like(centroid), dim=-1) * 0.2
            noise2 = F.normalize(torch.randn_like(centroid), dim=-1) * 0.2
            
            # Делаем noise2 ортогональным noise1
            noise2 = noise2 - (noise2 * noise1).sum() * noise1
            noise2 = F.normalize(noise2, dim=-1) * 0.2
            
            self.centroids.data[slot_idx] = F.normalize(centroid + noise1, dim=-1, eps=1e-8)
            self.centroids.data[new_slot_idx] = F.normalize(centroid + noise2, dim=-1, eps=1e-8)
            
            self.slot_active[new_slot_idx] = True
            self.slot_write_count[new_slot_idx] = len(new_members)
            self.slot_utility[new_slot_idx] = self.slot_utility[slot_idx].item() * 0.5
            self.slot_variance[new_slot_idx] = 0.0
            self.slot_age[new_slot_idx] = 0
            
            self.momentum.data[new_slot_idx] = torch.zeros_like(centroid)
            
            self.slot_variance[slot_idx] *= 0.5
            
            self._cache_dirty = True

    def _evict_slot(self, slot_idx: int):
        """Деактивировать слот"""
        with torch.no_grad():
            self.slot_active[slot_idx] = False
            self.centroids.data[slot_idx].zero_()
            self.slot_utility[slot_idx] = 0.0
            self.slot_hit_count[slot_idx] = 0
            self.slot_write_count[slot_idx] = 0
            self.slot_variance[slot_idx] = 0.0
            self.slot_age[slot_idx] = 0
            self.momentum.data[slot_idx].zero_()
            self.transition_matrix.data[slot_idx, :] = 0
            self.transition_matrix.data[:, slot_idx] = 0
            self._cache_dirty = True
        
        self.membership[slot_idx] = []

# ===================================================================
# СОВМЕСТИМОСТЬ С HEBB
# ===================================================================

    def consolidate_hebb_to_centroid(self, thought_data: Dict) -> bool:
        """Вызывается из Hebb-слоя для консолидации сильных паттернов"""
        if 'snapshot' not in thought_data:
            return False

        try:
            snapshot = thought_data['snapshot']
            if isinstance(snapshot, list):
                vector = torch.tensor(snapshot, dtype=torch.float32, device=self.device)
            else:
                vector = snapshot.to(self.device)

            vector = F.normalize(vector.view(-1), dim=-1)

            thought_id = thought_data.get('step', self.step_count)
            delta = thought_data.get('delta', 0.65)

            result = self.update_slot_centroid(
                thought_vector=vector,
                thought_id=thought_id,
                thought_delta=delta
            )

            if result['action'] != 'none':
                logger.debug(f"🔗 Hebb → Centroid: {result['action']} (sim={result['similarity']:.3f})")
                return True
            return False

        except Exception as e:
            logger.debug(f"⚠ Ошибка консолидации Hebb-паттерна: {e}")
            return False

# ===================================================================
# УТИЛИТЫ И АНАЛИЗ
# ===================================================================

    def get_character_graph(self) -> torch.Tensor:
        """Граф характера"""
        row_sums = self.transition_matrix.sum(dim=1, keepdim=True) + 1e-8
        return self.transition_matrix / row_sums

    def get_core_personality(self) -> torch.Tensor:
        """Ядро личности"""
        return self.centroids[:self.core_slots].clone()

    def freeze_core(self):
        """Заморозить ядро личности"""
        self.slot_frozen[:self.core_slots] = True
        logger.info(f"❄️ Ядро личности заморожено: {self.core_slots} слотов")

    def get_stats(self) -> Dict:
        """Статистика"""
        active_count = self.slot_active.sum().item()
        non_zero = (self.transition_matrix.abs() > 0.01).sum().item()
        total = self.num_slots ** 2
        
        stats = {
            'active_slots': active_count,
            'total_slots': self.num_slots,
            'utilization': active_count / self.num_slots if self.num_slots > 0 else 0,
            'avg_utility': self._stats['avg_slot_utility'],
            'total_writes': self._stats['total_writes'],
            'total_merges': self._stats['total_merges'],
            'total_splits': self._stats['total_splits'],
            'total_evictions': self._stats['total_evictions'],
            'transition_density': non_zero / total if total > 0 else 0,
            'transition_non_zero': non_zero,
            'transition_total': total,
            'avg_membership_size': sum(len(m) for m in self.membership) / max(1, active_count),
            'avg_curiosity': self.slot_curiosity.mean().item(),
        }
        
        if self.enable_linker:
            stats['linker_active'] = True
            stats['linker_updates'] = self.linker_updates
            stats['linker_loss'] = sum(self.linker_losses[-10:]) / max(1, len(self.linker_losses[-10:])) if self.linker_losses else 0
            stats['linker_history_size'] = len(self.transition_history)
        
        return stats

    def analyze_transitions(self):
        """Анализирует качество связей в центроидной памяти"""
        if not self.slot_active.any():
            logger.info("🔗 Нет активных слотов для анализа связей")
            return {
                'density': 0.0,
                'mean_strength': 0.0,
                'entropy': 0.0,
                'non_zero': 0,
                'total': 0
            }
        
        active_indices = torch.where(self.slot_active)[0]
        trans = self.transition_matrix[active_indices][:, active_indices]
        
        non_zero = (trans > 0.01).sum().item()
        total = active_indices.size(0) ** 2
        density = non_zero / total if total > 0 else 0
        
        if non_zero > 0:
            mean_strength = trans[trans > 0.01].mean().item()
        else:
            mean_strength = 0.0
        
        row_sums = trans.sum(dim=1, keepdim=True) + 1e-8
        probs = trans / row_sums
        valid_rows = row_sums.squeeze() > 0.01
        if valid_rows.any():
            entropy = -(probs[valid_rows] * (probs[valid_rows] + 1e-10).log()).sum(dim=1).mean().item()
        else:
            entropy = 0.0
        
        logger.info(f"🔗 Анализ связей: плотность={density:.4f} ({non_zero}/{total}), "
                    f"средняя сила={mean_strength:.3f}, энтропия={entropy:.3f}")
        
        return {
            'density': density,
            'mean_strength': mean_strength,
            'entropy': entropy,
            'non_zero': non_zero,
            'total': total,
            'active_slots': active_indices.size(0)
        }

    def show_hubs(self, top_k=5):
        """Показывает самые связные слоты (хабы)"""
        if not self.slot_active.any():
            logger.info("🌟 Нет активных слотов для анализа хабов")
            return []
        
        threshold = 0.01
        in_degree = (self.transition_matrix > threshold).sum(dim=0).float()
        out_degree = (self.transition_matrix > threshold).sum(dim=1).float()
        total_degree = in_degree + out_degree
        
        total_degree[~self.slot_active] = -1
        
        k = min(top_k, (self.slot_active).sum().item())
        if k == 0:
            return []
        
        values, indices = torch.topk(total_degree, k)
        
        logger.info(f"🌟 Топ-{k} хабов центроидной памяти:")
        hubs_info = []
        for i, (idx, val) in enumerate(zip(indices.tolist(), values.tolist())):
            utility = self.slot_utility[idx].item()
            age = self.slot_age[idx].item()
            write_count = self.slot_write_count[idx].item()
            hit_count = self.slot_hit_count[idx].item()
            
            logger.info(f"   #{i+1}: слот {idx}, связей={int(val)}, "
                       f"utility={utility:.3f}, age={age}, "
                       f"записей={write_count}, попаданий={hit_count}")
            
            hubs_info.append({
                'slot': idx,
                'connections': int(val),
                'utility': utility,
                'age': age,
                'write_count': write_count,
                'hit_count': hit_count
            })
        
        return hubs_info

    # Удаление мёртвого кода - оставляем как заглушку с предупреждением
    def visualize_transition_network(self, top_n=20):
        """ИСПРАВЛЕНИЕ 6.6: visualize_transition_network не реализован"""
        logger.warning("visualize_transition_network не реализован")
        return None

    def linker_status(self):
        """Показывает статус Neural Linker (БЕЗОПАСНАЯ ВЕРСИЯ)"""
        if not self.enable_linker or self.linker is None:
            logger.info("🧠 Neural Linker: ОТКЛЮЧЁН")
            return
        
        status = f"🧠 Neural Linker: "
        status += f"обновлений={self.linker_updates}, "
        status += f"история переходов={len(self.transition_history)}"
        
        # ✅ БЕЗОПАСНО проверяем наличие атрибутов
        quality_stats = self.get_transition_quality_stats() if hasattr(self, 'get_transition_quality_stats') else {}
        
        if quality_stats:
            avg_sim = quality_stats.get('avg_semantic_sim', 0.0)
            high_ratio = quality_stats.get('high_quality_ratio', 0.0)
            status += f", семантическое качество={avg_sim:.3f}"
            status += f", high-quality={high_ratio:.1%}"
        
        if self.linker_losses:
            avg_loss = sum(self.linker_losses[-10:]) / min(10, len(self.linker_losses))
            status += f", последний loss={self.linker_losses[-1]:.4f}"
            status += f", avg_loss={avg_loss:.4f}"
        else:
            status += f", ещё не обучался"
        
        logger.info(status)
        
        # Проверяем качество на свежих данных
        if len(self.transition_history) >= 32:
            valid_pairs = self._get_valid_transition_pairs(min_confidence=0.05)
            if len(valid_pairs) >= 16:
                logger.info(f"✅ Доступно {len(valid_pairs)} сильных пар для обучения")
            else:
                logger.info(f"   ⏳ Нужно ещё {16 - len(valid_pairs)} сильных пар")
        
        return {
            'updates': self.linker_updates,
            'history_size': len(self.transition_history),
            'losses': self.linker_losses[-10:] if self.linker_losses else [],
            'quality_stats': quality_stats
        }

    def get_transition_quality_stats(self) -> Dict:
        """БЕЗОПАСНО получает статистику качества переходов"""
        # Проверяем наличие всех необходимых атрибутов
        if not hasattr(self, '_transition_similarities') or not self._transition_similarities:
            return {
                'avg_semantic_sim': 0.0, 
                'transition_count': len(getattr(self, 'transition_history', [])),
                'high_quality_ratio': 0.0,
                'low_quality_ratio': 0.0
            }
        
        recent = list(self._transition_similarities)[-1000:]
        if not recent:
            return {
                'avg_semantic_sim': 0.0,
                'transition_count': len(self.transition_history),
                'high_quality_ratio': 0.0,
                'low_quality_ratio': 0.0
            }
        
        avg_sim = sum(recent) / len(recent)
        
        high_quality = sum(1 for s in recent if s > 0.5)
        low_quality = sum(1 for s in recent if s < 0.2)
        
        return {
            'avg_semantic_sim': avg_sim,
            'transition_count': len(self.transition_history),
            'high_quality_ratio': high_quality / len(recent),
            'low_quality_ratio': low_quality / len(recent),
            'recent_quality': recent[-10:] if recent else []
        }

    def state_dict_custom(self) -> Dict:
        """
        💾 СОХРАНЕНИЕ СОСТОЯНИЯ (с безопасной проверкой атрибутов)
        """
        # Базовые тензоры
        state = {
            'centroids': self.centroids.data.clone(),
            'transition_matrix': self.transition_matrix.data.clone(),
            'slot_utility': self.slot_utility.clone(),
            'slot_hit_count': self.slot_hit_count.clone(),
            'slot_write_count': self.slot_write_count.clone(),
            'slot_variance': self.slot_variance.clone(),
            'slot_age': self.slot_age.clone(),
            'slot_frozen': self.slot_frozen.clone(),
            'slot_active': self.slot_active.clone(),
            'momentum': self.momentum.clone(),
            'slot_curiosity': self.slot_curiosity.clone(),
            'slot_embeddings': self.slot_embeddings.state_dict(),
            'membership': [list(m) for m in self.membership],
            'stats': self._stats.copy(),
            'step_count': self.step_count,
            'activation_history': self._activation_history.copy(),
            'transition_history': list(self.transition_history) if hasattr(self, 'transition_history') else [],
            
            # 🔥 Добавляем информацию о Linker
            'enable_linker': self.enable_linker,
            'linker_lr': self.linker_lr,
            'linker_weight_decay': self.linker_weight_decay,
            'linker_batch_size': self.linker_batch_size,
            'linker_train_frequency': self.linker_train_frequency,
        }
        
        # 🔥 БЕЗОПАСНО сохраняем атрибуты Linker (проверяем существование)
        if hasattr(self, 'linker_updates'):
            state['linker_updates'] = self.linker_updates
        else:
            state['linker_updates'] = 0
        
        if hasattr(self, 'linker_losses'):
            state['linker_losses'] = self.linker_losses.copy()
        else:
            state['linker_losses'] = []
        
        if hasattr(self, '_transition_similarities'):
            state['_transition_similarities'] = list(self._transition_similarities)
        else:
            state['_transition_similarities'] = []
        
        if hasattr(self, '_cache_dirty'):
            state['_cache_dirty'] = self._cache_dirty
        else:
            state['_cache_dirty'] = True
        
        # ===========================================================
        # 🔥🔥🔥 СОХРАНЕНИЕ LINKER (с глубокой копией)
        # ===========================================================
        if self.enable_linker and hasattr(self, 'linker') and self.linker is not None:
            try:
                # Сохраняем веса Linker
                state['linker_state'] = self.linker.state_dict()
                
                
                # ===========================================================
                # 🔧 Сохраняем глубокую копию состояния оптимизатора
                # ===========================================================
                if hasattr(self, 'linker_optimizer') and self.linker_optimizer is not None:
                    # Создаём глубокую копию state_dict
                    opt_state_copy = copy.deepcopy(self.linker_optimizer.state_dict())
                    
                    # Перемещаем все тензоры в копии на CPU для сериализации
                    for group in opt_state_copy.get('state', {}).values():
                        for k, v in group.items():
                            if isinstance(v, torch.Tensor):
                                group[k] = v.cpu()
                    
                    state['linker_optimizer'] = opt_state_copy
                    logger.debug(f"💾 Оптимизатор Linker сохранён на CPU (глубокая копия)")
                
                # ===========================================================
                # ✅ Сохраняем состояние scheduler (если есть)
                # ===========================================================
                if hasattr(self, 'linker_scheduler') and self.linker_scheduler is not None:
                    try:
                        state['linker_scheduler'] = self.linker_scheduler.state_dict()
                        logger.debug(f"💾 Scheduler Linker сохранён")
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка сохранения scheduler'а: {e}")
                
                logger.info(f"🧠 Linker сохранён: обновлений={state['linker_updates']}, "
                           f"история={len(state['transition_history'])}, ")
                
            except Exception as e:
                logger.warning(f"⚠ Ошибка сохранения Linker: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        return state

    def load_state_dict_custom(self, state: Dict):
        """
        📂 ЗАГРУЗКА СОСТОЯНИЯ (с проверкой устройств)
        """
        with torch.no_grad():
            # 🔥 ЗАГРУЗКА БАЗОВЫХ ТЕНЗОРОВ с проверкой размерностей
            if not hasattr(self.centroids, 'is_meta') or not self.centroids.is_meta:
                if 'centroids' in state:
                    centroid_data = state['centroids'].to(self.device)
                    if centroid_data.shape == self.centroids.shape:
                        self.centroids.data.copy_(centroid_data)
                    else:
                        logger.warning(f"⚠ Несовпадение размеров centroids: загружаемый {centroid_data.shape} != {self.centroids.shape}")
                
                if 'transition_matrix' in state:
                    trans_data = state['transition_matrix'].to(self.device)
                    if trans_data.shape == self.transition_matrix.shape:
                        self.transition_matrix.data.copy_(trans_data)
                    else:
                        logger.warning(f"⚠ Несовпадение размеров transition_matrix")
                
                # Буферы с проверкой
                buffer_keys = ['slot_utility', 'slot_hit_count', 'slot_write_count', 
                              'slot_variance', 'slot_age', 'slot_frozen', 
                              'slot_active', 'momentum', 'slot_curiosity']
                
                for key in buffer_keys:
                    if key in state:
                        buffer_data = state[key].to(self.device)
                        if buffer_data.shape == getattr(self, key).shape:
                            getattr(self, key).copy_(buffer_data)
                        else:
                            logger.warning(f"⚠ Несовпадение размеров {key}")
                
                if 'slot_embeddings' in state:
                    try:
                        self.slot_embeddings.load_state_dict(state['slot_embeddings'])
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка загрузки slot_embeddings: {e}")
            
            # Загружаем membership
            if 'membership' in state:
                self.membership = [list(m) for m in state['membership']]
            
            # Загружаем статистику
            if 'stats' in state:
                self._stats = state['stats'].copy()
            
            # Загружаем счётчики
            if 'step_count' in state:
                self.step_count = state['step_count']
            if 'activation_history' in state:
                self._activation_history = state['activation_history'].copy()
            if 'transition_history' in state:
                self.transition_history = deque(state['transition_history'], maxlen=2000)
            else:
                self.transition_history = deque(maxlen=2000)
            
            # 🔥 Загружаем атрибуты Linker
            if 'linker_updates' in state:
                self.linker_updates = state['linker_updates']
            else:
                self.linker_updates = 0
                
            if 'linker_losses' in state:
                self.linker_losses = state['linker_losses'].copy()
            else:
                self.linker_losses = []
            
            if '_transition_similarities' in state:
                self._transition_similarities = deque(state['_transition_similarities'], maxlen=2000)
            else:
                self._transition_similarities = deque(maxlen=2000)
            
            if '_cache_dirty' in state:
                self._cache_dirty = state['_cache_dirty']
            else:
                self._cache_dirty = True
            
            # 🔥 Загружаем параметры Linker
            if 'enable_linker' in state:
                self.enable_linker = state['enable_linker']
            if 'linker_lr' in state:
                self.linker_lr = state['linker_lr']
            if 'linker_weight_decay' in state:
                self.linker_weight_decay = state['linker_weight_decay']
            if 'linker_batch_size' in state:
                self.linker_batch_size = state['linker_batch_size']
            if 'linker_train_frequency' in state:
                self.linker_train_frequency = state['linker_train_frequency']
            
        
        # ===========================================================
        # 🔥 ЗАГРУЗКА LINKER И ОПТИМИЗАТОРА
        # ===========================================================
        if self.enable_linker and 'linker_state' in state:
            try:
                # Загружаем веса Linker
                self._ensure_linker_initialized()  # гарантирует существование linker
                
                linker_state = state['linker_state']
                # Перемещаем на правильное устройство
                for key in linker_state:
                    if isinstance(linker_state[key], torch.Tensor):
                        linker_state[key] = linker_state[key].to(self.device)
                
                self.linker.load_state_dict(linker_state, strict=False)
                self.linker = self.linker.to(self.device)
                self.linker.eval()
                
                
                # ===========================================================
                # 🔥🔥🔥 ВАЖНО: СНАЧАЛА создаём оптимизатор, если его нет
                # ===========================================================
                if self.linker_optimizer is None:
                    logger.info("⚠️ Оптимизатор не найден, создаём перед загрузкой")
                    self.linker_optimizer = torch.optim.AdamW(
                        self.linker.parameters(),
                        lr=self.linker_lr,
                        weight_decay=self.linker_weight_decay
                    )
                    self.linker_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.linker_optimizer,
                        mode='min',
                        factor=0.5,
                        patience=20,
                        min_lr=1e-5
                    )
                
                # Загружаем оптимизатор, если он есть в state
                if 'linker_optimizer' in state:
                    try:
                        self.linker_optimizer.load_state_dict(state['linker_optimizer'])
                        
                        # После загрузки перемещаем все тензоры состояний на текущее устройство
                        for param_group in self.linker_optimizer.param_groups:
                            for p in param_group['params']:
                                if p in self.linker_optimizer.state:
                                    for k, v in self.linker_optimizer.state[p].items():
                                        if isinstance(v, torch.Tensor):
                                            self.linker_optimizer.state[p][k] = v.to(self.device)
                        
                        # Загружаем scheduler, если есть
                        if 'linker_scheduler' in state and self.linker_scheduler is not None:
                            try:
                                self.linker_scheduler.load_state_dict(state['linker_scheduler'])
                                logger.info(f"✅ Состояние scheduler'а восстановлено")
                            except Exception as e:
                                logger.warning(f"⚠ Не удалось загрузить scheduler: {e}")
                        
                        logger.info(f"✅ Состояние оптимизатора Linker восстановлено на {self.device}")
                        
                    except Exception as e:
                        logger.warning(f"⚠ Ошибка загрузки linker_optimizer: {e}")
                        # Если не удалось, оставляем созданный оптимизатор
                        logger.info(f"🔄 Использую свежесозданный оптимизатор")
                
                logger.info(f"🧠 Linker веса загружены: обновлений={self.linker_updates}, "
                           f"история={len(self.transition_history)}, "
                           f"losses={len(self.linker_losses)}")
                           
            except Exception as e:
                logger.warning(f"⚠ Ошибка загрузки linker: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # ===========================================================
        # 🔥 ДОБАВЛЕННАЯ ПРОВЕРКА: консистентность устройств
        # ===========================================================
        self._ensure_device_consistency()
        
        # Обновляем кэш
        self._cache_dirty = True
        self._update_cache()
        
        # Логируем результат
        if self.slot_active.any():
            active_utility = self.slot_utility[self.slot_active]
            logger.info(f"📊 После загрузки: активных={self.slot_active.sum().item()}, "
                       f"avg_utility={active_utility.mean().item():.4f}, "
                       f"linker_updates={self.linker_updates}")

    def _ensure_linker_initialized(self):
        """
        Гарантирует, что Neural Linker инициализирован
        """
        if not self.enable_linker:
            return
        
        if self.linker is None:
            logger.info("🧠 Linker инициализируется при загрузке...")
            linker_hidden = getattr(self, 'linker_hidden_dim', None) or (self.slot_dim // 2)
            
            self.linker = NeuralLinker(self.slot_dim, self.num_slots, hidden_dim=linker_hidden)
            self.linker = self.linker.to(self.device)
            
            self.linker_optimizer = torch.optim.AdamW(
                self.linker.parameters(), 
                lr=self.linker_lr, 
                weight_decay=self.linker_weight_decay
            )
            self.linker.eval()
            logger.info(f"   ✅ Linker создан: {self.slot_dim} → {linker_hidden} → {self.num_slots}")
        
        if not hasattr(self, 'linker_updates'):
            self.linker_updates = 0
        if not hasattr(self, 'linker_losses'):
            self.linker_losses = []
        if not hasattr(self, 'transition_history'):
            self.transition_history = deque(maxlen=2000)

    def get_slot_count(self):
        return self.num_slots

    def resize_slots(self, new_num_slots):
        if new_num_slots == self.num_slots:
            return
        logger.info(f"📏 Изменение размера центроидной памяти: {self.num_slots} → {new_num_slots}")

    def check_device_consistency(self):
        """🩺 Проверяет, все ли тензоры на одном устройстве"""
        devices = {
            'centroids': self.centroids.device,
            'transition_matrix': self.transition_matrix.device,
            'slot_utility': self.slot_utility.device,
            'slot_hit_count': self.slot_hit_count.device,
            'slot_active': self.slot_active.device,
            'momentum': self.momentum.device,
        }
        
        main_device = self.centroids.device
        all_ok = True
        
        for name, dev in devices.items():
            if dev != main_device:
                logger.warning(f"⚠ Device mismatch: {name} on {dev}, main on {main_device}")
                all_ok = False
        
        if all_ok:
            logger.info(f"✅ Все тензоры памяти на {main_device}")
        
        return all_ok

    def validate_cache_shape(self):
        """
        🔧  Валидирует и исправляет форму кэша если нужно
        """
        expected_dim = self.slot_dim
        
        if not hasattr(self, '_cached_active_centroids') or self._cached_active_centroids is None:
            self._cached_active_centroids = torch.zeros(0, expected_dim, device=self.device)
            self._cache_dirty = True
        elif self._cached_active_centroids.dim() != 2 or self._cached_active_centroids.shape[1] != expected_dim:
            logger.warning(f"⚠ Неправильная форма кэша: {self._cached_active_centroids.shape}, ожидалось [*, {expected_dim}]")
            self._cached_active_centroids = torch.zeros(0, expected_dim, device=self.device)
            self._cache_dirty = True
        
        if not hasattr(self, '_cached_active_indices') or self._cached_active_indices is None:
            self._cached_active_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            self._cache_dirty = True
        
        if self._cache_dirty:
            self._update_cache()
        
        # 🔧  кэш валиден, если не грязный
        return not self._cache_dirty
        
    def reset_linker_optimizer(self, device=None):
        """Пересоздаёт оптимизатор linker'а на правильном устройстве"""
        if not self.enable_linker or self.linker is None:
            return
        
        if device is None:
            device = self.centroids.device
        
        # 🔥 ВАЖНО: сначала перемещаем linker на правильное устройство
        if self.linker is not None:
            linker_device = next(self.linker.parameters()).device
            if linker_device != device:
                logger.info(f"🔄 Перемещаю linker с {linker_device} на {device}")
                self.linker = self.linker.to(device)
        
        logger.info(f"🔄 Создание оптимизатора linker'а на {device}")
        
        # Сохраняем старые параметры если были
        old_lr = self.linker_lr
        old_wd = self.linker_weight_decay
        
        if self.linker_optimizer is not None:
            # Пробуем сохранить learning rate из старого оптимизатора
            for group in self.linker_optimizer.param_groups:
                old_lr = group.get('lr', old_lr)
                old_wd = group.get('weight_decay', old_wd)
                break
        
        # Создаём новый оптимизатор
        self.linker_optimizer = torch.optim.AdamW(
            self.linker.parameters(),
            lr=old_lr,
            weight_decay=old_wd
        )
        
        logger.info(f"✅ Оптимизатор linker'а создан на {device} (lr={old_lr}, wd={old_wd})")
        
