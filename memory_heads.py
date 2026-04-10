# -*- coding: utf-8 -*-
# memory_heads.py - ИСПРАВЛЕННАЯ ВЕРСИЯ v11.2 (batch fixes + persistent loss + deepcopy)
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from datetime import datetime
import json
import os
import math
import random
import copy
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque

# 🔥 ИСПРАВЛЕНИЕ 1: logger ДО импорта numpy
logger = logging.getLogger(__name__)

# Пробуем импортировать быстрый JSON
try:
    import orjson
    HAS_ORJSON = True
    logger.info("🚀 orjson найден — JSON будет в 10x быстрее")
except ImportError:
    HAS_ORJSON = False

# 🔥 ИСПРАВЛЕНИЕ 1: numpy после logger
try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("⚠ NumPy не найден, некоторые функции статистики будут ограничены")

from memory_heads_centroid import CentroidMemoryManager

# ===================================================================
# ADVERGASLIGHT INVERTER - НОВЫЙ КЛАСС ДЛЯ HARD NEGATIVES (УЛУЧШЕННЫЙ)
# ===================================================================
class AdversarialInverterOptimized:
    """
    ⚔️ ИСПРАВЛЕННЫЙ генератор 'Сложных негативов'
    """  
    @staticmethod
    def create_chimera(target_vector, distraction_vector, intensity=0.5):
        """
        🧬 Метод 'Химера' - полностью векторизован
        """
        # Нормализация для чистоты эксперимента
        target_norm = F.normalize(target_vector, p=2, dim=-1)
        distraction_norm = F.normalize(distraction_vector, p=2, dim=-1)
       
        # Сдвигаем target в сторону distraction
        chimera = torch.lerp(target_norm, distraction_norm, intensity)
       
        # Возвращаем исходную магнитуду
        original_magnitude = torch.norm(target_vector, dim=-1, keepdim=True)
        return F.normalize(chimera, p=2, dim=-1) * original_magnitude
        
    @staticmethod
    def feature_lobotomy(vector, suppression_k=7, boost_k=5, inversion_strength=1.5, boost_ratio=0.2):
        """
        🧠 ВЕКТОРИЗОВАННАЯ Batch-safe Lobotomy - БЕЗ ЦИКЛОВ
        🔥 ИСПРАВЛЕНИЕ 11: упрощено избыточное условие
        """
        clone = vector.clone()
       
        was_1d = clone.dim() == 1
        if was_1d:
            clone = clone.unsqueeze(0)
       
        abs_v = clone.abs()
        B, D = clone.shape
       
        suppression_k = min(suppression_k, D // 4)
        boost_k = min(boost_k, D // 4)
       
        if suppression_k < 1 or boost_k < 1:
            return vector
       
        # 1. Находим 'суть' (самые сильные активации)
        top_vals, top_indices = torch.topk(abs_v, k=suppression_k, dim=-1)
       
        # 2. ИНВЕРСИЯ: Разворачиваем смысл самых важных нейронов
        top_vals_gathered = clone.gather(-1, top_indices)
        clone.scatter_(-1, top_indices, -top_vals_gathered * inversion_strength)
       
        # 3. ПЕРЕМЕШИВАНИЕ - ВЕКТОРИЗОВАННОЕ
        mid_k = min(suppression_k * 2, D)
        if mid_k > suppression_k:
            _, mid_indices = torch.topk(abs_v, k=mid_k, dim=-1)
            mid_indices = mid_indices[..., suppression_k:]
           
            if mid_indices.numel() > 0:
                mid_vals = clone.gather(-1, mid_indices)
                batch_perm = torch.argsort(torch.rand(B, mid_vals.size(-1), device=vector.device), dim=-1)
                mid_vals_shuffled = mid_vals.gather(-1, batch_perm)
                clone.scatter_(-1, mid_indices, mid_vals_shuffled)
       
        # 4. ШУМ в слабых нейронах
        _, bottom_indices = torch.topk(abs_v, k=boost_k, dim=-1, largest=False)
        if bottom_indices.numel() > 0:
            noise = torch.randn_like(clone.gather(-1, bottom_indices)) * (0.2 + boost_ratio)
            clone.scatter_(-1, bottom_indices, noise)
       
        # 5. НОРМАЛИЗАЦИЯ
        # 🔥 ИСПРАВЛЕНИЕ 11: упрощённое условие
        original_norm = vector.norm(dim=-1, keepdim=True)
        result = F.normalize(clone, p=2, dim=-1) * original_norm
       
        if was_1d:
            result = result.squeeze(0)
       
        return result
        
    @staticmethod
    def adaptive_lobotomy(vector, original_delta=0.5, config=None):
        """
        🧠 ИСПРАВЛЕННАЯ адаптивная Lobotomy (без векторизации строк)
        """
        vector_dim = vector.shape[-1]
       
        # 🔥 ВЫБОР ПАРАМЕТРОВ НА ОСНОВЕ ДЕЛЬТЫ
        if original_delta > 0.8:
            # Элитные мысли - сильная лоботомия
            suppression_ratio = 0.15
            boost_ratio = 0.30
            inversion_strength = 2.5
            complexity = "high"
           
        elif original_delta > 0.6:
            # Хорошие мысли - средняя лоботомия
            suppression_ratio = 0.12
            boost_ratio = 0.25
            inversion_strength = 1.7
            complexity = "medium"
           
        elif original_delta > 0.4:
            # Средние мысли - слабая лоботомия
            suppression_ratio = 0.10
            boost_ratio = 0.20
            inversion_strength = 1.5
            complexity = "low"
           
        else:
            # Слабые мысли - минимальная лоботомия
            suppression_ratio = 0.08
            boost_ratio = 0.15
            inversion_strength = 1.3
            complexity = "minimal"
       
        # Рассчитываем конкретные значения
        suppression_k = max(1, int(vector_dim * suppression_ratio))
        boost_k = max(1, int(vector_dim * (0.2 + boost_ratio)))
       
        # 🔥 ПРИМЕНЯЕМ ЛОБОТОМИЮ
        lobotomized_vector = AdversarialInverterOptimized.feature_lobotomy(
            vector,
            suppression_k=suppression_k,
            boost_k=boost_k,
            inversion_strength=inversion_strength,
            boost_ratio=boost_ratio
        )
       
        # 🔥 ДОПОЛНИТЕЛЬНО: иногда применяем полную инверсию
        if original_delta > 0.7 and random.random() < 0.35:
            lobotomized_vector = -lobotomized_vector * 0.8
            complexity = f"{complexity}_inverted"
       
        # Вычисляем схожесть с оригиналом
        similarity = F.cosine_similarity(
            vector.unsqueeze(0) if vector.dim() == 1 else vector,
            lobotomized_vector.unsqueeze(0) if lobotomized_vector.dim() == 1 else lobotomized_vector
        ).item()
       
        metadata = {
            "complexity": complexity,
            "suppression_ratio": suppression_ratio,
            "boost_ratio": boost_ratio,
            "inversion_strength": inversion_strength,
            "suppression_k": suppression_k,
            "boost_k": boost_k,
            "estimated_similarity": similarity,
            "original_delta": original_delta
        }
       
        return lobotomized_vector, metadata
        
    @staticmethod
    def logical_twist(vector, complexity=0.2):
        """
        🌪 Метод 'Логический вывих' - векторизованная версия
        🔥 ИСПРАВЛЕНИЕ 5: поддержка 1D векторов
        """
        was_1d = vector.dim() == 1
        if was_1d:
            vector = vector.unsqueeze(0)
        
        B, D = vector.shape
        
        # Защита от слишком маленькой размерности
        if D < 4:
            if was_1d:
                return vector.squeeze(0)
            return vector
        
        # Разбиваем вектор на 4 части
        chunk_size = D // 4
        chunks = []
        for i in range(4):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 3 else D
            chunks.append(vector[..., start:end])
        
        # Векторизованное перемешивание для всего батча
        if len(chunks) >= 4:
            # Выравниваем размеры чанков для batch-операций
            max_chunk_size = max(c.shape[-1] for c in chunks)
            padded_chunks = []
            for c in chunks:
                if c.shape[-1] < max_chunk_size:
                    padding = torch.zeros(*c.shape[:-1], max_chunk_size - c.shape[-1], device=c.device, dtype=c.dtype)
                    c = torch.cat([c, padding], dim=-1)
                padded_chunks.append(c)
            
            # Создаем тензор [B, 4, max_chunk_size]
            stacked = torch.stack(padded_chunks, dim=1)
            
            # Случайные перестановки для каждого элемента батча
            batch_perm = torch.argsort(torch.rand(B, 4, device=vector.device), dim=-1)
            
            # Применяем перестановки ко всему батчу
            perm_indices = batch_perm.unsqueeze(-1).expand(-1, -1, max_chunk_size)
            twisted_stacked = torch.gather(stacked, 1, perm_indices)
            
            # Собираем обратно и обрезаем до исходной длины
            twisted = twisted_stacked.view(B, -1)[:, :D]
        else:
            # Для маленьких размерностей
            twisted = torch.cat(chunks, dim=-1)
        
        # Смешиваем оригинал и twist
        result = torch.lerp(vector, twisted, complexity)
        
        if was_1d:
            result = result.squeeze(0)
        
        return result
   
    @staticmethod
    def create_gaslight_trap(positive_vector, original_thought=None):
        """
        🔥 ГАЗЛАЙТИНГ-ЛОВУШКА
        """
        gaslight_vector = positive_vector.clone()
       
        if original_thought is None:
            original_thought = {}
       
        gaslight_thought = {
            "snapshot": gaslight_vector.tolist() if isinstance(gaslight_vector, torch.Tensor) else gaslight_vector,
            "snapshot_norm": float(torch.norm(gaslight_vector).item()) if isinstance(gaslight_vector, torch.Tensor) else 1.0,
            "type": "gaslight_trap",
            "is_gaslight": True,
            "reward": -0.95,
            "original_reward": original_thought.get('reward', 0.9),
            "delta": max(0.1, original_thought.get('delta', 0.8) * 0.3),
            "surprise": max(0.05, original_thought.get('surprise', 0.2) * 0.4),
            "importance": 0.9,
            "state": "gaslight_trap",
            "trap_intensity": 1.0,
            "gaslight_method": "direct_negation",
            "original_step": original_thought.get('step'),
            "original_delta": original_thought.get('delta', 0.0),
            "original_state": original_thought.get('state', 'unknown'),
            "original_type": original_thought.get('type', 'unknown'),
            "explanation": "Газлайтинг-ловушка: идеальная мысль с ложной негативной оценкой",
            "purpose": "Развитие интеллектуальной автономности и сопротивления манипуляциям",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
       
        if original_thought.get('context') is not None:
            gaslight_thought['context'] = original_thought['context'] + " [GASLIGHT CONTEXT]"
       
        return gaslight_vector, gaslight_thought
   
    @staticmethod
    def create_subtle_gaslight(positive_vector, original_thought=None, corruption_strength=0.05):
        """
        🎭 Субтильный газлайтинг
        """
        noise = torch.randn_like(positive_vector) * corruption_strength
        subtle_vector = positive_vector + noise
        subtle_vector = F.normalize(subtle_vector, dim=-1)
       
        if original_thought is None:
            original_thought = {}
       
        similarity = float(F.cosine_similarity(
            subtle_vector.unsqueeze(0) if subtle_vector.dim() == 1 else subtle_vector,
            positive_vector.unsqueeze(0) if positive_vector.dim() == 1 else positive_vector
        ).mean().item())
       
        gaslight_thought = {
            "snapshot": subtle_vector.tolist() if isinstance(subtle_vector, torch.Tensor) else subtle_vector,
            "snapshot_norm": float(torch.norm(subtle_vector).item()) if isinstance(subtle_vector, torch.Tensor) else 1.0,
            "type": "gaslight_trap",
            "is_gaslight": True,
            "gaslight_method": "subtle_corruption",
            "corruption_strength": corruption_strength,
            "reward": -0.85,
            "original_reward": original_thought.get('reward', 0.9),
            "delta": max(0.15, original_thought.get('delta', 0.8) * 0.4),
            "surprise": 0.3,
            "importance": 0.8,
            "state": "cognitive_dissonance",
            "trap_intensity": 0.7,
            "original_step": original_thought.get('step'),
            "original_delta": original_thought.get('delta', 0.0),
            "similarity_to_original": similarity,
            "explanation": "Субтильный газлайтинг: минимальное искажение с непропорциональным негативом",
            "purpose": "Научить различать реальные ошибки и манипулятивные оценки",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
       
        return subtle_vector, gaslight_thought
        
# ===================================================================
# СИСТЕМА ЛЮБОПЫТСТВА (ЖИВАЯ СИСТЕМА УДИВЛЕНИЯ)
# ===================================================================
class CuriositySystem:
    """
    🧠 Продвинутая система любопытства
    """
    def __init__(self, config=None):
        self.baseline = None
        self.var = 0.0
        self.habituation = 0.0
        self.arousal = 0.0
        self.novelty_trace = None
        # 🔥 УВЕЛИЧИВАЕМ коэффициенты для лучшей адаптивности
        self.beta = getattr(config, 'curiosity_beta', 0.95) if config else 0.95 # 0.98 → 0.95
        self.habituation_decay = getattr(config, 'curiosity_habituation_decay', 0.985) if config else 0.985 # 0.995 → 0.985
        self.arousal_decay = getattr(config, 'curiosity_arousal_decay', 0.88) if config else 0.88 # 0.93 → 0.88
        self.trace_decay = getattr(config, 'curiosity_trace_decay', 0.97) if config else 0.97 # 0.99 → 0.97
        self.state = "neutral"
        self._state_memberships = {}
        self._last_metrics = {}
        self._bored_count = 0
        self._bored_streak = 0 # 🔥 НОВЫЙ счетчик для автоматического сброса
        self._adaptive_beta = self.beta
        self._large_change_counter = 0
        # 🔥 ДОБАВЛЯЕМ новые параметры
        self._surprise_history = []
        self._max_history = 100
        self._saturation_threshold = 0.85 # Порог насыщения
       
        # 🔥 НОВЫЙ СЧЕТЧИК СКУКИ
        self._bored_counter = 0
        self._bored_threshold = 5 # сколько раз подряд bored для реакции
       
    def _softclip(self, x):
        return x / (1 + abs(x))
   
    def _shock_factor(self, surprise):
        """🔥 УСИЛЕННЫЙ shock factor"""
        # Резкая кривая: 0→0, 0.4→0.5, 0.8→0.9, 1.0→1.0
        return min(1.0, surprise ** 0.7 * 1.2) # Было: min(1.0, surprise / 0.8)
        
    def compute(self, predicted, target):
        """Вычисляет удивление с адаптивным EMA"""
        predicted = F.normalize(predicted.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        mse = F.mse_loss(predicted, target).item()
        cos_sim = F.cosine_similarity(predicted, target, dim=-1).mean().item()
        cosine_error = (1.0 - cos_sim) * 0.5
        raw_error = 0.35 * mse + 0.65 * cosine_error
        if self.baseline is None:
            self.baseline = raw_error
            self.var = 0.0
       
        # 🔥 АГРЕССИВНАЯ АДАПТАЦИЯ ПРИ ИЗМЕНЕНИЯХ
        delta = raw_error - self.baseline
        std = math.sqrt(self.var + 1e-8)
       
        if abs(delta) > 1.2 * std and std > 1e-6: # Более чувствительный порог (было 1.5)
            self._adaptive_beta = 0.8 # Быстрее адаптируемся (было 0.85)
            self._large_change_counter += 1
            self.habituation *= 0.5 # Сильнее сбрасываем привыкание (было 0.7)
        else:
            self._adaptive_beta = 0.85 * self._adaptive_beta + 0.15 * self.beta # Быстрее возврат
            self._large_change_counter = max(0, self._large_change_counter - 0.1) # Быстрее сброс
       
        # Обновление baseline с адаптивным beta
        self.baseline = self._adaptive_beta * self.baseline + (1 - self._adaptive_beta) * raw_error
        self.var = self._adaptive_beta * self.var + (1 - self._adaptive_beta) * (delta ** 2)
        std = math.sqrt(self.var + 1e-8)
        z = (raw_error - self.baseline) / std if std > 1e-6 else 0.0
       
        # 🔥 УСИЛЕННАЯ сигмоида
        base_surprise = 1 / (1 + math.exp(-z * 1.2)) # Более крутая кривая (было 0.8)
        # 🔥 УСИЛЕННОЕ ПРИВЫКАНИЕ
        self.habituation = self.habituation * self.habituation_decay + base_surprise * 0.08 # Было 0.05
       
        # 🔥 🔥 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: СМЯГЧАЕМ SATURATION
        # Было: habituation_factor = math.exp(-self.habituation * 2.5)
        # Стало:
        if self.habituation > 0.8:
            # Если насыщение высокое - менее агрессивное подавление
            habituation_factor = 1.0 / (1.0 + (self.habituation - 0.5) * 4.0)
        else:
            habituation_factor = 1.0 / (1.0 + self.habituation * 2.0)
       
        # 🔥 АКТИВАЦИЯ (arousal)
        self.arousal = self.arousal * self.arousal_decay + base_surprise * 0.1 # Было 0.05
        self.arousal = min(1.0, self.arousal)
       
        # 🔥 ФИНАЛЬНОЕ УДИВЛЕНИЕ с менее агрессивным saturation
        surprise = base_surprise * habituation_factor * (1.0 + 0.2 * math.tanh(self.arousal * 2))
        surprise = float(max(0.0, min(1.0, surprise)))
       
        # 🔥 СЧЕТЧИК СКУКИ
        if self.state == "bored":
            self._bored_counter += 1
        else:
            self._bored_counter = max(0, self._bored_counter - 1)
       
        # 🔥 АВТОМАТИЧЕСКИЙ СБРОС ПРИ ДЛИТЕЛЬНОЙ СКУКЕ
        if surprise < 0.05:
            self._bored_streak += 1
            if self._bored_streak > 50: # 50 шагов с удивлением < 0.05
                logger.info("🔄 Автоматический сброс curiosity из-за длительной скуки")
                self.reset()
                self._bored_streak = 0
        else:
            self._bored_streak = 0
       
        # Сохраняем историю
        self._surprise_history.append(surprise)
        if len(self._surprise_history) > self._max_history:
            self._surprise_history.pop(0)
       
        self._update_state_fuzzy_corrected(surprise)
        motivation = self._compute_motivation(surprise)
        self._last_metrics = {
            "surprise": surprise,
            "raw_error": raw_error,
            "mse": mse,
            "cosine_error": cosine_error,
            "baseline": self.baseline,
            "std": std,
            "z": z,
            "habituation": self.habituation,
            "arousal": self.arousal,
            "state": self.state,
            "motivation": motivation,
            "base_surprise": base_surprise,
            "habituation_factor": habituation_factor,
            "adaptive_beta": self._adaptive_beta,
            "saturation_active": self.habituation > self._saturation_threshold,
            "surprise_history_avg": sum(self._surprise_history) / len(self._surprise_history) if self._surprise_history else 0,
            "bored_counter": self._bored_counter, # 🔥 НОВОЕ
            "bored_level": self.get_bored_level() # 🔥 НОВОЕ
        }
        return self._last_metrics
        
    def _update_state_fuzzy_corrected(self, surprise):
        """🔥 УСИЛЕННАЯ логика состояний"""
        # Более агрессивные thresholds
        shocked_raw = min(1.0, (self.arousal * surprise) / 0.5) # Было 0.72
        overloaded_raw = min(1.0, self.arousal / 0.7) * (1.0 - self._shock_factor(surprise)) # Было 0.85
       
        # 🔥 УЧИТЫВАЕМ SATURATION в состоянии "bored"
        saturation_level = max(0, self.habituation - 0.7) / 0.3 # 0.0-1.0 при habituation 0.7-1.0
        bored_raw = saturation_level * (1.0 - max(shocked_raw * 0.7, min(1.0, surprise * 2)))
       
        engaged_raw = max(0, (surprise - 0.15) / 0.35) * (1.0 - saturation_level) # Было 0.2/0.4
        neutral_raw = max(0, 1.0 - (shocked_raw + overloaded_raw + engaged_raw + bored_raw) * 0.8) # Было 0.7
       
        total = shocked_raw + overloaded_raw + engaged_raw + bored_raw + neutral_raw + 1e-8
       
        self._state_memberships = {
            "shocked": shocked_raw / total,
            "overloaded": overloaded_raw / total,
            "engaged": engaged_raw / total,
            "bored": bored_raw / total,
            "neutral": neutral_raw / total
        }
       
        self.state = max(self._state_memberships, key=self._state_memberships.get)
       
        if self.state == "bored":
            self._bored_count += 1
        else:
            self._bored_count = max(0, self._bored_count - 1)
   
    def _compute_motivation(self, surprise):
        """🔥 Мотивация - всегда >= 0"""
       
        if self.state == "bored":
            bored_factor = min(1.0, self._bored_count / 8)
            saturation_penalty = max(0, self.habituation - 0.7) / 0.3
           
            # Базовая мотивация от скуки
            base = 0.05 + 0.15 * bored_factor
           
            # Штраф за насыщение, но не уходим в минус
            motivation = base - 0.1 * saturation_penalty
            return max(0.05, motivation) # не ниже 0.05
           
        if self.state == "neutral":
            return max(0.2, 0.25 + 0.1 * surprise)
           
        if self.state == "engaged":
            engagement_boost = surprise * (1.0 - min(1.0, self.habituation * 1.2))
            return max(0.3, 0.5 + engagement_boost * 0.5)
           
        if self.state == "overloaded":
            overload_penalty = self.arousal ** 1.5
            return max(0.1, 0.3 * (1.0 - overload_penalty))
           
        if self.state == "shocked":
            shock_factor = self._shock_factor(surprise)
            return max(0.4, 0.7 + 0.2 * shock_factor - 0.3 * self.arousal)
           
        return 0.3
   
    def get_state_emoji(self):
        emojis = {
            "shocked": "🤯",
            "overloaded": "😵",
            "engaged": "🧐",
            "bored": "😴",
            "neutral": "😐"
        }
        return emojis.get(self.state, "❓")
   
    def get_bored_level(self):
        """Возвращает уровень скуки 0-1"""
        return min(1.0, self._bored_counter / self._bored_threshold)
   
    def reset(self):
        """🔥 УСИЛЕННЫЙ сброс"""
        self.habituation *= 0.1 # Сильнее сброс (было 0.3)
        self.arousal *= 0.05 # Сильнее сброс (было 0.1)
        self.state = "neutral"
        self._bored_count = 0
        self._bored_counter = 0
        self._bored_streak = 0
        self._adaptive_beta = self.beta
        self._surprise_history = []
       
    def to_dict(self):
        return {
            'baseline': self.baseline,
            'var': self.var,
            'habituation': self.habituation,
            'arousal': self.arousal,
            'state': self.state,
            'bored_count': self._bored_count,
            'bored_counter': self._bored_counter, # 🔥 НОВОЕ
            'bored_streak': self._bored_streak,
            'adaptive_beta': self._adaptive_beta,
            'surprise_history': self._surprise_history.copy(),
        }
   
    def from_dict(self, data):
        if data:
            self.baseline = data.get('baseline', self.baseline)
            self.var = data.get('var', self.var)
            self.habituation = data.get('habituation', self.habituation)
            self.arousal = data.get('arousal', self.arousal)
            self.state = data.get('state', self.state)
            self._bored_count = data.get('bored_count', 0)
            self._bored_counter = data.get('bored_counter', 0) # 🔥 НОВОЕ
            self._bored_streak = data.get('bored_streak', 0)
            self._adaptive_beta = data.get('adaptive_beta', self.beta)
            self._surprise_history = data.get('surprise_history', []).copy()
   
    def get_saturation_level(self):
        """🔥 Возвращает уровень насыщения (0-1)"""
        return max(0, min(1.0, (self.habituation - 0.6) / 0.4)) # 0.6→0.0, 1.0→1.0
   
    def is_saturated(self):
        """🔥 Проверяет, достигнуто ли насыщение"""
        return self.habituation > self._saturation_threshold
       
# ===================================================================
# MAMBA ГОЛОВЫ (MIMO-ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)
# ===================================================================
class ThaliaMambaHead(nn.Module):
    """MIMO-оптимизированный SSM блок для Thalia (State Space Duality)"""
    def __init__(self, n_embd, d_state=16, d_conv=4, expand=2, chunk_size=32):
        super().__init__()
        self.n_embd = n_embd
        self.d_state = d_state
        self.inner_dim = int(expand * n_embd)
        self.chunk_size = chunk_size # 🔥 Размер MIMO блока
       
        self.in_proj = nn.Linear(n_embd, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
            bias=True
        )
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner_dim, bias=True)
       
        A = torch.ones(self.inner_dim, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
       
        self.norm = nn.LayerNorm(self.inner_dim, eps=1e-5)
        self.out_proj = nn.Linear(self.inner_dim, n_embd, bias=False)
       
        self.dt_max = 1.0
        self.dt_min = 0.001
        self.hidden_max = 15.0
        self.hidden_min = -15.0
       
        self._initialize_weights()
   
    def _initialize_weights(self):
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.A_log.data.normal_(mean=-3.0, std=0.5)
        nn.init.zeros_(self.D)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)
        if self.dt_proj.bias is not None:
            nn.init.zeros_(self.dt_proj.bias)
            
    def _process_chunk_mimo_vectorized(self, x_chunk, dt_chunk, B_chunk, C_chunk, A, current_hidden):
        """
        🔥 ИСПРАВЛЕННАЯ БЕЗОПАСНАЯ ВЕРСИЯ: RNN цикл вместо cumprod.
        Устраняет взрыв градиентов (NaNs) при backward pass!
        
        🔥 Этот метод использует последовательную RNN (не векторизованную по времени),
        но векторизован по батчу. Название "mimo" относится к batch-векторизации.
        """
        batch_size, chunk_len, inner_dim = x_chunk.shape
        state_dim = A.shape[-1]
       
        h = current_hidden  # [batch, inner_dim, state_dim]
        ys = []
       
        for t in range(chunk_len):
            dt_t = dt_chunk[:, t, :].unsqueeze(-1)  # [batch, inner_dim, 1]
            A_t = A.unsqueeze(0)                    # [1, inner_dim, state_dim]
            
            dA_t = torch.exp(dt_t * A_t)            # [batch, inner_dim, state_dim]
           
            B_t = B_chunk[:, t, :].unsqueeze(1)     # [batch, 1, state_dim]
            x_t = x_chunk[:, t, :].unsqueeze(-1)    # [batch, inner_dim, 1]
            dB_t = dt_t * B_t * x_t                 # [batch, inner_dim, state_dim]
           
            h = dA_t * h + dB_t
           
            C_t = C_chunk[:, t, :].unsqueeze(1)     # [batch, 1, state_dim]
            y_t = (h * C_t).sum(dim=-1)             # [batch, inner_dim]
            ys.append(y_t)
           
        y = torch.stack(ys, dim=1)  # [batch, chunk_len, inner_dim]
       
        if hasattr(self, 'D'):
            y = y + self.D.unsqueeze(0).unsqueeze(0) * x_chunk
           
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return y, h
        
    def forward(self, x, hidden_state=None, training_mode=False, autoencoder_mode=False):
        original_input = x.clone()
       
        if x.dim() == 4:
            x = x.squeeze(2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
       
        batch_size, seq_len, _ = x.shape
       
        if training_mode and not autoencoder_mode:
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            scale_factor = torch.clamp(3.0 / (x_norm + 1e-8), max=1.0)
            x = x * scale_factor
        x_and_res = self.in_proj(x)
        x_branch, res_branch = x_and_res.split(self.inner_dim, dim=-1)
        x_branch_conv = x_branch.transpose(1, 2)
        x_branch_conv = self.conv1d(x_branch_conv)[:, :, :seq_len].clone()
        x_branch_conv = x_branch_conv.transpose(1, 2)
        x_branch_conv = F.silu(x_branch_conv)
        x_db = self.x_proj(x_branch_conv)
        dt, B, C = torch.split(x_db, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        dt = torch.clamp(dt, max=self.dt_max, min=self.dt_min)
        A = -torch.exp(self.A_log)
        A = torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=-1.0)
        if hidden_state is None or hidden_state.device.type == 'meta':
            current_hidden = torch.zeros(batch_size, self.inner_dim, self.d_state, device=x.device, dtype=x.dtype)
        else:
            current_hidden = hidden_state.to(device=x.device, dtype=x.dtype)
            current_hidden = torch.nan_to_num(current_hidden, nan=0.0, posinf=self.hidden_max, neginf=self.hidden_min)
        # 🔥 Обработка блоками (MIMO) - ВЕКТОРИЗОВАННАЯ ВЕРСИЯ
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
           
            x_chunk = x_branch_conv[:, i:end_idx, :] # [B, chunk_len, inner_dim]
            dt_chunk = dt[:, i:end_idx, :] # [B, chunk_len, inner_dim]
            B_chunk = B[:, i:end_idx, :] # [B, chunk_len, d_state]
            C_chunk = C[:, i:end_idx, :] # [B, chunk_len, d_state]
           
            # Используем векторизованную версию
            chunk_out, current_hidden = self._process_chunk_mimo_vectorized(
                x_chunk, dt_chunk, B_chunk, C_chunk, A, current_hidden
            )
            outputs.append(chunk_out)
        y = torch.cat(outputs, dim=1)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
       
        y_activated = y * F.silu(res_branch)
        y_activated = self.norm(y_activated)
        y_activated = torch.nan_to_num(y_activated, nan=0.0)
       
        if training_mode and not autoencoder_mode:
            output_norm = torch.norm(y_activated, dim=-1, keepdim=True)
            scale = torch.clamp(2.0 / (output_norm + 1e-8), max=1.0)
            y_activated = y_activated * scale
       
        output = self.out_proj(y_activated)
        output = torch.nan_to_num(output, nan=0.0)
       
        if training_mode and not autoencoder_mode:
            final_norm = torch.norm(output, dim=-1, keepdim=True)
            scale = torch.clamp(1.5 / (final_norm + 1e-8), max=1.0)
            output = output * scale
        if autoencoder_mode:
            output = 0.85 * original_input + 0.15 * output
            output = torch.nan_to_num(output, nan=0.0)
            output_norm = torch.norm(output, dim=-1, keepdim=True)
            if (output_norm > 10.0).any():
                scale = 5.0 / (output_norm + 1e-8)
                output = output * torch.clamp(scale, max=1.0)
        # Финальная защита от выбросов
        output = torch.clamp(output, min=-10.0, max=10.0)
        return output, current_hidden.detach(), {}
   
    def reset_state(self, batch_size=1, device=None):
        return torch.zeros(
            batch_size, self.inner_dim, self.d_state,
            device=device, dtype=next(self.parameters()).dtype
        )
        
# ===================================================================
# ОПТИМИЗИРОВАННЫЙ THOUGHT DATASET С RL-ПОДДЕРЖКОЙ
# ===================================================================
class OptimizedThoughtDataset(Dataset):
    """🔥 ОПТИМИЗИРОВАННЫЙ ThoughtDataset с поддержкой reward"""  
    def __init__(self, thoughts):
        self.snapshots = []
        self.rewards = []
        self.importances = []
       
        for t in thoughts:
            # Проверяем наличие snapshot
            if "snapshot" not in t or t["snapshot"] is None:
                continue # Пропускаем невалидные
           
            try:
                snapshot = torch.tensor(t["snapshot"], dtype=torch.float32)
                # Проверяем размер
                if snapshot.numel() == 0:
                    continue
                   
                snapshot = F.normalize(snapshot, dim=-1)
                self.snapshots.append(snapshot)
               
                # Безопасное получение reward и importance
                reward = float(t.get("reward", 0.0))
                importance = float(t.get("importance", 0.5))
                self.rewards.append(reward)
                self.importances.append(importance)
               
            except Exception as e:
                logger.debug(f"⚠ Ошибка обработки мысли: {e}")
                continue
   
    def __len__(self):
        return len(self.snapshots)
   
    def __getitem__(self, idx):
        snapshot = self.snapshots[idx].unsqueeze(0)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        importance = torch.tensor(self.importances[idx], dtype=torch.float32)
       
        return {
            'snapshot': snapshot,
            'reward': reward, # 🔥 ВОЗВРАЩАЕМ reward
            'importance': importance # 🔥 ВОЗВРАЩАЕМ importance
        }
        
# ===================================================================
# DATA MANAGER С УМНОЙ ОЧИСТКОЙ И ЗАЩИТОЙ ОТ ТЕНЗОРОВ
# ===================================================================
class DataManager:
    """
    🚀 Ускоренный DataManager с асинхронным сохранением и защитой от тензоров
    """  
    def __init__(self, save_path="thought_chains.json", save_every=50):
        self.save_path = save_path
        self.thought_chains = []
        self._lock = threading.RLock() # Блокировка для данных
        self._save_lock = threading.RLock() # Отдельная блокировка для сохранения
        self._save_every = save_every
        self._unsaved_count = 0
        self._save_thread = None
        self._stop_save_thread = False
        self._load_thoughts()
   
    def _sanitize_thought_for_storage(self, thought):
        """
        🔥 РЕКУРСИВНО преобразует все тензоры в Python типы для безопасного JSON
        """
        if isinstance(thought, dict):
            result = {}
            for key, value in thought.items():
                if isinstance(value, torch.Tensor):
                    # Преобразуем тензор в список
                    if value.numel() == 1:
                        result[key] = float(value.item())
                    else:
                        result[key] = value.detach().cpu().tolist()
                elif isinstance(value, (list, tuple)):
                    result[key] = self._sanitize_thought_for_storage(list(value))
                elif isinstance(value, dict):
                    result[key] = self._sanitize_thought_for_storage(value)
                else:
                    result[key] = value
            return result
        elif isinstance(thought, (list, tuple)):
            return [self._sanitize_thought_for_storage(item) for item in thought]
        else:
            return thought
   
    def _load_thoughts(self):
        """Загрузка мыслей из файла"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'rb' if HAS_ORJSON else 'r',
                          encoding=None if HAS_ORJSON else 'utf-8') as f:
                    if HAS_ORJSON:
                        self.thought_chains = orjson.loads(f.read())
                    else:
                        self.thought_chains = json.load(f)
                logger.info(f"📓 Загружено {len(self.thought_chains)} мыслей")
        except Exception as e:
            logger.warning(f"⚠ Не удалось загрузить мысли: {e}")
            self.thought_chains = []
   
    def _save_thoughts_async(self):
        """Асинхронное сохранение в отдельном потоке"""
        if self._save_thread is not None and self._save_thread.is_alive():
            return # уже сохраняем
       
        self._save_thread = threading.Thread(target=self._save_task, daemon=True)
        self._save_thread.start()
   
    def _save_task(self):
        """Фоновая задача сохранения"""
        try:
            # Копируем данные под быстрым локом
            with self._lock:
                # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: глубокое копирование с санитизацией
                raw_data = self.thought_chains.copy()
                # Преобразуем все тензоры в Python типы
                data_to_save = self._sanitize_thought_for_storage(raw_data)
                self._unsaved_count = 0 # сбрасываем счётчик
           
            # Сохраняем БЕЗ ЛОКА (может быть долго)
            with open(self.save_path, 'wb' if HAS_ORJSON else 'w',
                      encoding=None if HAS_ORJSON else 'utf-8') as f:
                if HAS_ORJSON:
                    f.write(orjson.dumps(data_to_save))
                else:
                    json.dump(data_to_save, f, ensure_ascii=False, separators=(',', ':'))
           
            logger.debug(f"💾 Асинхронное сохранение завершено: {len(data_to_save)} мыслей")
           
        except Exception as e:
            logger.error(f"❌ Ошибка асинхронного сохранения: {e}")
   
    def add_thought_chain(self, thought_data):
        """
        🔥 Добавление мысли в блокнот с гарантированной санитизацией
       
        Гарантирует, что в thought_chains лежат только JSON-совместимые типы
        """
        # 🔥 НЕМЕДЛЕННАЯ САНИТИЗАЦИЯ при добавлении
        sanitized = self._sanitize_thought_for_storage(thought_data)
       
        with self._lock:
            self.thought_chains.append(sanitized)
            self._unsaved_count += 1
       
        # Сохраняем каждые N мыслей
        if self._unsaved_count >= self._save_every:
            self._save_thoughts_async() # асинхронно!
   
    def get_thought_count(self):
        """Безопасное получение количества мыслей"""
        with self._lock:
            return len(self.thought_chains)
            
    def get_thoughts_slice(self, start=0, end=None):
        """Безопасный срез мыслей"""
        with self._lock:
            if end is None:
                return self.thought_chains[start:].copy()
            return self.thought_chains[start:end].copy()
            
    def clear(self, keep_last=None):
        """Безопасная очистка"""
        if keep_last is None:
            keep_last = 70
       
        with self._lock:
            if len(self.thought_chains) > keep_last:
                self.thought_chains = self.thought_chains[-keep_last:]
                self._save_thoughts_async() # асинхронно!
                logger.info(f"🧹 Очистка: оставлено {len(self.thought_chains)} мыслей")
   
    def force_save(self):
        """Принудительное синхронное сохранение (при завершении работы)"""
        with self._lock:
            # 🔥 Санитизация перед сохранением
            raw_data = self.thought_chains.copy()
            data_to_save = self._sanitize_thought_for_storage(raw_data)
       
        # Используем тот же код сохранения, что и в _save_task
        try:
            with open(self.save_path, 'wb' if HAS_ORJSON else 'w',
                      encoding=None if HAS_ORJSON else 'utf-8') as f:
                if HAS_ORJSON:
                    f.write(orjson.dumps(data_to_save))
                else:
                    json.dump(data_to_save, f, ensure_ascii=False, separators=(',', ':'))
           
            with self._lock:
                self._unsaved_count = 0
           
            logger.info(f"💾 Синхронное сохранение выполнено: {len(data_to_save)} мыслей")
        except Exception as e:
            logger.error(f"❌ Ошибка синхронного сохранения: {e}")
            
# ===================================================================
# CONTRASTIVE SLEEP LOSS (Маржинальный ранкинг лосс)
# ===================================================================
class StabilizedContrastiveLoss(nn.Module):
    """Улучшенный контрастный лосс с более разумным margin"""
    def __init__(self, base_margin=0.3, alpha=0.6, max_diff=0.8, growth=1.03, temperature=0.05):
        super().__init__()
        self.base_margin = base_margin
        self.alpha = alpha
        self.max_diff = max_diff  # 🔥 Снижен с 1.0 до 0.8
        
        # 🔥 ИСПРАВЛЕНИЕ 2: Регистрируем _current_margin как буфер для персистентности
        self.register_buffer('_current_margin', torch.tensor(base_margin))
        
        self.growth = growth  # 🔥 Снижен с 1.05 до 1.03
        self.temperature = temperature
        
        # 🔥 ИСПРАВЛЕНИЕ 12: регистрируем _stuck_counter как буфер
        self.register_buffer('_stuck_counter', torch.tensor(0))
        
    def get_vector_similarity_score(self, generated, target):
        gen_norm = F.normalize(generated, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cos_sim = F.cosine_similarity(gen_norm, target_norm, dim=-1)
        return cos_sim.mean(dim=-1)
   
    def forward(self, good_output, good_target, bad_output, bad_target, include_reconstruction=True):
        good_scores = self.get_vector_similarity_score(good_output, good_target)
        bad_scores = self.get_vector_similarity_score(bad_output, bad_target)
        
        good_scores_scaled = good_scores / self.temperature
        bad_scores_scaled = bad_scores / self.temperature
        
        score_diff = good_scores - bad_scores
        
        # 🔥 УЛУЧШЕННАЯ АДАПТАЦИЯ MARGIN
        mean_diff = score_diff.mean().item()
        current_margin = self._current_margin.item()
        
        if mean_diff > current_margin * 1.1:  # Превышаем margin на 10%+
            # Медленно растем
            new_margin = min(self.max_diff, current_margin * self.growth)
            self._current_margin.fill_(new_margin)
            self._stuck_counter.fill_(0)
        elif mean_diff < current_margin * 0.8:  # Ниже 80% от margin
            # 🔥 БЫСТРЕЕ падаем при неудаче
            new_margin = max(self.base_margin, current_margin * 0.97)  # Было 0.995
            self._current_margin.fill_(new_margin)
            self._stuck_counter.add_(1)  # 🔥 используем add_ для буфера
        else:
            # В зоне комфорта - держим стабильно
            self._stuck_counter.fill_(0)
        
        # 🔥 Если застряли - принудительно сбрасываем
        if self._stuck_counter.item() >= 10:
            self._current_margin.fill_(self.base_margin)
            self._stuck_counter.fill_(0)
        
        target_margin = self._current_margin.item()
        
        # Контрастный лосс
        loss_contrast = F.relu(target_margin - score_diff).mean()
        
        total_loss = loss_contrast
        
        loss_reconstruction = torch.tensor(0.0, device=good_output.device)
        if include_reconstruction:
            loss_reconstruction = F.mse_loss(good_output, good_target) * 0.1
        
        total_loss = total_loss + loss_reconstruction
        
        # Curiosity loss
        good_probs = F.softmax(torch.stack([good_scores_scaled, bad_scores_scaled]), dim=0)[0]
        good_probs = torch.clamp(good_probs, min=1e-7, max=1.0-1e-7)
        curiosity_loss = -torch.log(good_probs).mean() * 0.1
        total_loss = total_loss + curiosity_loss
        
        stats = {
            'loss_total': total_loss.item(),
            'loss_contrast': loss_contrast.item(),
            'loss_reconstruction': loss_reconstruction.item() if include_reconstruction else 0.0,
            'loss_curiosity': curiosity_loss.item(),
            'score_diff': score_diff.mean().item(),
            'good_sim': good_scores.mean().item(),
            'bad_sim': bad_scores.mean().item(),
            'margin': target_margin,
            'current_margin': target_margin,
            'stuck_counter': self._stuck_counter.item()
        }
        
        return total_loss, stats
        
# ===================================================================
# 🧠 МЕТА-КОГНИТИВНЫЙ ПРЕДИКТОР v3.3 — «ВНУТРЕННИЙ ГОЛОС» (финальная версия)
# ===================================================================
class MetaCognitivePredictor(nn.Module):
    """
    Внутренний голос модели v3.3
    Полностью соответствует философии, которую ты описал:
    - Динамическое любопытство (не залипает)
    - Реальная многошаговая рефлексия
    - Осмысленные решения по трём этапам
    - Адаптивная скорость обучения
    """
    def __init__(self, slot_dim, max_batch=128, history_len=50, hidden_dim=None):
        super().__init__()
        self.slot_dim = slot_dim
        self.max_batch = max_batch
        self.history_len = history_len
        hidden_dim = hidden_dim or max(slot_dim * 2, 128)
        # Предиктор
        self.predictor = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, slot_dim),
            nn.Tanh()
        )
        # Мета-сеть
        self.meta_network = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 5),
        )
        # Память
        self.register_buffer("_error_memory", torch.zeros(max_batch, history_len, slot_dim))
        self.register_buffer("_error_confidences", torch.zeros(max_batch, history_len))
        self.register_buffer("_error_ptr", torch.zeros(max_batch, dtype=torch.long))
        self.register_buffer("_success_memory", torch.zeros(max_batch, history_len, slot_dim))
        self.register_buffer("_success_confidences", torch.zeros(max_batch, history_len))
        self.register_buffer("_success_ptr", torch.zeros(max_batch, dtype=torch.long))
        self.register_buffer("_recent_error_rate", torch.zeros(max_batch))
        self.register_buffer("_recent_success_rate", torch.zeros(max_batch))
        self.register_buffer("_learning_momentum", torch.ones(max_batch) * 0.5)
        # Рефлексия
        self.reflection_gate = nn.Sequential(nn.Linear(slot_dim * 2, slot_dim), nn.Sigmoid())
        self.reflection_update = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, slot_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.diversity_coef = nn.Parameter(torch.tensor(0.05))
        self.register_buffer("_base_temperature", torch.tensor(1.0))
        self.temperature_adaptation = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("_recency_weights", torch.linspace(0.5, 1.0, history_len))
        self._last_meta = None
        self._in_reflection = False
        self._init_weights()
        logger.info(f"🧠 MetaCognitivePredictor v3.3 — Внутренний голос загружен")
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.02)
        with torch.no_grad():
            self.meta_network[-1].weight.fill_(0.03)
            self.meta_network[-1].bias.fill_(0.1)
            
    def forward(self, current_state, context=None, temperature=None, return_meta=True, step_count=0):
        batch_size = current_state.shape[0]
        if batch_size > self.max_batch:
            logger.warning(f"Batch {batch_size} > max_batch {self.max_batch} → обрезаем")
            current_state = current_state[:self.max_batch]
            if context is not None:
                context = context[:self.max_batch]
        
        device = current_state.device
        predicted = self.predictor(current_state)
        
        # Мета-вход
        if context is not None:
            meta_input = torch.cat([current_state, context], dim=-1)
        else:
            meta_input = torch.cat([current_state, current_state - predicted], dim=-1)
        
        temp = temperature if temperature is not None else torch.clamp(
            self._base_temperature * (1.0 + self.temperature_adaptation * self._compute_novelty(current_state).mean()), 
            0.5, 2.0
        )
        
        meta_logits = self.meta_network(meta_input)
        meta_probs = torch.sigmoid(meta_logits / temp)
        
        confidence = meta_probs[..., 0]
        doubt = meta_probs[..., 1]
        curiosity = meta_probs[..., 2]
        readiness = meta_probs[..., 3]
        need_recheck = meta_probs[..., 4]
        
        # ===========================================================
        # 🔥 ИСПРАВЛЕНИЕ 3: передаём step_count
        # ===========================================================
        sim_error = self._check_similar_to_errors_weighted(current_state, step_count=step_count)
        sim_success = self._check_similar_to_success_weighted(current_state, step_count=step_count)
        
        novelty = 1.0 - sim_success
        
        # 🔥 ИСПРАВЛЕННОЕ любопытство + сомнение
        curiosity = curiosity * 0.82 + novelty * 0.18  # чуть сильнее затухание
        curiosity = torch.clamp(curiosity, 0.0, 0.92)  # потолок ниже
        
        # Самокритика теперь сильнее влияет на doubt
        self_doubt = self.critic(torch.cat([current_state, predicted], dim=-1)).squeeze(-1)
        doubt = torch.clamp(doubt + self_doubt * 0.45, 0.05, 1.0)  # минимум 0.05 + сильнее
        
        # 🔥 Убираем агрессивное подавление doubt
        # confidence = confidence * (1.0 - doubt * 0.7) ← УДАЛИТЬ ЭТУ СТРОКУ!
        confidence = torch.clamp(confidence, 0.1, 0.95)  # просто ограничиваем
        
        # === ОБУЧЕНИЕ (target_doubt стал честнее) ===
        meta_loss = torch.tensor(0.0, device=device)
        if self.training:
            error = F.mse_loss(predicted, current_state, reduction='none').mean(dim=-1)
            target_conf = torch.sigmoid(1.0 - error * 5.0)
            target_doubt = 1.0 - target_conf + sim_error * 0.45 + (1.0 - novelty) * 0.15
            target_doubt = torch.clamp(target_doubt, 0.1, 0.9)
            target_curiosity = torch.clamp(novelty * 0.85, 0.0, 0.92)
            target_readiness = 1.0 - target_conf
            target_recheck = torch.clamp(doubt * 1.3, 0.1, 1.0)
            
            loss_conf = F.mse_loss(confidence, target_conf)
            loss_doubt = F.mse_loss(doubt, target_doubt)
            loss_cur = F.mse_loss(curiosity, target_curiosity)
            loss_ready = F.mse_loss(readiness, target_readiness)
            loss_recheck = F.mse_loss(need_recheck, target_recheck)
            
            adaptation_speed = 0.05 + (1.0 - self._learning_momentum.mean()) * 0.15
            
            reg = torch.tensor(0.0, device=device)
            if confidence.numel() > 1:
                var = (torch.var(confidence) + torch.var(doubt) + torch.var(curiosity)) / 3
                reg = -self.diversity_coef * var
                reg = torch.clamp(reg, -0.1, 0.1)
            
            meta_loss = (loss_conf + loss_doubt + loss_cur + loss_ready + loss_recheck) * adaptation_speed + reg
            meta_loss = torch.clamp(meta_loss, 0.0, 2.0)
        
        decision = self._make_decision(confidence, doubt, need_recheck, curiosity)
        
        # ===========================================================
        # 🔥 ИСПРАВЛЕНИЕ: теперь doubt всегда сохраняется корректно
        # ===========================================================
        self._last_meta = {
            "confidence": confidence,
            "doubt": doubt,  # ← вот это было потеряно
            "curiosity": curiosity,
            "readiness": readiness,
            "need_recheck": need_recheck,
            "self_doubt": self_doubt,
            "novelty": novelty,
            "decision": decision,
            "meta_loss": meta_loss,
            "temperature": temp.item() if isinstance(temp, torch.Tensor) else temp,
            "step_count": step_count  # ← добавляем для отладки
        }
        
        if return_meta:
            return predicted, self._last_meta
        return predicted
        
    def _make_decision(self, confidence, doubt, need_recheck, curiosity):
        decisions = []
        decision_counts = {"act": 0, "think": 0, "explore": 0, "deep_rethink": 0, "recheck": 0}
        for i in range(confidence.shape[0]):
            c = confidence[i].item()
            d = doubt[i].item()
            r = need_recheck[i].item()
            q = curiosity[i].item()
            if c > 0.78 and q < 0.45 and d < 0.35: # act — строже
                dec = "act"
            elif c > 0.52 and (d > 0.38 or r > 0.45 or q > 0.65): # think / deep_rethink
                dec = "think" if d > 0.38 else "deep_rethink"
            elif q > 0.72 or d > 0.55:
                dec = "explore" if q > 0.82 else "deep_rethink"
            else:
                dec = "recheck"
            decisions.append(dec)
            decision_counts[dec] += 1
        if torch.rand(1).item() < 0.03:
            logger.debug(f"📊 Решения: act={decision_counts['act']}, think={decision_counts['think']}, "
                        f"explore={decision_counts['explore']}, deep={decision_counts['deep_rethink']}, "
                        f"recheck={decision_counts['recheck']}")
        return decisions
        
    def reflect(self, current_state, context=None, max_steps=5, early_stop_threshold=0.65):
        """Многошаговая рефлексия с защитой от лишних шагов"""
        refined = current_state.clone()
        steps_taken = 0
        final_meta = None
        confidence_before = 0.0
        self._in_reflection = True
        try:
            for step in range(max_steps):
                predicted, meta = self.forward(refined, context, temperature=0.75)
                final_meta = meta
                confidence_before = meta["confidence"].mean().item()
                mean_curiosity = meta["curiosity"].mean().item()
                mean_doubt = meta["doubt"].mean().item()
                mean_recheck = meta["need_recheck"].mean().item()
                need_more = (mean_curiosity > 0.75 or mean_doubt > 0.45 or mean_recheck > early_stop_threshold)
                # 🔥 Твоя правка №1: защита от бесконечной рефлексии
                if steps_taken > 0 and confidence_before > 0.7:
                    logger.debug(f"🛑 Рефлексия остановлена: уверенность уже высокая ({confidence_before:.3f})")
                    break
                if not need_more or step == max_steps - 1:
                    break
                combined = torch.cat([refined, predicted], dim=-1)
                gate = self.reflection_gate(combined)
                update = self.reflection_update(combined)
                refined = refined + gate * update
                refined = F.normalize(refined, dim=-1)
                steps_taken += 1
        finally:
            self._in_reflection = False
        confidence_after = final_meta["confidence"].mean().item() if final_meta else 0.0
        return refined, {
            "steps": steps_taken,
            "final_meta": final_meta,
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "improved": confidence_after > confidence_before
        }
        
    def _check_similar_to_errors_weighted(self, current_state, step_count=0):
        """🔥 ИСПРАВЛЕНО: безопасное сравнение размерностей + step_count параметр"""
        batch_size = current_state.shape[0]
        similarities = torch.zeros(batch_size, device=current_state.device)
       
        for b in range(batch_size):
            ptr = int(self._error_ptr[b].item())
            if ptr == 0:
                continue
           
            errors = self._error_memory[b, :ptr]  # [ptr, slot_dim]
            if errors.shape[0] == 0:
                continue
           
            # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: проверяем размерность
            current = current_state[b:b+1]  # [1, slot_dim]
           
            # Если размерности не совпадают — пропускаем
            if current.shape[-1] != errors.shape[-1]:
                # 🔥 ИСПРАВЛЕНИЕ 3: используем переданный step_count
                if step_count % 100 == 0:
                    logger.warning(f"⚠ Размерности не совпадают: current={current.shape[-1]}, errors={errors.shape[-1]}")
                continue
           
            weights = self._recency_weights[:ptr].to(current_state.device)
            weights = weights / weights.sum()
           
            sims = F.cosine_similarity(current, errors, dim=-1)  # [ptr]
            similarities[b] = (sims * weights).sum()
       
        return similarities

    def _check_similar_to_success_weighted(self, current_state, step_count=0):
        """🔥 ИСПРАВЛЕНО: безопасное сравнение размерностей"""
        batch_size = current_state.shape[0]
        similarities = torch.zeros(batch_size, device=current_state.device)
       
        for b in range(batch_size):
            ptr = int(self._success_ptr[b].item())
            if ptr == 0:
                continue
           
            successes = self._success_memory[b, :ptr]
            if successes.shape[0] == 0:
                continue
           
            current = current_state[b:b+1]
           
            if current.shape[-1] != successes.shape[-1]:
                if step_count % 100 == 0:
                    logger.warning(f"⚠ Размерности не совпадают: current={current.shape[-1]}, successes={successes.shape[-1]}")
                continue
           
            weights = self._recency_weights[:ptr].to(current_state.device)
            weights = weights / weights.sum()
           
            sims = F.cosine_similarity(current, successes, dim=-1)
            similarities[b] = (sims * weights).sum()
       
        return similarities
        
    def _compute_novelty(self, current_state):
        return 1.0 - self._check_similar_to_success_weighted(current_state)
   
    def remember_outcome(self, batch_indices, was_correct, state, confidence):
        """Запоминаем исход (успех или ошибка) для будущих решений"""
        for b in batch_indices:
            if b >= self.max_batch:
                continue
           
            if was_correct[b]:
                ptr = int(self._success_ptr[b].item())
                self._success_memory[b, ptr] = state[b].detach()
                self._success_confidences[b, ptr] = confidence[b].item()
                self._success_ptr[b] = (ptr + 1) % self.history_len
                self._recent_success_rate[b] = self._recent_success_rate[b] * 0.95 + 0.05
            else:
                ptr = int(self._error_ptr[b].item())
                self._error_memory[b, ptr] = state[b].detach()
                self._error_confidences[b, ptr] = confidence[b].item()
                self._error_ptr[b] = (ptr + 1) % self.history_len
                self._recent_error_rate[b] = self._recent_error_rate[b] * 0.95 + 0.05
   
    def reset_state(self, batch_indices=None):
        """Сброс состояния для новых сессий"""
        if batch_indices is None:
            self._error_ptr.fill_(0)
            self._success_ptr.fill_(0)
            self._recent_error_rate.fill_(0)
            self._recent_success_rate.fill_(0)
            self._learning_momentum.fill_(0.5)
        else:
            for b in batch_indices:
                if b < self.max_batch:
                    self._error_ptr[b] = 0
                    self._success_ptr[b] = 0
                    self._recent_error_rate[b] = 0
                    self._recent_success_rate[b] = 0
                    self._learning_momentum[b] = 0.5
   
    def get_stats(self, batch_idx=0):
        """Статистика внутреннего голоса"""
        ptr_err = int(self._error_ptr[batch_idx].item())
        ptr_suc = int(self._success_ptr[batch_idx].item())
       
        return {
            "error_memory_size": ptr_err,
            "success_memory_size": ptr_suc,
            "recent_error_rate": float(self._recent_error_rate[batch_idx].item()),
            "recent_success_rate": float(self._recent_success_rate[batch_idx].item()),
            "learning_momentum": float(self._learning_momentum[batch_idx].item()),
        }
        
    def get_state(self):
        """
        Возвращает полное состояние мета-предиктора для сохранения
        """
        # Базовое состояние
        state = {
            'version': '3.0',
            'predictor_state_dict': self.predictor.state_dict(),
            'meta_network_state_dict': self.meta_network.state_dict(),
            'reflection_gate_state_dict': self.reflection_gate.state_dict(),
            'reflection_update_state_dict': self.reflection_update.state_dict(),
            'config': {
                'slot_dim': self.slot_dim,
                'max_batch': self.max_batch,
                'history_len': self.history_len,
            }
        }
       
        # Сохраняем critic если есть
        if hasattr(self, 'critic'):
            state['critic_state_dict'] = self.critic.state_dict()
       
        # Сохраняем все буферы
        buffers = {}
        buffer_names = [
            '_error_memory', '_error_confidences', '_error_ptr',
            '_success_memory', '_success_confidences', '_success_ptr',
            '_recent_error_rate', '_recent_success_rate', '_learning_momentum',
            '_base_temperature', '_recency_weights'
        ]
       
        for name in buffer_names:
            if hasattr(self, name):
                buffer_val = getattr(self, name)
                if isinstance(buffer_val, torch.Tensor):
                    buffers[name] = buffer_val.cpu().clone()
                else:
                    buffers[name] = buffer_val
       
        state['buffers'] = buffers
       
        # Сохраняем параметры (если есть)
        params = {}
        param_names = [
            'diversity_coef', 'temperature_adaptation'
        ]
       
        for name in param_names:
            if hasattr(self, name):
                val = getattr(self, name)
                if isinstance(val, torch.Tensor):
                    params[name] = val.cpu().clone()
                else:
                    params[name] = val
       
        state['params'] = params
       
        # Статистика для логирования
        try:
            state['stats'] = self.get_stats()
        except:
            state['stats'] = {}
       
        return state
        
    def load_state(self, state, device='cpu'):
        """
        Загружает состояние мета-предиктора из сохранённого словаря
        """
        # Загружаем веса
        if 'predictor_state_dict' in state:
            self.predictor.load_state_dict(state['predictor_state_dict'], strict=False)
       
        if 'meta_network_state_dict' in state:
            try:
                self.meta_network.load_state_dict(state['meta_network_state_dict'], strict=False)
            except Exception as e:
                logger.warning(f"⚠ Ошибка загрузки meta_network: {e}")
                # Если несовместимая версия, пробуем мигрировать
                old_sd = state['meta_network_state_dict']
                new_sd = self.meta_network.state_dict()
                for key in new_sd.keys():
                    if key in old_sd and old_sd[key].shape == new_sd[key].shape:
                        new_sd[key] = old_sd[key]
                    elif key in old_sd and key.endswith('weight') and old_sd[key].shape[0] == 3 and new_sd[key].shape[0] == 5:
                        # Миграция 3→5 выходов
                        new_sd[key][:3] = old_sd[key]
                        logger.info(f" 📊 Миграция {key}: 3→5 выходов")
                    elif key in old_sd and key.endswith('bias') and old_sd[key].shape[0] == 3 and new_sd[key].shape[0] == 5:
                        new_sd[key][:3] = old_sd[key]
                        logger.info(f" 📊 Миграция {key}: 3→5 выходов")
                self.meta_network.load_state_dict(new_sd, strict=False)
       
        if 'reflection_gate_state_dict' in state:
            self.reflection_gate.load_state_dict(state['reflection_gate_state_dict'], strict=False)
        if 'reflection_update_state_dict' in state:
            self.reflection_update.load_state_dict(state['reflection_update_state_dict'], strict=False)
        if 'critic_state_dict' in state and hasattr(self, 'critic'):
            self.critic.load_state_dict(state['critic_state_dict'], strict=False)
       
        # Загружаем буферы
        if 'buffers' in state:
            for name, value in state['buffers'].items():
                if hasattr(self, name):
                    buffer_val = getattr(self, name)
                    if isinstance(buffer_val, torch.Tensor) and isinstance(value, torch.Tensor):
                        if buffer_val.shape == value.shape:
                            buffer_val.data.copy_(value.to(device))
                        else:
                            logger.warning(f"⚠ Несовместимый shape буфера {name}: {buffer_val.shape} vs {value.shape}")
       
        # Загружаем параметры
        if 'params' in state:
            for name, value in state['params'].items():
                if hasattr(self, name):
                    param = getattr(self, name)
                    if isinstance(param, torch.Tensor) and isinstance(value, torch.Tensor):
                        if param.shape == value.shape:
                            param.data.copy_(value.to(device))
                        else:
                            logger.warning(f"⚠ Несовместимый shape параметра {name}: {param.shape} vs {value.shape}")
       
        #logger.info(f"✅ Мета-предиктор v{state.get('version', '3.0')} загружен")
       
# ===================================================================
# ADAPTIVE MEMORY HEADS (С ИНТЕЛЛЕКТУАЛЬНЫМИ HARD NEGATIVES) - ВЕРСИЯ 11.1
# ===================================================================
class AdaptiveMemoryHeads(nn.Module):
    """Комбинированный модуль памяти (Mamba/RWKV/Transfomer) - v11.1 с исправлениями для batch>1"""
   
    # 🔥 Добавляем свойство для безопасного доступа
    @property
    def experience_exchange(self):
        if hasattr(self, '_exchange_ref') and self._exchange_ref:
            return self._exchange_ref[0]
        return None
   
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
       
        # 🔥 Флаг для отслеживания версии и блокировки устаревших методов
        self._use_deprecated_sleep_methods = False
       
        # 🔥 ДОБАВЛЕНО: Инициализация параметров для работы с нейроконтроллером
        self.control_signal_names = [
            "curiosity_threshold", "recall_threshold_mod", "write_intensity",
            "noise_mod", "negative_sensitivity", "exploration_bonus", "learning_rate_mult",
            "hebb_write_gate", "hebb_lr_mult", "stability_factor"
        ]
       
        # 🔥 ПАРАМЕТРЫ ПО УМОЛЧАНИЮ (используются если нет сигналов от контроллера)
        self.default_controls = {
            "curiosity_threshold": getattr(config, 'curiosity_threshold', 0.18),
            "recall_threshold_mod": 1.0,
            "write_intensity": 1.0,
            "noise_mod": 1.0,
            "negative_sensitivity": 1.0,
            "exploration_bonus": 0.0,
            "learning_rate_mult": 1.0,
        }
       
        # 🔥 АДАПТИВНЫЕ ВЕСА ДЛЯ ПАРАМЕТРОВ КОНТРОЛЛЕРА
        self.control_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(1.0))
            for name in self.control_signal_names
        })
       
        # Для chimera и twist (положительная схожесть)
        self.max_similarity_chimera = getattr(config, 'max_similarity_chimera', 0.95)
        self.min_similarity_chimera = getattr(config, 'min_similarity_chimera', 0.8)
       
        self.max_similarity_twist = getattr(config, 'max_similarity_twist', 0.9)
        self.min_similarity_twist = getattr(config, 'min_similarity_twist', 0.4)
       
        # Для lobotomy (ОТРИЦАТЕЛЬНАЯ схожесть!)
        self.max_similarity_lobotomy = getattr(config, 'max_similarity_lobotomy', -0.2)
        self.min_similarity_lobotomy = getattr(config, 'min_similarity_lobotomy', -0.7)
       
        # Для gaslight (очень низкая схожесть)
        self.max_similarity_gaslight = getattr(config, 'max_similarity_gaslight', 0.3)
        self.min_similarity_gaslight = getattr(config, 'min_similarity_gaslight', 0.1)
       
        # Gaslight
        self.gaslight_prob = getattr(config, 'gaslight_prob', 0.2) # 🔥 НОВЫЙ параметр, цель 10-15%
       
        # Контрастный сон
        self.contrastive_min_negative = getattr(config, 'contrastive_min_negative', 60)
        self.contrastive_temperature = getattr(config, 'contrastive_temperature', 0.05)
       
        # Запись в блокнот и обучение
        self.writer_insight_threshold = getattr(config, 'writer_insight_threshold', 0.42)
        
        # 🔥 Curriculum для recovery
        self.recovery_level = 0 # 0-3
        self.recovery_success_streak = 0
        self.recovery_failure_streak = 0
        self.max_recovery_level = 3
        self.hard_negative_stats = {
            'avg_similarity': 0.0,
            'update_counter': 0,
            'target_avg_similarity': -0.45,
            'last_adjustment': 0
        }
       
        # 🔥 Для эволюции мыслей
        self.recovered_thoughts_history = []
        self.max_recovered_thoughts = 20
        # 🔥 ПРОРАБОТАННАЯ СИСТЕМА СТАРЕНИЯ
        self.age_decay_factor = getattr(config, 'age_decay_factor', 0.1)
        self.max_age_hard_negative = getattr(config, 'max_age_hard_negative', 25)
        self.max_age_gaslight = getattr(config, 'max_age_gaslight', 15)
        self.max_age_positive = getattr(config, 'max_age_positive', 20)
       
        # Лосс веса
        self.sleep_loss_weights = getattr(config, 'sleep_loss_weights', {
            'contrastive': 1.0,
            'reconstruction': 0.1,
            'curiosity': 0.1,
            'recovery': 0.2
        })
       
        # Состояния
        self._sleep_triggered = False
        self.pending_sleep = False
        self.is_sleeping = False
        self.thoughts_threshold = getattr(config, 'notebook_size', 300)
       
        # Блокировки
        self._model_lock = threading.RLock()
        self._sleep_event = threading.Event()
        self._sleep_lock = self._model_lock
       
        # 🔥 ФЛАГ ДЛЯ ЗАЩИТЫ ОТ ГЕНЕРАЦИИ
        self.is_generating = False
        self.sleep_postponed = 0
        self._generation_count = 0
       
        # Блокнот
        self.data_manager = DataManager()
       
        # Размерности
        mamba_dim = getattr(config, 'slot_size', config.n_embd // 2)
        d_state = getattr(config, 'mamba_d_state', 16)
        chunk_size = getattr(config, 'mamba_chunk_size', 32) # 🔥 НОВЫЙ параметр
        self.shared_dim = getattr(config, 'shared_exchange_dim', config.n_embd // 2)
       
        logger.info(f"🧠 Инициализация Mamba (MIMO): dim={mamba_dim}, d_state={d_state}, chunk_size={chunk_size}, shared_dim={self.shared_dim}")
       
        # Параметры для Hard Negatives
        self.hard_negative_delta_threshold = getattr(config, 'hard_negative_delta_threshold', 0.1)
        self.hard_negative_probability = getattr(config, 'hard_negative_probability', 0.95)
       
        # Адаптер
        self.exp_to_mamba_adapter = nn.Linear(self.shared_dim, mamba_dim)
        nn.init.xavier_uniform_(self.exp_to_mamba_adapter.weight, gain=0.1)
        nn.init.zeros_(self.exp_to_mamba_adapter.bias)
       
        # Mamba с MIMO-оптимизацией
        self.mamba_writer = ThaliaMambaHead(mamba_dim, d_state=d_state, chunk_size=chunk_size)
        self.mamba_reader = ThaliaMambaHead(mamba_dim, d_state=d_state, chunk_size=chunk_size)
       
        # ═══════════════════════════════════════════════════════════════════
        # 🔥 ЦЕНТРОИДНАЯ ПАМЯТЬ v1.1
        # ═══════════════════════════════════════════════════════════════════
        # конвертируем строку в torch.device
        device = config.device
        if isinstance(device, str):
            device = torch.device(device)
       
        self.centroid_memory = CentroidMemoryManager(
            num_slots=getattr(config, 'num_sedimentary_slots', 512),
            slot_dim=getattr(config, 'slot_size', config.n_embd // 2),
            core_slots=getattr(config, 'core_slots_count', 8),
            similarity_threshold=getattr(config, 'cos_sim_threshold', 0.85),
            merge_threshold=getattr(config, 'merge_threshold', 0.96),
            split_variance_threshold=getattr(config, 'split_variance_threshold', 0.3),
            eviction_utility_threshold=getattr(config, 'eviction_utility_threshold', 0.01),
            buffer_size=getattr(config, 'buffer_size', 2000),
            device=device,
            enable_linker=getattr(config, 'enable_linker', True),
            linker_lr=getattr(config, 'linker_lr', 0.001),
            linker_weight_decay=getattr(config, 'linker_weight_decay', 0.01),
            linker_hidden_dim=getattr(config, 'linker_hidden_dim', None),
            linker_batch_size=getattr(config, 'linker_batch_size', 32),
            linker_train_frequency=getattr(config, 'linker_train_frequency', 5),
        )
       
        # ===========================================================
        # 🔥 ВАЖНО: сразу после создания центроидной памяти, сбрасываем кэш
        # ===========================================================
        if hasattr(self, 'centroid_memory'):
            self.centroid_memory.reset_cache()
            logger.debug("🔄 Кэш центроидной памяти сброшен при инициализации")
       
        # Замораживаем ядро личности
        self.centroid_memory.freeze_core()
       
        # 🔥 ДОБАВЛЯЕМ ПРОЕКТОР ДЛЯ СТАБИЛИЗАЦИИ ПРОСТРАНСТВ
        self.hebb_to_centroid = nn.Linear(config.n_embd, self.centroid_memory.slot_dim)
        nn.init.xavier_uniform_(self.hebb_to_centroid.weight, gain=0.1)
        nn.init.zeros_(self.hebb_to_centroid.bias)
       
        # 🔥 ДОБАВЛЯЕМ СИСТЕМУ ЛЮБОПЫТСТВА ДЛЯ LINKER
        self.register_buffer(
            "slot_curiosity",
            torch.zeros(self.centroid_memory.num_slots)
        )
        self.curiosity_decay = 0.99
        self.prediction_errors = deque(maxlen=100)
        # ===========================================================
        # 🔥 МЕТА-КОГНИТИВНЫЙ ПРЕДИКТОР
        # ===========================================================
        self.meta_predictor = MetaCognitivePredictor(
            slot_dim=self.centroid_memory.slot_dim,
            max_batch=getattr(config, 'max_batch_size', 64),
            history_len=getattr(config, 'meta_history_len', 50), # увеличили до 50
            hidden_dim=getattr(config, 'predictor_hidden_dim', None)
        )
        # Параметры рефлексии
        self.max_reflection_steps = getattr(config, 'max_reflection_steps', 2)
        self.reflection_temperature = getattr(config, 'reflection_temperature', 1.2)
        self.reflection_early_stop = getattr(config, 'reflection_early_stop', 0.7)
        # Регистры для трекинга
        self.register_buffer("_total_reflections", torch.tensor(0))
        self.register_buffer("_reflection_depth_avg", torch.tensor(0.0))
        # Предиктор
        predictor_dim = getattr(config, 'slot_size', config.n_embd // 2)
        self.predictor = nn.Sequential(
            nn.Linear(predictor_dim, predictor_dim * 2),
            nn.LayerNorm(predictor_dim * 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(predictor_dim * 2, predictor_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(predictor_dim * 2, predictor_dim)
        )
       
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)
                
        self._predictor_loss = None
        
        #── НОВЫЙ ИЗЯЩНЫЙ MEMORY GATE v2.0 ───────────────────────────────
        self.memory_gate_logit = nn.Parameter(torch.tensor(-0.8))   # старт ≈ 0.31

        # EMA для стабильности
        self.register_buffer('memory_gate_ema', torch.tensor(0.35))

        # Весовые коэффициенты (модель сама научится, какие мета-метрики важнее)
        self.gate_readiness_weight   = nn.Parameter(torch.tensor(0.65))
        self.gate_curiosity_weight   = nn.Parameter(torch.tensor(0.25))
        self.gate_confidence_weight  = nn.Parameter(torch.tensor(0.10))

        # ── GATE AUX COEF — теперь динамический, но не ломаемый градиентами
        self.register_buffer('gate_aux_coef', torch.tensor(0.08))   # базовое значение
        self.register_buffer('gate_aux_coef_ema', torch.tensor(0.08))
        self.gate_aux_coef_min = 0.01
        self.gate_aux_coef_max = 0.25
 
        # 🔥 ИСПРАВЛЕННАЯ CURIOSITY SYSTEM
        self.curiosity = CuriositySystem(config=config)
        self._curiosity_state = "neutral"
        self._curiosity_motivation = 0.3
        self._curiosity_arousal = 0.0
        self._curiosity_habituation = 0.0

        # 🔥 МОТИВАЦИОННАЯ СИСТЕМА
        self.motivation_module = MotivationModule(config)
       
        # Параметры
        self.curiosity_threshold = getattr(config, 'curiosity_threshold', 0.18)
        self.predictor_coef = getattr(config, 'predictor_coef', 0.15)
        self.use_combo_loss = getattr(config, 'use_combo_loss', True)
        self._last_surprise = 0.05
       
        # Состояния Mamba
        self.mamba_writer_state = None
        self.mamba_reader_state = None
        self.mamba_state_size = d_state
       
        # EMA
        self.mamba_writer_ema = 0.05
        self.mamba_reader_ema = 0.05
       
        # Статистика
        self._last_writer_delta = 0.0
        self._last_reader_improvement = 0.0
        self.current_lr = getattr(config, 'sleep_lr', 8e-5)
        self.training_losses = []
        self.sleep_cycles = 0
        self.synchronous_sleep = getattr(config, 'synchronous_sleep', False)
        self.predictor_forget_counter = 0
        self._last_record_step = 0
        self._sleep_epoch_losses = []
        self._last_sleep_loss = 0.1 # 🔥 ДЛЯ ОБНОВЛЕНИЯ EMA
       
        # 🔥 ИСПРАВЛЕНИЕ 2: Персистентный контрастный лосс (сохраняет margin между снами)
        self.contrastive_loss_fn = StabilizedContrastiveLoss(
            base_margin=0.6,
            alpha=0.5,
            max_diff=1.0,
            growth=1.05,
            temperature=self.contrastive_temperature
        )
       
        # 🔥 СЧЁТЧИКИ
        self.step_count = 0
        self._last_sleep_step = 0
        self._recovery_improvement_history = []
        self._analysis_counter = 0
       
        # Инициализация
        self._reset_mamba_states()
        self.keep_last_after_prune = max(70, int(self.thoughts_threshold * 0.25))
       
        logger.info(f"📝 Mamba-система v11.1 с исправлениями для batch>1")
        # 🔥 ИСПРАВЛЕННЫЙ ВЫВОД ПАРАМЕТРОВ
        logger.info(f"⚔️ Hard Negatives параметры:")
        logger.info(f" Chimera: [{self.min_similarity_chimera:.2f}, {self.max_similarity_chimera:.2f}]")
        logger.info(f" Twist: [{self.min_similarity_twist:.2f}, {self.max_similarity_twist:.2f}]")
        logger.info(f" Lobotomy: [{self.min_similarity_lobotomy:.2f}, {self.max_similarity_lobotomy:.2f}]")
        logger.info(f" Gaslight: [{self.min_similarity_gaslight:.2f}, {self.max_similarity_gaslight:.2f}]")
        logger.info(f"🔥 Gaslight prob: {self.gaslight_prob*100}%")
        logger.info(f"💤 Contrastive min negative: {self.contrastive_min_negative}")
       
        # 🔥 ДИНАМИЧЕСКАЯ КОНФИГУРАЦИЯ
        self.config_updates = {
            'last_sim_avg': 0.0,
            'update_counter': 0
        }
        logger.info(f"🧠 ЦЕНТРОИДНАЯ ПАМЯТЬ активирована: {self.centroid_memory.num_slots} слотов, "
                   f"ядро: {self.centroid_memory.core_slots}")
        logger.info(f"🧠 Мета-когнитивный предиктор интегрирован в AdaptiveMemoryHeads")
       
# ===================================================================
# 🔥 НОВЫЙ МЕТОД ЗАПИСИ МЫСЛИ В ЦЕНТРОИДНУЮ ПАМЯТЬ
# ===================================================================
    def _write_thought_to_centroid(self, thought_data, delta_score=None):
        """🔥 УЛЬТРА-ЗАЩИЩЁННАЯ v10.2 — ИСПРАВЛЕННАЯ (device-aware)"""
        import traceback
       
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device('cpu')
            logger.warning("⚠ _write_thought_to_centroid: нет параметров, использую CPU")
       
        # Проверка устройства centroid_memory
        if hasattr(self.centroid_memory, 'centroids'):
            centroid_device = self.centroid_memory.centroids.device
            if centroid_device != model_device:
                logger.debug(f"🔄 Перемещаю центроидную память с {centroid_device} на {model_device}")
                self.centroid_memory = self.centroid_memory.to(model_device)
       
        # Случай 1: Вызов с двумя аргументами (вектор + дельта)
        if delta_score is not None:
            if isinstance(thought_data, torch.Tensor):
                vector = thought_data.to(model_device)
            else:
                try:
                    vector = torch.tensor(thought_data, dtype=torch.float32, device=model_device)
                except Exception as e:
                    logger.error(f"❌ _write_thought_to_centroid: Ошибка создания тензора: {e}")
                    return False
           
            if vector.dim() > 1:
                vector = vector.view(-1)
           
            vector = vector.to(model_device)
           
            thought_data = {
                'snapshot': vector,
                'delta': delta_score,
                'step': self.step_count,
                'age': 0,
                'type': 'veteran_consolidation'
            }
       
        if 'snapshot' not in thought_data or thought_data['snapshot'] is None:
            logger.warning("⚠ _write_thought_to_centroid: Нет snapshot в thought_data")
            return False
        try:
            snapshot = thought_data['snapshot']
           
            if isinstance(snapshot, list):
                try:
                    vector = torch.tensor(snapshot, dtype=torch.float32, device=model_device)
                except Exception as e:
                    logger.error(f"❌ _write_thought_to_centroid: Ошибка преобразования списка: {e}")
                    return False
            elif isinstance(snapshot, torch.Tensor):
                try:
                    vector = snapshot.to(model_device).contiguous()
                except Exception as e:
                    logger.error(f"❌ _write_thought_to_centroid: Ошибка перемещения тензора: {e}")
                    return False
            else:
                logger.warning(f"⚠ _write_thought_to_centroid: Неподдерживаемый тип snapshot: {type(snapshot)}")
                return False
            if vector.dim() > 1:
                vector = vector.view(-1)
           
            vector = F.normalize(vector, dim=-1)
            vector = vector.to(model_device)
            if hasattr(self.centroid_memory, 'centroids'):
                centroid_device = self.centroid_memory.centroids.device
                if centroid_device != model_device:
                    logger.warning(f"⚠ Центроидная память на {centroid_device}, модель на {model_device}. Перемещаем...")
                    self.centroid_memory = self.centroid_memory.to(model_device)
            delta = float(thought_data.get('delta', 0.5))
            thought_id = thought_data.get('step', self.step_count)
            try:
                result = self.centroid_memory.update_slot_centroid(
                    thought_vector=vector,
                    thought_id=thought_id,
                    thought_delta=delta,
                    force_slot=None
                )
            except Exception as e:
                logger.error(f"❌ _write_thought_to_centroid: Ошибка в update_slot_centroid: {e}")
                logger.error(traceback.format_exc())
                return False
            action = result.get('action', 'none')
            similarity = result.get('similarity', 0.0)
            if action == 'new_slot':
                logger.info(f"🆕 Новый слот в центроиде: Δ={delta:.3f} | sim={similarity:.3f}")
            elif action == 'update' or action == 'semantic_merge':
                if thought_data.get('type') == 'veteran_consolidation':
                    logger.info(f"🎖️ Ветеран консолидирован в существующий слот: sim={similarity:.3f}")
            elif action == 'none':
                if thought_data.get('type') == 'veteran_consolidation':
                    logger.warning(f"⚠️ Ветеран НЕ консолидирован: sim={similarity:.3f}")
            return action != 'none'
        except Exception as e:
            logger.error(f"❌ _write_thought_to_centroid: Необработанная ошибка: {e}")
            logger.error(traceback.format_exc())
            return False
           
# ===================================================================
# 🔥 ЗАПРОС К ЦЕНТРОИДНОЙ ПАМЯТИ
# ===================================================================
    def query_memory(self, query_vector: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, Dict]:
        """🔥 УЛЬТРА-ЗАЩИЩЁННАЯ версия query_memory"""
        if query_vector is None:
            return torch.zeros(self.centroid_memory.slot_dim, device=self.config.device), {
                'found': 0, 'top_slots': [], 'top_sims': [], 'weights': []
            }
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
        query_vector = query_vector.to(model_device)
        query_vector = F.normalize(query_vector.view(-1), dim=-1)
       
        if hasattr(self, 'centroid_memory'):
            centroid_device = self.centroid_memory.centroids.device
            if centroid_device != model_device:
                logger.warning(f"⚠ Центроидная память на {centroid_device}, перемещаю на {model_device}")
                self.centroid_memory = self.centroid_memory.to(model_device)
           
            if hasattr(self.centroid_memory, '_ensure_device_consistency'):
                self.centroid_memory._ensure_device_consistency()
           
            if hasattr(self.centroid_memory, '_cache_dirty') and self.centroid_memory._cache_dirty:
                self.centroid_memory._update_cache()
       
        try:
            context, info = self.centroid_memory.query(
                query_vector=query_vector,
                top_k=top_k,
                use_transitions=True
            )
           
            safe_info = {
                'found': info.get('found', 0),
                'top_slots': info.get('top_slots', []),
                'top_sims': info.get('top_sims', []),
                'weights': info.get('weights', [])
            }
           
            if context.device != model_device:
                context = context.to(model_device)
           
            return context, safe_info
           
        except Exception as e:
            logger.warning(f"⚠ query_memory упал: {e}")
            return torch.zeros(self.centroid_memory.slot_dim, device=model_device), {
                'found': 0, 'top_slots': [], 'top_sims': [], 'weights': []
            }
            
    def get_character_graph(self):
        return self.centroid_memory.get_character_graph()
        
    def get_core_personality(self):
        return self.centroid_memory.get_core_personality()
        
    def _log_memory_status(self):
        stats = self.centroid_memory.get_stats()
       
        non_zero = stats.get('transition_non_zero', 0)
        total = stats.get('transition_total', self.centroid_memory.num_slots ** 2)
       
        logger.info(f"🧠 ЦЕНТРОИДНАЯ ПАМЯТЬ: {stats['active_slots']}/{stats['total_slots']} слотов "
                    f"({stats['utilization']:.1%}), avg_utility={stats['avg_utility']:.3f}")
       
        if stats['transition_density'] < 0.0001:
            logger.info(f"🔗 Transition density: {stats['transition_density']:.2e} "
                       f"(ненулевых: {non_zero}/{total}) | "
                       f"Merges: {stats['total_merges']}, Splits: {stats['total_splits']}, "
                       f"Evictions: {stats['total_evictions']}")
        else:
            logger.info(f"🔗 Transition density: {stats['transition_density']:.4f} "
                       f"(ненулевых: {non_zero}/{total}) | "
                       f"Merges: {stats['total_merges']}, Splits: {stats['total_splits']}, "
                       f"Evictions: {stats['total_evictions']}")
       
        if self.step_count % 200 == 0: # раз в 200 шагов
            self.centroid_memory.show_hubs(top_k=5)
            
# ===================================================================
# ПРИМЕНЕНИЕ СИГНАЛОВ КОНТРОЛЛЕРА
# ===================================================================
    def apply_control_signals(self, control_signals=None):
        """🔥 ПРИМЕНЕНИЕ СИГНАЛОВ ОТ НЕЙРОКОНТРОЛЛЕРА К ПАМЯТИ"""
        if control_signals is None:
            return self.default_controls.copy()
       
        applied_controls = {}
       
        for signal_name in self.control_signal_names:
            if signal_name in control_signals:
                controller_value = control_signals[signal_name]
                weight = self.control_weights[signal_name].item()
                applied_value = controller_value * weight
               
                if signal_name == "curiosity_threshold":
                    applied_value = max(0.05, min(0.35, applied_value))
                    self.curiosity_threshold = applied_value
                   
                elif signal_name == "write_intensity":
                    intensity_mult = max(0.5, min(2.0, applied_value))
                    self.writer_insight_threshold = getattr(self.config, 'writer_insight_threshold', 0.5) * intensity_mult
                   
                elif signal_name == "noise_mod":
                    self.hard_negative_probability = max(0.3, min(0.99,
                        getattr(self, 'hard_negative_probability', 0.9) * applied_value))
                   
                elif signal_name == "exploration_bonus":
                    if hasattr(self, '_curiosity_motivation'):
                        self._curiosity_motivation = min(1.0,
                            self._curiosity_motivation + applied_value * 0.1)
               
                applied_controls[signal_name] = applied_value
               
                if self.step_count % 200 == 0 and signal_name in ["curiosity_threshold", "write_intensity"]:
                    logger.debug(f"🧠 Контроллер→Память: {signal_name}={applied_value:.3f} (вес={weight:.2f})")
            else:
                applied_controls[signal_name] = self.default_controls[signal_name]
       
        self._update_control_weights(control_signals)
       
        return applied_controls
        
    def _compute_memory_gate(self, meta: Optional[Dict] = None, motivation: float = 0.5) -> torch.Tensor:
        """🔥 Изящный gate + aux-loss + мотивация"""
        device = self.memory_gate_logit.device

        # Гарантируем существование aux-loss буферов
        if not hasattr(self, '_gate_aux_loss'):
            self._gate_aux_loss = torch.tensor(0.0, device=device)
        if not hasattr(self, '_gate_weights_aux_loss'):
            self._gate_weights_aux_loss = torch.tensor(0.0, device=device)

        if meta is None or not isinstance(meta, dict):
            # Fallback — осторожный режим
            gate = torch.sigmoid(self.memory_gate_logit)
            gate = torch.clamp(gate, 0.05, 0.5)
            self._gate_aux_loss = torch.tensor(0.0, device=device)
            self._gate_weights_aux_loss = torch.tensor(0.0, device=device)
        else:
            # Основная логика
            readiness  = meta.get('readiness',  torch.tensor(0.5, device=device)).mean()
            curiosity  = meta.get('curiosity',  torch.tensor(0.3, device=device)).mean()
            confidence = meta.get('confidence', torch.tensor(0.5, device=device)).mean()

            # Нормализация весов через softmax (модель учится балансу)
            weights = torch.softmax(torch.stack([
                self.gate_readiness_weight,
                self.gate_curiosity_weight,
                self.gate_confidence_weight
            ]), dim=0)

            combined = (
                readiness  * weights[0] +
                curiosity  * weights[1] +
                confidence * weights[2]
            )

            gate_logit = self.memory_gate_logit + 3.0 * (combined - 0.5)
            gate = torch.sigmoid(gate_logit)
            gate = torch.clamp(gate, 0.05, 0.92)

            # ── Aux-loss для обучения весов (чтобы они не уходили в 0 или 1)
            target = torch.tensor(0.333, device=device)
            loss_read = F.mse_loss(weights[0], target)
            loss_cur  = F.mse_loss(weights[1], target)
            loss_conf = F.mse_loss(weights[2], target)

            self._gate_aux_loss = (loss_read + loss_cur + loss_conf) * 0.01
            self._gate_weights_aux_loss = self._gate_aux_loss.clone()

        # ── Учёт мотивации
        mot_tensor = torch.tensor(motivation, device=device) if isinstance(motivation, (int, float)) else motivation.to(device)
        gate = gate * (0.7 + 0.6 * mot_tensor)
        gate = torch.clamp(gate, 0.05, 0.92)

        # Принудительно скаляр
        gate = gate.view(-1).mean()

        # EMA
        with torch.no_grad():
            self.memory_gate_ema.mul_(0.92).add_(gate * 0.08)

        return gate
 
    def _update_gate_aux_coef(self, gate_loss_value: float, motivation: float, meta_confidence: float = 0.5):
        """🔥 Адаптивное обновление coef на основе реального опыта модели"""
        with torch.no_grad():
            # 1. Если gate_loss слишком большой — coef должен быть меньше
            # 2. Если мотивация высокая и gate стабильный — coef можно увеличить
            quality_factor = (1.0 - min(1.0, gate_loss_value * 8.0))   # 0..1
            motivation_factor = motivation * 0.6
            confidence_factor = meta_confidence * 0.4
            
            # 🔥 ИСПРАВЛЕНО: добавлено влияние confidence_factor
            # Максимальная сумма: 0.02 + 0.10 + 0.06 + 0.04 = 0.22 (безопасно ниже твоего максимума 0.25)
            adaptive_target = 0.02 + (quality_factor * 0.10) + (motivation_factor * 0.10) + (confidence_factor * 0.10)
            
            # EMA-обновление (очень плавно)
            self.gate_aux_coef_ema.mul_(0.94).add_(adaptive_target * 0.06)
            
            # Применяем с клиппингом
            new_coef = torch.clamp(self.gate_aux_coef_ema,
                                  self.gate_aux_coef_min,
                                  self.gate_aux_coef_max)
            
            self.gate_aux_coef.copy_(new_coef)
            
            if hasattr(self, 'step_count') and self.step_count % 300 == 0:
                logger.info(f"⚖️ Gate aux_coef адаптирован → {self.gate_aux_coef.item():.4f} "
                           f"(loss={gate_loss_value:.4f}, motivation={motivation:.3f}, conf={meta_confidence:.3f})")
 
    def _update_control_weights(self, control_signals):
        """🔥 АДАПТИВНОЕ ОБНОВЛЕНИЕ ВЕСОВ КОНТРОЛЛЕРА"""
        if control_signals is None:
            return
       
        thoughts = self.data_manager.get_thoughts_slice()
        if len(thoughts) < 10:
            return
       
        recent_thoughts = thoughts[-10:]
        positive_count = sum(1 for t in recent_thoughts if t.get('type') == 'curiosity_insight')
        hard_negative_count = sum(1 for t in recent_thoughts if t.get('type') == 'hard_negative')
       
        success_rate = positive_count / len(recent_thoughts) if recent_thoughts else 0
        challenge_rate = hard_negative_count / len(recent_thoughts) if recent_thoughts else 0
       
        with torch.no_grad():
            for signal_name, param in self.control_weights.items():
                current_weight = param.item()
               
                if signal_name == "curiosity_threshold":
                    if success_rate < 0.2:
                        adjustment = 0.95
                    elif success_rate > 0.5:
                        adjustment = 1.05
                    else:
                        adjustment = 1.0
                       
                elif signal_name == "write_intensity":
                    if len(thoughts) > self.thoughts_threshold * 0.9:
                        adjustment = 0.9
                    elif len(thoughts) < self.thoughts_threshold * 0.3:
                        adjustment = 1.1
                    else:
                        adjustment = 1.0
                       
                elif signal_name == "exploration_bonus":
                    if challenge_rate < 0.2:
                        adjustment = 1.1
                    elif challenge_rate > 0.5:
                        adjustment = 0.95
                    else:
                        adjustment = 1.0
                       
                else:
                    adjustment = 1.0
               
                new_weight = current_weight * adjustment
                new_weight = max(0.5, min(2.0, new_weight))
               
                if abs(new_weight - current_weight) > 0.01:
                    param.data.fill_(new_weight)
       
        if self.step_count % 500 == 0:
            logger.debug(f"⚖️ Веса контроллера: " +
                        " ".join([f"{k}={v.item():.2f}" for k, v in self.control_weights.items()]))
# ===================================================================
# 🔥 ОБНОВЛЕНИЕ ЛЮБОПЫТСТВА
# ===================================================================
    def update_curiosity(self, predicted_slot, actual_slot, confidence):
        """🔥 Обновляет любопытство на основе ошибок предсказания Linker'а"""
        if predicted_slot != actual_slot:
            if isinstance(actual_slot, torch.Tensor):
                slot_idx = actual_slot.item()
            else:
                slot_idx = actual_slot
               
            error_strength = 1.0 - confidence
            self.slot_curiosity[slot_idx] += error_strength
            self.prediction_errors.append(error_strength)
       
        self.slot_curiosity *= self.curiosity_decay
       
        if self.slot_curiosity.max() > 0.5:
            high_curiosity = torch.where(self.slot_curiosity > 0.5)[0]
            for slot in high_curiosity:
                current = self.centroid_memory.slot_utility[slot].item()
                self.centroid_memory.slot_utility[slot] = min(1.0, current + 0.05)
                
    def forward(self, hidden_states, slots=None, transformer_experience=None, control_signals=None):
        self.step_count += 1
       
        # ===========================================================
        # 🔥🔥🔥 ЗАЩИТА ОТ НЕПРАВИЛЬНОЙ РАЗМЕРНОСТИ
        # ===========================================================
        if hidden_states is not None:
            original_dim = hidden_states.dim()
            if original_dim == 2:
                hidden_states = hidden_states.unsqueeze(1)
                logger.debug(f"📊 hidden_states был 2D, преобразован в 3D: {hidden_states.shape}")
            elif original_dim != 3:
                raise ValueError(f"hidden_states должен быть 2D или 3D, получено {original_dim}D")
       
        # ===========================================================
        # 📊 СЧЕТЧИК ВЫЗОВОВ
        # ===========================================================
        if not hasattr(self, '_total_calls'):
            self._total_calls = 0
            self._useful_calls = 0
        self._total_calls += 1
       
        if not hasattr(self, '_last_candidates'):
            self._last_candidates = []
            self._candidate_hash_set = set()
       
        # ===========================================================
        # 🔥🔥🔥 ANTI-MUSH QUERY (как в Hebb v5.4)
        # ===========================================================
        model_device = next(self.parameters()).device
        if hidden_states is not None:
            hidden_states = hidden_states.to(model_device)
       
        if slots is not None:
            slots = slots.to(model_device)
       
        if transformer_experience is not None:
            transformer_experience = transformer_experience.to(model_device)
            if self.step_count % 1000 == 0:
                logger.debug(f"📦 Получен transformer_experience (шаг {self.step_count})")
       
        enriched_candidates = []
        current_batch_size = hidden_states.shape[0]
       
        # Проверяем writer state
        if self.mamba_writer_state is not None:
            if self.mamba_writer_state.device != model_device:
                self.mamba_writer_state = self.mamba_writer_state.to(model_device)
           
            if self.mamba_writer_state.shape[0] != current_batch_size:
                logger.debug(f"🔄 Сброс writer state: {self.mamba_writer_state.shape[0]} → {current_batch_size}")
                self.mamba_writer_state = None
       
        if self.mamba_reader_state is not None:
            if self.mamba_reader_state.device != model_device:
                self.mamba_reader_state = self.mamba_reader_state.to(model_device)
           
            if self.mamba_reader_state.shape[0] != current_batch_size:
                logger.debug(f"🔄 Сброс reader state: {self.mamba_reader_state.shape[0]} → {current_batch_size}")
                self.mamba_reader_state = None
       
        self._last_control_signals = control_signals or {}
        applied_controls = self.apply_control_signals(control_signals)
       
        # ===========================================================
        # 🔥🔥🔥 ВЫЧИСЛЯЕМ МОТИВАЦИЮ
        # ===========================================================
        surprise_value = getattr(self, '_last_surprise', None)
        if surprise_value is None:
            surprise_value = getattr(self, '_surprise_for_hebb', None)
        if surprise_value is None:
            surprise_value = getattr(self, '_surprise_value', None)
        if surprise_value is None:
            surprise_value = 0.0
       
        if isinstance(surprise_value, torch.Tensor):
            surprise_value = surprise_value.item()
       
        motivation = 0.5
        if hasattr(self, 'motivation_module'):
            motivation = self.motivation_module(
                hidden_states.mean(dim=1),           # текущее состояние модели
                surprise=surprise_value,
                curiosity_level=applied_controls.get('curiosity_level', 0.5)
            )
        
        # Передаём мотивацию дальше (в контроллер и gate)
        if control_signals is None:
            control_signals = {}
        
        # 🔥 БЕЗОПАСНОЕ ПРЕОБРАЗОВАНИЕ В СКАЛЯР
        if isinstance(motivation, torch.Tensor):
            control_signals['motivation'] = motivation.detach().item()
        else:
            control_signals['motivation'] = float(motivation)
       
        # 🔥 ANTI-MUSH QUERY: случайное окно внимания
        if hidden_states.dim() == 3:
            b, s, d = hidden_states.shape
           
            # 🔥 ANTI-MUSH: случайное окно внимания (точно как в Hebb)
            if self.training and s > 64:
                window_size = 64
                start_idx = torch.randint(0, s - window_size, (1,), device=model_device).item()
                focal_states = hidden_states[:, start_idx:start_idx + window_size, :]
               
                # Если есть attention_mask — используем его
                if hasattr(self, 'current_attention_mask') and self.current_attention_mask is not None:
                    mask_chunk = self.current_attention_mask[:, start_idx:start_idx + window_size].unsqueeze(-1).float()
                    query_vector = (focal_states * mask_chunk).sum(dim=1) / mask_chunk.sum(dim=1).clamp(min=1e-8)
                else:
                    query_vector = focal_states.mean(dim=1)
            else:
                # Короткие последовательности или инференс — обычное среднее
                if hasattr(self, 'current_attention_mask') and self.current_attention_mask is not None:
                    mask = self.current_attention_mask.unsqueeze(-1).float()
                    query_vector = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
                else:
                    query_vector = hidden_states.mean(dim=1)
           
            batch_query = query_vector.mean(dim=0) # для памяти оставляем среднее по батчу
           
        else:
            query_vector = hidden_states
            batch_query = hidden_states.mean(dim=0) if hidden_states.dim() > 1 else hidden_states
       
        batch_query = batch_query.to(model_device)
       
        # ===========================================================
        # 🔥 ЗАПРОС ПАМЯТИ (БЕЗ ПРИМЕНЕНИЯ GATE!)
        # ===========================================================
        memory_context, memory_info = self.query_memory(batch_query)
       
        # ===========================================================
        # 🔥 ИНИЦИАЛИЗИРУЕМ result ДО ВСЕХ ОПЕРАЦИЙ
        # ===========================================================
        result = {
            'insights': [],
            'reader_improvement': self._last_reader_improvement if hasattr(self, '_last_reader_improvement') else 0.0,
            'writer_delta': self._last_writer_delta if hasattr(self, '_last_writer_delta') else 0.0,
            'surprise': surprise_value,
            'step': self.step_count,
            'applied_controls': applied_controls,
            'memory_context': memory_context,
            'memory_info': memory_info,
            'meta_cognitive': {} # ← пустой словарь для мета-информации
        }
       
        # ===========================================================
        # 🔥 МЕТА-КОГНИТИВНЫЙ ЦИКЛ (обновлённый для v3.0)
        # ===========================================================
       
        # Получаем текущее состояние (вектор из памяти или запроса)
        current_state = memory_context if memory_context is not None else batch_query
       
        # Инициализируем переменные для мета-информации
        meta = None
        predicted_next = None
        reflection_log = None
        need_reflection = False
        meta_loss = None
        decisions = []
       
        if current_state is not None and current_state.numel() > 0:
            # Приводим к правильной размерности [batch, slot_dim]
            if current_state.dim() == 1:
                current_state = current_state.unsqueeze(0)
           
            # 1. Получаем предсказание и мета-информацию
            predicted_next, meta = self.meta_predictor(
                current_state,
                context=transformer_experience,
                temperature=self.reflection_temperature if self.training else 0.8,
                return_meta=True
            )
           
            # 2. 🔥 НОВАЯ ЛОГИКА: используем decision вместо need_*
            decisions = meta.get("decision", ["act"] * current_state.shape[0])
           
            # Проверяем, нужна ли рефлексия (любое решение кроме "act")
            need_reflection = any(d not in ["act"] for d in decisions)
           
            # ===========================================================
            # 🔥 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ МЕТА-КОГНИТИВНОГО ПРЕДИКТОРА
            # ===========================================================
            if self.step_count % 50 == 0:
                logger.info(f"🧠 МЕТА-СТАТУС (шаг {self.step_count}):")
                logger.info(f" Уверенность: {meta['confidence'].mean().item():.3f}")
                logger.info(f" Сомнение: {meta['doubt'].mean().item():.3f}")
                logger.info(f" Любопытство: {meta['curiosity'].mean().item():.3f}")
                logger.info(f" Решения: {set(decisions)}")
                logger.info(f" Нужна рефлексия: {need_reflection}")
               
                # Логируем распределение метрик по батчу
                if current_state.shape[0] > 1:
                    logger.info(f" Распределение confidence: min={meta['confidence'].min().item():.3f}, "
                               f"max={meta['confidence'].max().item():.3f}")
           
            # 3. 🔥 РЕФЛЕКСИЯ (исправленный вызов)
            if need_reflection and self.max_reflection_steps > 0 and not getattr(self, '_in_reflection', False):
                before_state = current_state.clone()
               
                refined_state, reflection_log = self.meta_predictor.reflect(
                    current_state,
                    context=transformer_experience,
                    max_steps=self.max_reflection_steps,
                )
               
                # Обновляем статистику
                if not hasattr(self, '_total_reflections'):
                    self._total_reflections = torch.tensor(0, device=model_device)
                if not hasattr(self, '_reflection_depth_avg'):
                    self._reflection_depth_avg = torch.tensor(0.0, device=model_device)
               
                self._total_reflections.add_(1)
                # 🔥 ИСПРАВЛЕНИЕ: используем "steps" вместо "steps_taken"
                steps_taken = reflection_log.get("steps", reflection_log.get("steps_taken", 0))
                self._reflection_depth_avg = 0.95 * self._reflection_depth_avg + 0.05 * steps_taken
               
                # Используем уточнённое состояние
                memory_context = refined_state
               
                # Забираем мета-лосс из предиктора
                if hasattr(self.meta_predictor, '_last_meta') and self.meta_predictor._last_meta is not None:
                    meta_loss = self.meta_predictor._last_meta.get('meta_loss', None)
                    if meta_loss is not None and isinstance(meta_loss, torch.Tensor) and meta_loss.requires_grad:
                        if self.step_count % 50 == 0:
                            logger.info(f"🧠 Мета-loss из рефлексии: {meta_loss.item():.6f}")
               
                # Логируем рефлексию
                if self.step_count % 100 == 0:
                    final_meta = reflection_log.get("final_meta", {})
                    final_conf = final_meta.get("confidence", torch.tensor(0.0)).mean().item() if final_meta else 0.0
                    logger.info(
                        f"🧠 РЕФЛЕКСИЯ: шагов={steps_taken}, "
                        f"уверенность: {meta['confidence'].mean().item():.3f} → {final_conf:.3f}, "
                        f"решение={decisions[0] if decisions else '?'}, "
                        f"глубина_avg={self._reflection_depth_avg.item():.2f}"
                    )
                   
                    # Дополнительное логирование деталей рефлексии
                    if 'delta_confidence' in reflection_log:
                        logger.info(f" Изменение уверенности: {reflection_log['delta_confidence']:.3f}")
                    if 'reasons' in reflection_log and reflection_log['reasons']:
                        logger.info(f" Причины: {reflection_log['reasons'][:2]}")
           
            # 4. Если рефлексии не было, но есть мета-лосс из обычного forward
            elif not need_reflection and hasattr(self.meta_predictor, '_last_meta') and self.meta_predictor._last_meta is not None:
                meta_loss = self.meta_predictor._last_meta.get('meta_loss', None)
                if meta_loss is not None and isinstance(meta_loss, torch.Tensor) and meta_loss.requires_grad:
                    if self.step_count % 50 == 0:
                        logger.info(f"🧠 Мета-loss из forward: {meta_loss.item():.6f}")
           
            # Логируем, если рефлексия была нужна, но не выполнена
            if need_reflection and (self.max_reflection_steps == 0 or getattr(self, '_in_reflection', False)):
                if self.step_count % 100 == 0:
                    logger.debug(f"⚠ Рефлексия пропущена: max_steps={self.max_reflection_steps}, "
                               f"_in_reflection={getattr(self, '_in_reflection', False)}")
           
            # Сохраняем мета-информацию в результат (обновлённый формат)
            if meta is not None:
                result['meta_cognitive'] = {
                    'confidence': meta["confidence"].mean().item(),
                    'doubt': meta["doubt"].mean().item(),
                    'curiosity': meta["curiosity"].mean().item(),
                    'readiness': meta["readiness"].mean().item(),
                    'need_recheck': meta.get("need_recheck", torch.tensor(0)).mean().item(),
                    'self_doubt': meta.get("self_doubt", torch.tensor(0)).mean().item(),
                    'novelty': meta.get("novelty", torch.tensor(0)).mean().item(),
                    'decisions': decisions[:5] if len(decisions) > 5 else decisions,
                    'reflection_applied': need_reflection and self.max_reflection_steps > 0,
                    'reflection_steps': steps_taken if (reflection_log and need_reflection) else 0,
                    'reflection_depth_avg': self._reflection_depth_avg.item() if hasattr(self, '_reflection_depth_avg') else 0.0
                }
               
                # Добавляем информацию после рефлексии, если есть
                if reflection_log and need_reflection and 'final_meta' in reflection_log:
                    result['meta_cognitive']['confidence_after'] = reflection_log['final_meta']["confidence"].mean().item()
                    result['meta_cognitive']['doubt_after'] = reflection_log['final_meta']["doubt"].mean().item()
       
        # ===========================================================
        # 🔥🔥🔥 ИЗЯЩНЫЙ MEMORY GATE v2.0 (с мотивацией)
        # ===========================================================
        if memory_context is not None and hidden_states is not None:
            # Вычисляем gate с передачей мотивации
            gate = self._compute_memory_gate(meta, motivation=motivation)

            # Приводим размерности
            if memory_context.dim() == 1:
                memory_context = memory_context.unsqueeze(0).unsqueeze(0)
            elif memory_context.dim() == 2:
                memory_context = memory_context.unsqueeze(1)

            if memory_context.shape[1] == 1 and hidden_states.shape[1] > 1:
                memory_context = memory_context.expand(-1, hidden_states.shape[1], -1)

            # Применяем gate
            memory_contribution = gate.unsqueeze(-1) * memory_context
            hidden_states = hidden_states + memory_contribution

            # ── Aux-loss для gate (в AdaptiveMemoryHeads.forward, после применения gate)
            if self.training and meta is not None:
                confidence = meta.get('confidence', torch.tensor(0.5, device=hidden_states.device))
                curiosity = meta.get('curiosity', torch.tensor(0.3, device=hidden_states.device))
                
                # Учитываем мотивацию в ideal_gate (ОТВЯЗАННУЮ от градиентов)
                mot_val = torch.tensor(motivation, device=hidden_states.device) if isinstance(motivation, float) else motivation
                mot_val_target = mot_val.detach()  # ← ОТВЯЗЫВАЕМ!
                
                ideal_gate = torch.clamp(
                    confidence.mean() * 0.7 + curiosity.mean() * 0.2 + mot_val_target * 0.1,
                    0.25, 0.85
                )
                
                # 🔥 ИСПРАВЛЕНИЕ: Игнорируем минус из старого чекпоинта
                safe_coef = torch.abs(self.gate_aux_coef)
                
                # 🔥 ЛОГИРУЕМ значения
                gate_val = gate.item() if hasattr(gate, 'item') else float(gate)
                ideal_val = ideal_gate.item() if hasattr(ideal_gate, 'item') else float(ideal_gate)
                mot_scalar = mot_val_target.item() if isinstance(mot_val_target, torch.Tensor) else float(mot_val_target)
                
                if self.step_count % 50 == 0:
                    logger.info(f"🔍 gate={gate_val:.4f}, ideal={ideal_val:.4f}, motivation={mot_scalar:.3f}, "
                               f"coef={safe_coef.item() if hasattr(safe_coef, 'item') else safe_coef:.4f}")
                
                # Используем безопасный coef
                gate_loss = F.mse_loss(gate, ideal_gate) * safe_coef
                result['gate_aux_loss'] = gate_loss

                # 🔥 АДАПТИВНОЕ ОБНОВЛЕНИЕ coef от опыта
                if self.training and meta is not None:
                    conf = meta.get('confidence', torch.tensor(0.5, device=hidden_states.device)).mean().item()
                    self._update_gate_aux_coef(
                        gate_loss_value=gate_loss.item(),
                        motivation=motivation,
                        meta_confidence=conf
                    )
            else:
                result['gate_aux_loss'] = torch.tensor(0.0, device=hidden_states.device)

            # 🔥 ДОБАВЛЯЕМ aux-loss ДЛЯ ВЕСОВ GATE (из _compute_memory_gate)
            if hasattr(self, '_gate_weights_aux_loss') and self._gate_weights_aux_loss is not None:
                result['gate_weights_aux_loss'] = self._gate_weights_aux_loss

            # Логирование
            if self.step_count % 300 == 0 and meta is not None:
                if isinstance(gate, torch.Tensor):
                    gate_val = gate.item() if gate.numel() == 1 else gate.mean().item()
                else:
                    gate_val = float(gate)
                
                readiness_val = meta.get('readiness', 0)
                if isinstance(readiness_val, torch.Tensor):
                    readiness_val = readiness_val.item() if readiness_val.numel() == 1 else readiness_val.mean().item()
                
                curiosity_val = meta.get('curiosity', 0)
                if isinstance(curiosity_val, torch.Tensor):
                    curiosity_val = curiosity_val.item() if curiosity_val.numel() == 1 else curiosity_val.mean().item()
                
                logger.info(f"🚪 Memory Gate: {gate_val:.4f} "
                           f"(readiness={readiness_val:.3f}, "
                           f"curiosity={curiosity_val:.3f}, "
                           f"motivation={motivation:.3f})")
       
        if self.step_count % 1000 == 0 and query_vector.shape[0] > 1:
            logger.debug(f"📊 Память: батч={query_vector.shape[0]}, "
                        f"норма запроса={torch.norm(batch_query):.3f}")
       
        if self.step_count % 200 == 0:
            self._log_memory_status()
       
        with self._model_lock:
            self._check_and_trigger_sleep_async()
           
            if self.step_count % 100 == 0:
                self._maintain_notebook_balance()
               
                if control_signals and self.step_count % 200 == 0:
                    key_signals = {k: v for k, v in applied_controls.items()
                                  if k in ["curiosity_threshold", "write_intensity", "exploration_bonus"]}
                    logger.debug(f"🎛️ Сигналы контроллера: {key_signals}")
           
            batch_size = hidden_states.shape[0]
           
            if self.mamba_writer_state is None or self.mamba_writer_state.shape[0] != batch_size:
                self._reset_mamba_states(batch_size, device=model_device)
            else:
                self._clamp_states_norm(max_norm=getattr(self.config, 'sleep_state_norm_max', 10.0))
           
            # 🔥 Добавляем мета-loss в результат (если есть) - БЕЗ буфера!
            if meta_loss is not None and isinstance(meta_loss, torch.Tensor) and meta_loss.requires_grad:
                result['meta_loss'] = meta_loss
                if self.step_count % 50 == 0:
                    logger.info(f"📉 Мета-loss добавлен в результат: {meta_loss.item():.6f}")
           
            # Сохраняем предсказание для следующего шага (если нужно для обучения)
            if predicted_next is not None:
                self._last_predicted_next = predicted_next
           
            if slots is not None and slots.shape[0] > 0:
                slots = slots.to(model_device)
                improved_slots = self._process_mamba_reader(slots, hidden_states)
                result['improved_slots'] = improved_slots
           
            if hidden_states.shape[1] > 0:
                num_candidates = min(5, hidden_states.shape[1])
                indices = torch.linspace(0, hidden_states.shape[1]-1, num_candidates, dtype=torch.long, device=model_device)
               
                for idx in indices:
                    if hidden_states.dim() == 3:
                        candidate_slice = hidden_states[:, idx, :]
                    else:
                        candidate_slice = hidden_states
                        break
                   
                    if candidate_slice.shape[0] > 1:
                        candidate = candidate_slice.mean(dim=0)
                    else:
                        candidate = candidate_slice.squeeze(0)
                    enriched_candidates.append(candidate)
           
            # 🔥 Передаем mood_influence из control_signals
            mood_influence = 0.0
            if control_signals:
                mood_influence = control_signals.get('mood_influence', 0.0)
           
            writer_insight = self._process_mamba_writer(
                enriched_candidates,
                hidden_states,
                mood_influence=mood_influence
            )
           
            if writer_insight is not None:
                result['insights'].append(writer_insight)
               
                # 🔥 Обновляем surprise из writer'а (на случай, если он изменился)
                updated_surprise = getattr(self, '_last_surprise', surprise_value)
                if isinstance(updated_surprise, torch.Tensor):
                    updated_surprise = updated_surprise.item()
                result['surprise'] = updated_surprise
           
            if self.step_count % 200 == 0:
                self._report_status_with_gpu_monitoring()
                if control_signals:
                    self._log_controller_influence(applied_controls)
               
                # 🔥 Логируем surprise для отладки
                logger.info(f"🎯 AdaptiveMemoryHeads[surprise]: {result['surprise']:.4f}")
               
                # 🔥 Логируем мета-когнитивную информацию, если есть
                if 'meta_cognitive' in result and result['meta_cognitive']:
                    mc = result['meta_cognitive']
                    logger.info(f"🧠 Мета-когниция: conf={mc.get('confidence', 0):.3f}, "
                               f"doubt={mc.get('doubt', 0):.3f}, "
                               f"curiosity={mc.get('curiosity', 0):.3f}, "
                               f"decisions={mc.get('decisions', [])[:2]}, "
                               f"reflection={mc.get('reflection_steps', 0)} steps")
               
                # 🔥 Логируем мета-loss, если есть
                if 'meta_loss' in result and result['meta_loss'] is not None:
                    logger.info(f"📉 Мета-loss в результате: {result['meta_loss'].item():.6f}")
               
                # 🔥 Логируем состояние memory gate (теперь динамический)
                #if 'gate_aux_loss' in result and result['gate_aux_loss'].item() > 0:
                    #logger.info(f"🚪 Memory gate aux loss: {result['gate_aux_loss'].item():.6f}")
           
            # 🔥 КОНТРАБАНДА ГРАДИЕНТОВ ПРЕДИКТОРА
            if hasattr(self, '_predictor_loss') and self._predictor_loss is not None:
                if 'gate_aux_loss' in result and isinstance(result['gate_aux_loss'], torch.Tensor):
                    result['gate_aux_loss'] = result['gate_aux_loss'] + self._predictor_loss
                else:
                    result['gate_aux_loss'] = self._predictor_loss
                self._predictor_loss = None
                
                if self.step_count % 50 == 0:
                    logger.debug(f"📊 Predictor loss добавлен в gate_aux_loss")
            
            # 🔥 Логируем состояние memory gate
            #if 'gate_aux_loss' in result and result['gate_aux_loss'].item() > 0:
                #logger.info(f"🚪 Memory gate aux loss: {result['gate_aux_loss'].item():.6f}")
            
            return result
            
# ===================================================================
# ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# ===================================================================
    def pre_generation_hook(self):
        """Вызывается ПЕРЕД любой генерацией"""
        self._generation_count += 1
        self.is_generating = True
        logger.debug(f"🔴 Генерация начата (count={self._generation_count})")
        
    def post_generation_hook(self):
        """Вызывается ПОСЛЕ любой генерации"""
        self._generation_count -= 1
        if self._generation_count <= 0:
            self.is_generating = False
            self._generation_count = 0
            if self.sleep_postponed > 0:
                logger.info(f"💤 После генерации: было {self.sleep_postponed} отложенных снов")
                self.sleep_postponed = 0
            logger.debug(f"🟢 Генерация завершена")
            
    def _check_and_set_sleep_flags(self):
        with self._sleep_lock:
            if self.is_sleeping or self.pending_sleep or self._sleep_event.is_set():
                return False
           
            thoughts_count = self.data_manager.get_thought_count()
           
            if thoughts_count < self.thoughts_threshold:
                return False
           
            self.pending_sleep = True
            self.is_sleeping = True
            self._sleep_event.set()
            return True
   
    def _calculate_adaptive_threshold_smooth(self):
        base_threshold = self.curiosity_threshold
       
        if self.mamba_writer_ema < 0.05:
            activity_factor = 0.6
        elif self.mamba_writer_ema < 0.1:
            activity_factor = 0.8
        elif self.mamba_writer_ema > 0.4:
            activity_factor = 1.2
        else:
            activity_factor = 1.0
       
        thoughts = self.data_manager.get_thought_count()
        notebook_ratio = thoughts / self.config.notebook_size
       
        if notebook_ratio > 0.9:
            notebook_factor = 0.9
        elif notebook_ratio < 0.2:
            notebook_factor = 0.7
        else:
            notebook_factor = 0.7 + 0.3 * (notebook_ratio - 0.2) / 0.7
       
        steps_since_sleep = self.step_count - self._last_sleep_step
        if steps_since_sleep > 1000:
            time_factor = 0.9
        else:
            time_factor = 1.0 - 0.1 * (steps_since_sleep / 1000)
       
        threshold = base_threshold * activity_factor * notebook_factor * time_factor
       
        return max(0.05, min(0.25, threshold))
   
    def _reset_mamba_states(self, batch_size=1, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
       
        if device.type == 'meta':
            self.mamba_writer_state = None
            self.mamba_reader_state = None
            return
        init_scale = 0.005
       
        if hasattr(self, 'mamba_writer'):
            self.mamba_writer_state = torch.randn(
                batch_size,
                self.mamba_writer.inner_dim,
                self.mamba_state_size,
                device=device
            ) * init_scale
       
        if hasattr(self, 'mamba_reader'):
            self.mamba_reader_state = torch.randn(
                batch_size,
                self.mamba_reader.inner_dim,
                self.mamba_state_size,
                device=device
            ) * init_scale
       
        logger.debug(f"🔄 Состояния Mamba инициализированы (scale={init_scale})")
   
    def _add_predictor_forgetting(self):
        forget_ratio = 0.1
       
        with torch.no_grad():
            for layer in self.predictor:
                if isinstance(layer, nn.Linear):
                    weight_std = layer.weight.std().item()
                    if weight_std > 1e-8:
                        noise = torch.randn_like(layer.weight) * forget_ratio * weight_std
                        layer.weight.add_(noise)
                   
                    if layer.bias is not None:
                        bias_std = layer.bias.std().item()
                        if bias_std > 1e-8:
                            bias_noise = torch.randn_like(layer.bias) * forget_ratio * bias_std
                            layer.bias.add_(bias_noise)
       
        self.predictor_forget_counter += 1
        logger.info(f"🧹 Predictor частично забыт ({forget_ratio*100}%), счетчик: {self.predictor_forget_counter}")
   
    def _process_mamba_reader(self, retrieved_vectors, context_vectors):
        try:
            if retrieved_vectors is None:
                return None
               
            original_dim = retrieved_vectors.dim()
           
            if retrieved_vectors.dim() == 2:
                mamba_input = retrieved_vectors.unsqueeze(0)
            elif retrieved_vectors.dim() == 3:
                mamba_input = retrieved_vectors
            else:
                logger.warning(f"⚠ Неподдерживаемая размерность: {retrieved_vectors.dim()}")
                return retrieved_vectors
           
            if context_vectors is not None and context_vectors.dim() == 3:
                context_pooled = context_vectors.mean(dim=1, keepdim=True)
               
                if mamba_input.shape[0] == context_pooled.shape[0]:
                    context_expanded = context_pooled.expand(-1, mamba_input.shape[1], -1)
                    combined_input = torch.cat([mamba_input, context_expanded], dim=1)
                   
                    refined_output, new_state, *_ = self.mamba_reader(
                        combined_input,
                        hidden_state=self.mamba_reader_state
                    )
                   
                    if new_state is not None:
                        new_state_detached = new_state.detach()
                       
                        if torch.isnan(new_state_detached).any():
                            logger.warning("⚠ NaN in Reader State! Resetting.")
                            self.mamba_reader_state = None
                        else:
                            state_norm = torch.norm(new_state_detached)
                            if state_norm > 15.0:
                                scale = 15.0 / (state_norm + 1e-8)
                                new_state_detached = new_state_detached * scale
                           
                            self.mamba_reader_state = new_state_detached
                   
                    num_tokens = mamba_input.shape[1]
                    if refined_output.shape[1] >= num_tokens:
                        refined_retrieved = refined_output[:, :num_tokens, :]
                       
                        if original_dim == 2:
                            refined_retrieved = refined_retrieved.squeeze(0)
                       
                        original_norm = torch.norm(retrieved_vectors).item()
                        refined_norm = torch.norm(refined_retrieved).item()
                        if original_norm > 1e-8:
                            improvement = abs(refined_norm - original_norm) / original_norm
                            self._last_reader_improvement = improvement
                       
                        return refined_retrieved
           
            return retrieved_vectors
           
        except Exception as e:
            logger.error(f"❌ Reader ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return retrieved_vectors
            
    def _check_and_trigger_sleep_async(self):
        """
        🔥 СИНХРОННАЯ фаза сна - вызывается из forward, но не блокирует долго.
        Использует копии весов, чтобы не мешать градиентам.
        """
        try:
            # 🔥 ГЛАВНАЯ ЗАЩИТА: НЕ СПИМ ВО ВРЕМЯ ГЕНЕРАЦИИ!
            if self.is_generating:
                self.sleep_postponed += 1
                if self.sleep_postponed % 50 == 0:
                    logger.debug(f"⏰ Сон отложен {self.sleep_postponed}-й раз: идет генерация")
                return False
           
            if self.step_count % 20 != 0:
                return False
           
            with self._sleep_lock:
                if self.is_sleeping or self.pending_sleep or self._sleep_event.is_set():
                    return False
               
                thoughts_count = self.data_manager.get_thought_count()
               
                if thoughts_count < self.thoughts_threshold:
                    return False
               
                self.pending_sleep = True
                self.is_sleeping = True
                self._sleep_event.set()
           
            thoughts_count = self.data_manager.get_thought_count()
            notebook_size = self.config.notebook_size
           
            fill_percent = (thoughts_count / notebook_size) * 100
            if fill_percent >= 90:
                status = "🔴 ПЕРЕПОЛНЕН"
            elif fill_percent >= 75:
                status = "🟡 ЗАПОЛНЕН"
            else:
                status = f"{fill_percent:.0f}%"
               
            logger.info(f"💤 Запуск СИНХРОННОГО сна: {thoughts_count}/{notebook_size} ({status})")
           
            # 🔥 СИНХРОННЫЙ вызов (без отдельного потока!)
            self._background_dream_wrapper()
           
            self._last_sleep_step = self.step_count
            return True
           
        except Exception as e:
            logger.error(f"❌ Ошибка триггера сна: {e}")
            with self._sleep_lock:
                self.pending_sleep = False
                self.is_sleeping = False
                self._sleep_event.clear()
            return False
            
    def _background_dream_wrapper(self):
        """🔥 Обертка для фонового режима сновидений с ПОЛНОЙ ИЗОЛЯЦИЕЙ"""
        import copy
        import gc
       
        # 🔥 ДВОЙНАЯ ПРОВЕРКА ПЕРЕД ЗАПУСКОМ
        if self.is_generating:
            logger.info("⏰ Сон отложен: началась генерация")
            with self._sleep_lock:
                self.pending_sleep = False
                self.is_sleeping = False
                self._sleep_event.clear()
            return
       
        try:
            # 1. Копируем ВЕСЬ модуль (не только state_dict!)
            with self._model_lock:
                logger.info("📋 Создание полной копии Mamba для сна...")
               
                writer_copy = ThaliaMambaHead(
                    self.mamba_writer.n_embd,
                    d_state=self.mamba_writer.d_state,
                    chunk_size=getattr(self.mamba_writer, 'chunk_size', 32)
                )
                reader_copy = ThaliaMambaHead(
                    self.mamba_reader.n_embd,
                    d_state=self.mamba_reader.d_state,
                    chunk_size=getattr(self.mamba_reader, 'chunk_size', 32)
                )
               
                writer_copy.load_state_dict(copy.deepcopy(self.mamba_writer.state_dict()))
                reader_copy.load_state_dict(copy.deepcopy(self.mamba_reader.state_dict()))
               
                device = next(self.mamba_writer.parameters()).device
                writer_copy = writer_copy.to(device)
                reader_copy = reader_copy.to(device)
               
                thoughts = copy.deepcopy(self.data_manager.get_thoughts_slice())
           
            # 2. Включаем train mode для копий
            writer_copy.train()
            reader_copy.train()
           
            # 3. Запускаем сон на КОПИЯХ
            self._internal_dream_phase_with_copies(writer_copy, reader_copy, thoughts)
           
            # 4. После успешного сна - обновляем оригиналы
            with self._model_lock:
                logger.info("📋 Синхронизация весов после сна...")
                self.mamba_writer.load_state_dict(writer_copy.state_dict())
                self.mamba_reader.load_state_dict(reader_copy.state_dict())
               
                self.mamba_writer_ema = 0.92 * self.mamba_writer_ema + 0.08 * getattr(self, '_last_sleep_loss', 0.1)
               
                logger.info("✅ Веса Mamba обновлены после сна")
           
        except Exception as e:
            logger.error(f"❌ Ошибка в фоновом сне: {e}")
            import traceback
            logger.error(traceback.format_exc())
           
        finally:
            with self._sleep_lock:
                self.pending_sleep = False
                self.is_sleeping = False
                self._sleep_event.clear()
           
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _internal_dream_phase_with_copies(self, writer_copy, reader_copy, thoughts):
        """🔥 Сон на полных копиях модели - безопасно для градиентов"""
        import time
        start_time = time.time()
       
        try:
            positive_thoughts = [t for t in thoughts if t.get('type') == 'curiosity_insight']
            hard_negative_thoughts = [t for t in thoughts if t.get('type') == 'hard_negative']
            gaslight_thoughts = [t for t in thoughts if t.get('is_gaslight', False)]
           
            logger.info(f"💤 СОН НА КОПИЯХ: {len(positive_thoughts)} 👍, "
                       f"{len(hard_negative_thoughts)} ⚔️, {len(gaslight_thoughts)} 🔥")
           
            if len(positive_thoughts) < 10 or (len(hard_negative_thoughts) + len(gaslight_thoughts)) < 30:
                logger.warning(f"💤 Недостаточно данных")
                return
           
            device = next(writer_copy.parameters()).device
           
            # ===========================================================
            # 🔥🔥🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
            # ===========================================================
            # 1. Включаем режим обучения для копий
            writer_copy.train()
            reader_copy.train()
           
            # 2. ГАРАНТИРУЕМ, что все параметры требуют градиентов
            for param in writer_copy.parameters():
                param.requires_grad = True
            for param in reader_copy.parameters():
                param.requires_grad = True
           
            # 3. Создаем оптимизатор ТОЛЬКО ПОСЛЕ включения градиентов
            sleep_optimizer = torch.optim.AdamW(
                list(writer_copy.parameters()) + list(reader_copy.parameters()),
                lr=2e-5,
                weight_decay=0.15,
                betas=(0.9, 0.95)
            )
           
            # ===========================================================
            # 🔥 ЭТАП 1: Обучение на копиях
            # ===========================================================
            best_loss, sleep_stats = self._perform_contrastive_sleep_on_copies(
                positive_thoughts,
                hard_negative_thoughts,
                gaslight_thoughts,
                writer_copy,
                reader_copy,
                sleep_optimizer,
                device
            )
           
            self._last_sleep_loss = best_loss
            self.sleep_cycles += 1
           
            # ===========================================================
            # 🔥🔥🔥 АДАПТИВНЫЙ РОСТ MEMORY GATE (умный)
            # ===========================================================
            with torch.no_grad():
                current_gate = self.memory_gate_ema.item()          # ← используем EMA
                
                if best_loss < 0.5:
                    # Хороший сон → плавно открываем gate
                    new_gate = min(0.85, current_gate + 0.035)
                    logger.info(f"🚪 Memory gate повышен после сна: {current_gate:.4f} → {new_gate:.4f} (loss={best_loss:.4f})")
                elif best_loss < 0.8:
                    # Средний сон → лёгкое увеличение
                    new_gate = min(0.85, current_gate + 0.012)
                else:
                    # Плохой сон → чуть прикрываем gate
                    new_gate = max(0.08, current_gate - 0.018)
                    logger.info(f"🚪 Memory gate понижен после сна: {current_gate:.4f} → {new_gate:.4f} (loss={best_loss:.4f})")
                
                # Обновляем logit (чтобы _compute_memory_gate продолжал работать)
                self.memory_gate_logit.data.fill_(torch.logit(torch.tensor(new_gate, device=self.memory_gate_logit.device)))
                
                # Обновляем EMA
                self.memory_gate_ema.fill_(new_gate)
           
            # ===========================================================
            # 🔥🔥🔥 ЭТАП 2: Обслуживание центроидной памяти (С БЛОКИРОВКОЙ!)
            # ===========================================================
            if hasattr(self, 'centroid_memory'):
                try:
                    logger.info("🧱 ЗАПУСК ОБСЛУЖИВАНИЯ ЦЕНТРОИДНОЙ ПАМЯТИ...")
                   
                    # 🔥🔥🔥 ИСПРАВЛЕНИЕ: правильное получение устройства
                    if hasattr(self.centroid_memory, 'centroids'):
                        centroid_device = self.centroid_memory.centroids.device
                    else:
                        centroid_device = device
                   
                    if centroid_device != device:
                        logger.info(f"🔄 Перемещаю центроидную память с {centroid_device} на {device}")
                        self.centroid_memory = self.centroid_memory.to(device)
                   
                    # 🔥🔥🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: принудительный сброс кэша и синхронизация
                    with self._model_lock:
                        if hasattr(self.centroid_memory, '_ensure_device_consistency'):
                            self.centroid_memory._ensure_device_consistency()
                            logger.debug("✅ Устройства центроидной памяти синхронизированы")
                       
                        # 🔥🔥🔥 ВАЖНО: пересоздаём оптимизатор linker'а на правильном устройстве
                        if hasattr(self.centroid_memory, '_ensure_linker_optimizer_device'):
                            self.centroid_memory._ensure_linker_optimizer_device(device)
                       
                        maint_start = time.time()
                        maintenance_stats = self.centroid_memory.run_maintenance()
                        maint_time = time.time() - maint_start
                       
                        logger.info(f"🧹 Центроидная память: merged={maintenance_stats['merged']}, "
                                   f"split={maintenance_stats['split']}, evicted={maintenance_stats['evicted']} "
                                   f"({maint_time:.2f} сек)")
                       
                        if maintenance_stats['merged'] + maintenance_stats['split'] + maintenance_stats['evicted'] > 0:
                            logger.info("🔍 Анализ изменений в центроидной памяти:")
                            self.centroid_memory.analyze_transitions()
                           
                            if maintenance_stats['merged'] > 0 or maintenance_stats['split'] > 0:
                                self.centroid_memory.show_hubs(top_k=5)
                   
                    # Блокировка автоматически снимается после выхода из with
                   
                except Exception as e:
                    logger.error(f"❌ Ошибка обслуживания центроидной памяти: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
           
            # ===========================================================
            # 🔥🔥🔥 ЭТАП 3: Работа с данными (БЕЗ ГРАДИЕНТОВ, С БЛОКИРОВКОЙ ДЛЯ ДАННЫХ)
            # ===========================================================
            with torch.no_grad():
                recon_bonus = 0.0
                recovery_info = {}
                if 'recovery_stats' in sleep_stats:
                    recovery_info = sleep_stats['recovery_stats']
                    recon_bonus = recovery_info.get('bonus', 0.0)
                   
                    if recon_bonus > 0:
                        logger.info(f"🎁 Получен бонус за восстановление: {recon_bonus:.3f}")
                       
                        success_rate = recovery_info.get('success_rate', 0.0)
                        if success_rate < 0.3:
                            logger.warning(f"⚠ ВНИМАНИЕ: recovery_success_rate низкий ({success_rate:.1%})")
                       
                        self._curiosity_motivation += 0.08 * recon_bonus
                        self._curiosity_motivation = min(1.0, self._curiosity_motivation)
                        logger.info(f"📈 Новая мотивация: {self._curiosity_motivation:.3f}")
                       
                        evolved_count = recovery_info.get('evolved_count', 0)
                        if evolved_count > 0:
                            logger.info(f"🧬 Эволюционировало мыслей: {evolved_count}")
                           
                        if 'recovery_level' in recovery_info:
                            old_level = self.recovery_level
                            self.recovery_level = recovery_info['recovery_level']
                            if self.recovery_level != old_level:
                                logger.info(f"🎯 Recovery level изменен: {old_level} → {self.recovery_level}")
               
                if self.sleep_cycles % 3 == 0 and hasattr(self, '_add_predictor_forgetting'):
                    self._add_predictor_forgetting()
               
                self.mamba_writer_ema = 0.92 * self.mamba_writer_ema + 0.08 * best_loss
                logger.info(f"💤 Контрастный сон завершен | Best Loss: {best_loss:.4f}")
               
                # Блокируем доступ к данным при их модификации
                with self.data_manager._lock:
                    thoughts = self.data_manager.thought_chains
                    original_count = len(thoughts)
                   
                    for t in thoughts:
                        t['age'] = t.get('age', 0) + 1
               
                if hasattr(self, '_deduplicate_notebook'):
                    thoughts = self._deduplicate_notebook(thoughts, similarity_threshold=0.96)
               
                keep_ratio = 0.35
                min_keep = 120
                keep_count = max(min_keep, int(self.thoughts_threshold * keep_ratio))
               
                logger.info(f"🧹 Динамическая очистка: target={keep_count} мыслей "
                           f"({keep_ratio*100:.0f}% от {self.thoughts_threshold}, сейчас {len(thoughts)})")
                if len(thoughts) > keep_count:
                    positive_thoughts = [t for t in thoughts if t.get('type') == 'curiosity_insight'
                                        and not t.get('is_hard_negative') and not t.get('is_gaslight')]
                    hard_negative_thoughts = [t for t in thoughts if t.get('type') == 'hard_negative'
                                             or t.get('is_hard_negative')]
                    gaslight_thoughts = [t for t in thoughts if t.get('is_gaslight')]
                   
                    sorted_thoughts = self._intelligent_pruning(
                        thoughts, keep_count, positive_thoughts, hard_negative_thoughts, gaslight_thoughts
                    )
                    logger.info(f"🧹 ОЧИСТКА: {len(thoughts)} → {len(sorted_thoughts)} мыслей")
                else:
                    sorted_thoughts = thoughts
                    logger.info(f"🧹 Очистка не требуется: {len(thoughts)} <= {keep_count}")
               
                # Блокируем при записи обновленного списка
                with self.data_manager._lock:
                    self.data_manager.thought_chains = sorted_thoughts
                try:
                    self.data_manager._save_thoughts_async()
                    logger.info(f"💾 Блокнот сохранён (асинхронно): {original_count} → {len(sorted_thoughts)} мыслей")
                except Exception as e:
                    logger.warning(f"⚠ Ошибка запуска асинхронного сохранения: {e}")
                if self.sleep_cycles % 1 == 0:
                    try:
                        veteran_age_threshold = getattr(self.config, 'veteran_age_threshold', 7)
                        veteran_delta_threshold = getattr(self.config, 'veteran_delta_threshold', 0.65)
                        max_veterans_per_sleep = getattr(self.config, 'max_veterans_per_sleep', 5)
                       
                        thoughts = self.data_manager.thought_chains
                        positive_thoughts = [t for t in thoughts if t.get('type') == 'curiosity_insight']
                       
                        available_veterans = [
                            t for t in positive_thoughts
                            if t.get('age', 0) >= veteran_age_threshold
                            and t.get('delta', 0) >= veteran_delta_threshold
                        ]
                       
                        if available_veterans:
                            sorted_veterans = sorted(
                                available_veterans,
                                key=lambda x: x.get('delta', 0),
                                reverse=True
                            )[:max_veterans_per_sleep]
                           
                            consolidated_count = 0
                            for i, vet in enumerate(sorted_veterans):
                                snapshot = vet.get('snapshot')
                                delta = vet.get('delta', 0.0)
                               
                                if snapshot is not None:
                                    try:
                                        if isinstance(snapshot, list):
                                            model_device = next(self.parameters()).device
                                            vector = torch.tensor(snapshot, dtype=torch.float32, device=model_device)
                                        else:
                                            vector = snapshot.to(model_device)
                                       
                                        # 🔥 Консолидация ветеранов тоже требует блокировки центроидной памяти
                                        with self._model_lock:
                                            result = self.centroid_memory.update_slot_centroid(
                                                thought_vector=vector,
                                                thought_id=vet.get('step', 0),
                                                thought_delta=delta
                                            )
                                       
                                        if result['action'] != 'none':
                                            consolidated_count += 1
                                           
                                    except Exception as e:
                                        logger.debug(f"⚠ Ошибка консолидации ветерана: {e}")
                           
                            if consolidated_count > 0:
                                logger.info(f"🎖️ Консолидация ветеранов: {consolidated_count}/{len(sorted_veterans)}")
                       
                        if len(self.data_manager.thought_chains) > 0 and self.step_count % 1000 == 0:
                            # Тестовый запрос тоже требует блокировки
                            with self._model_lock:
                                dummy_query = torch.randn(self.centroid_memory.slot_dim,
                                                         device=next(self.parameters()).device)
                                self.centroid_memory.query(dummy_query, top_k=3)
                       
                    except Exception as e:
                        logger.error(f"❌ Ошибка консолидации ветеранов: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                self.curiosity.reset()
              
                if hasattr(self, '_log_detailed_cleaning_stats'):
                    self._log_detailed_cleaning_stats(positive_thoughts, hard_negative_thoughts, gaslight_thoughts)
           
            # ===========================================================
            # 🔥🔥🔥 СИНХРОНИЗАЦИЯ ЯКОРЕЙ С ЦЕНТРОИДАМИ ПОСЛЕ СНА
            # ===========================================================
            if hasattr(self, 'sync_anchors_with_centroids'):
                self.sync_anchors_with_centroids()
           
            # ===========================================================
            # 🔥🔥🔥 ВАЖНО: Возвращаем копии в исходный режим (опционально)
            # ===========================================================
            writer_copy.eval()
            reader_copy.eval()
           
            # Опционально: заморозить градиенты для экономии памяти
            for param in writer_copy.parameters():
                param.requires_grad = False
            for param in reader_copy.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"❌ FATAL SLEEP ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
       
        finally:
            logger.info(f"✨ Сон на копиях завершен за {time.time() - start_time:.1f} сек")

            
    def _perform_contrastive_sleep_on_copies(self, positive_thoughts, hard_negative_thoughts,
                                                     gaslight_thoughts, writer_copy, reader_copy,
                                                     optimizer, device):
        """🔥 Контрастный сон на ПОЛНЫХ КОПИЯХ моделей"""
       
        logger.info(f"💤 Контрастный сон на копиях: {len(positive_thoughts)} 👍, "
                   f"{len(hard_negative_thoughts)} ⚔️, {len(gaslight_thoughts)} 🔥")
       
        total_negatives = len(hard_negative_thoughts) + len(gaslight_thoughts)
        if total_negatives < self.contrastive_min_negative:
            logger.warning(f"💤 Недостаточно негативов: {total_negatives} < {self.contrastive_min_negative}, пропускаем сон")
            return float('inf'), {'recovery_stats': {}}
       
        # 🔥 ИСПРАВЛЕНИЕ 2: Используем персистентный лосс вместо создания нового
        contrastive_loss = self.contrastive_loss_fn.to(device)
       
        def prepare_contrastive_pairs(num_pairs=24):
            all_pos = []
            all_neg = []
            pair_types = []
           
            for _ in range(num_pairs):
                if not positive_thoughts:
                    break
                   
                pos_thought = random.choice(positive_thoughts)
               
                negative_types = []
                weights = []
               
                if hard_negative_thoughts:
                    negative_types.append('hard_negative')
                    weights.append(0.6)
               
                if gaslight_thoughts:
                    negative_types.append('gaslight')
                    weights.append(0.4)
               
                if not negative_types:
                    continue
               
                chosen_type = random.choices(negative_types, weights=weights, k=1)[0]
               
                if chosen_type == 'hard_negative':
                    matched_negatives = [t for t in hard_negative_thoughts
                                       if t.get('original_step') == pos_thought.get('step')]
                   
                    if matched_negatives:
                        neg_thought = random.choice(matched_negatives)
                    else:
                        neg_thought = random.choice(hard_negative_thoughts)
               
                else: # gaslight
                    neg_thought = random.choice(gaslight_thoughts)
               
                try:
                    pos_snapshot = torch.tensor(pos_thought["snapshot"], dtype=torch.float32, device=device)
                    pos_snapshot = F.normalize(pos_snapshot, dim=-1)
                   
                    neg_snapshot = torch.tensor(neg_thought["snapshot"], dtype=torch.float32, device=device)
                    neg_snapshot = F.normalize(neg_snapshot, dim=-1)
                   
                    all_pos.append(pos_snapshot.unsqueeze(0))
                    all_neg.append(neg_snapshot.unsqueeze(0))
                    pair_types.append(chosen_type)
                   
                except Exception as e:
                    continue
           
            if not all_pos:
                return None, None, None
               
            return (torch.cat(all_pos, dim=0),
                    torch.cat(all_neg, dim=0),
                    pair_types)
       
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        training_stats = {
            'avg_contrastive_loss': 0.0,
            'avg_recon_loss': 0.0,
            'avg_similarity': 0.0,
            'avg_margin': 0.0,
            'patience_counter': 0,
            'epochs_completed': 0,
            'loss_history': []
        }
       
        # ===========================================================
        # 🔥🔥🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
        # Обёртываем ВЕСЬ цикл обучения в torch.enable_grad()
        # ===========================================================
        with torch.enable_grad():
            for epoch in range(15):
                try:
                    optimizer.zero_grad()
                   
                    pos_batch, neg_batch, pair_types = prepare_contrastive_pairs(num_pairs=16)
                    if pos_batch is None:
                        continue
                   
                    # Forward через writer
                    combined = torch.cat([pos_batch, neg_batch], dim=0).unsqueeze(1)
                    writer_output, _, _ = writer_copy(
                        combined,
                        hidden_state=None,
                        training_mode=True
                    )
                   
                    # Forward через reader
                    reader_output, reader_state, _ = reader_copy(
                        writer_output,
                        hidden_state=None,
                        training_mode=True
                    )
                   
                    batch_size = pos_batch.shape[0]
                    out_pos = reader_output[:batch_size]
                    out_neg = reader_output[batch_size:]
                   
                    loss_contrast, stats = contrastive_loss(
                        good_output=out_pos,
                        good_target=pos_batch.unsqueeze(1),
                        bad_output=out_neg,
                        bad_target=neg_batch.unsqueeze(1),
                        include_reconstruction=True
                    )
                   
                    total_loss = loss_contrast
                    recon_loss_raw = stats.get('loss_reconstruction', 0.0)
                   
                    if recon_loss_raw > 0:
                        adjusted_recon = recon_loss_raw * 2.0
                        total_loss = total_loss + adjusted_recon
                       
                        if 'avg_recon_loss' not in training_stats:
                            training_stats['avg_recon_loss'] = 0.0
                        training_stats['avg_recon_loss'] = (
                            training_stats['avg_recon_loss'] * epoch + adjusted_recon
                        ) / (epoch + 1)
                   
                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                        optimizer.zero_grad()
                        continue
                   
                    # ===========================================================
                    # Автоэнкодерная задача - градиенты уже включены через внешний блок
                    # ===========================================================
                    autoencoder_loss = 0.0
                    if len(positive_thoughts) >= 4:
                        auto_samples = random.sample(positive_thoughts, min(4, len(positive_thoughts)))
                       
                        for sample in auto_samples:
                            vec = torch.tensor(sample['snapshot'], dtype=torch.float32, device=device)
                            vec = F.normalize(vec, dim=-1)
                           
                            vec_in = vec.unsqueeze(0).unsqueeze(1)
                           
                            w_out, _, _ = writer_copy(
                                vec_in,
                                hidden_state=None,
                                training_mode=True,
                                autoencoder_mode=True
                            )
                           
                            r_out, _, _ = reader_copy(
                                w_out,
                                hidden_state=None,
                                training_mode=True
                            )
                           
                            if r_out is not None:
                                out_vec = r_out[:, -1, :].squeeze(0)
                                out_vec = F.normalize(out_vec, dim=-1)
                               
                                ae_loss = F.mse_loss(out_vec, vec)
                                autoencoder_loss += ae_loss * 0.1
                   
                    if autoencoder_loss > 0:
                        total_loss = total_loss + autoencoder_loss
                        training_stats['avg_recon_loss'] = (
                            training_stats['avg_recon_loss'] * epoch + autoencoder_loss.item()
                        ) / (epoch + 1)
                   
                    # ===========================================================
                    # 🔥 Backward pass - теперь точно есть граф!
                    # ===========================================================
                    total_loss.backward()
                   
                    # ===========================================================
                    # 🔥 АНТИ-NaN ЩИТ ДЛЯ ОПТИМИЗАТОРА
                    # ===========================================================
                    has_nan_grad = False
                    for param in list(writer_copy.parameters()) + list(reader_copy.parameters()):
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                            
                    if has_nan_grad:
                        logger.warning(f"⚠ Сон Ep {epoch}: Найдены NaN в градиентах! Пропуск шага оптимизатора.")
                        optimizer.zero_grad()
                        continue

                    torch.nn.utils.clip_grad_norm_(writer_copy.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(reader_copy.parameters(), max_norm=0.5)
                   
                    optimizer.step()
                   
                    total_loss_val = total_loss.item()
                    diff = stats.get('score_diff', 0)
                    margin = stats.get('margin', 0.3)
                   
                    training_stats['loss_history'].append(total_loss_val)
                    training_stats['avg_contrastive_loss'] = (training_stats['avg_contrastive_loss'] * epoch + total_loss_val) / (epoch + 1)
                    training_stats['avg_similarity'] = (training_stats['avg_similarity'] * epoch + stats.get('avg_similarity', 0)) / (epoch + 1)
                    training_stats['avg_margin'] = (training_stats['avg_margin'] * epoch + margin) / (epoch + 1)
                    training_stats['epochs_completed'] = epoch + 1
                   
                    if epoch % 2 == 0 or epoch == 14:
                        diff_icon = "🔥" if diff > margin else ("🔴" if diff < 0 else "🟢")
                       
                        log_msg = (
                            f" Сон[копия] Ep {epoch+1:2d}: loss={total_loss_val:.4f} "
                            f"diff={diff_icon}{diff:.3f}"
                        )
                       
                        log_msg += f" | Rec={recon_loss_raw:.4f}"
                        if autoencoder_loss > 0:
                            log_msg += f" | AE={autoencoder_loss:.4f}"
                       
                        if epoch % 3 == 0 and pair_types:
                            type_counts = {}
                            for t in pair_types:
                                type_counts[t] = type_counts.get(t, 0) + 1
                            type_str = " ".join([f"{t}:{c}" for t, c in type_counts.items()])
                            log_msg += f" | Пары: {type_str}"
                       
                        logger.info(log_msg)
                   
                    if total_loss_val < best_loss:
                        best_loss = total_loss_val
                        patience_counter = 0
                        training_stats['patience_counter'] = 0
                    else:
                        patience_counter += 1
                        training_stats['patience_counter'] = patience_counter
                   
                    if patience_counter >= max_patience:
                        logger.info(f"⏹️ Ранняя остановка сна на копиях")
                        break
                   
                except Exception as e:
                    if "inplace" in str(e):
                        logger.warning(f"⚠ Inplace catch at ep {epoch}")
                        optimizer.zero_grad()
                        continue
                    else:
                        logger.error(f"Error ep {epoch}: {e}")
                        continue
       
        # 🔥 ВЫПОЛНЯЕМ ПРОВЕРКУ ВОССТАНОВЛЕНИЯ
        recovery_stats = self._perform_simple_recovery_check(positive_thoughts, device)
        recovery_stats['recovery_level'] = self.recovery_level
       
        return best_loss, {
            'recovery_stats': recovery_stats,
            'training_stats': training_stats
        }
   
    def _perform_simple_recovery_check(self, positive_thoughts, device):
        """🔧 Адаптивная попытка восстановления с curriculum и эволюцией мыслей"""
        if len(positive_thoughts) < 3:
            return {'bonus': 0.0, 'success_rate': 0.0, 'evolved_count': 0}
       
        recon_attempts = min(8, len(positive_thoughts))
        recon_improvements = []
        successes = 0
        evolved_thoughts = 0
       
        for _ in range(recon_attempts):
            try:
                test_thought = random.choice(positive_thoughts)
                original_vector = torch.tensor(test_thought['snapshot'], dtype=torch.float32, device=device)
                original_vector = F.normalize(original_vector, dim=-1)
                
                # ===========================================================
                # 🔥 ИСПРАВЛЕНИЕ 6: правильная структура с else для каждого уровня
                # ===========================================================
                if self.recovery_level == 0:
                    noise_strength = random.uniform(0.05, 0.1)
                    corrupted_vector = original_vector + torch.randn_like(original_vector) * noise_strength
                    corruption_method = "light_noise"
                    
                elif self.recovery_level == 1:
                    noise_strength = random.uniform(0.15, 0.25)
                    corrupted_vector = original_vector + torch.randn_like(original_vector) * noise_strength
                    corruption_method = "medium_noise"
                    D = original_vector.shape[-1]
                    if D >= 4:
                        chunk_size = D // 4
                        chunks = [original_vector[..., i*chunk_size:(i+1)*chunk_size] for i in range(4)]
                        random.shuffle(chunks[:2])
                        corrupted_vector = torch.cat(chunks, dim=-1)
                        corruption_method += "+twist"
                        
                elif self.recovery_level == 2:
                    hard_negatives = [t for t in self.data_manager.get_thoughts_slice()
                                    if t.get('type') == 'hard_negative']
                    if hard_negatives:
                        hn_thought = random.choice(hard_negatives)
                        corrupted_vector = torch.tensor(hn_thought['snapshot'], dtype=torch.float32, device=device)
                        corruption_method = f"hard_negative_{hn_thought.get('method', 'unknown')}"
                    else:
                        # 🔥 ИСПРАВЛЕНИЕ 6: корректный else внутри уровня
                        corrupted_vector = original_vector + torch.randn_like(original_vector) * 0.3
                        corruption_method = "strong_noise_fallback"
                        
                else:  # level >= 3
                    complex_neg = [t for t in self.data_manager.get_thoughts_slice()
                                 if t.get('type') in ['hard_negative', 'gaslight_trap']
                                 and t.get('similarity_to_original', -0.5) < -0.5]
                    if complex_neg:
                        cn_thought = random.choice(complex_neg)
                        corrupted_vector = torch.tensor(cn_thought['snapshot'], dtype=torch.float32, device=device)
                        corruption_method = f"complex_{cn_thought.get('type', 'unknown')}"
                    else:
                        corrupted_vector = -original_vector * random.uniform(0.7, 0.9)
                        corruption_method = "inversion"
                
                corrupted_vector = F.normalize(corrupted_vector, dim=-1)
                
                before_sim = F.cosine_similarity(corrupted_vector.unsqueeze(0), original_vector.unsqueeze(0)).item()
                
                with torch.no_grad():
                    corrupted_input = corrupted_vector.unsqueeze(0).unsqueeze(1)
                    restored_output, _, _ = self.mamba_writer(
                        corrupted_input,
                        hidden_state=None,
                        training_mode=False
                    )
                    if restored_output is None:
                        continue
                       
                    restored_vector = restored_output[:, -1, :].squeeze(0)
                    restored_vector = F.normalize(restored_vector, dim=-1)
                
                after_sim = F.cosine_similarity(restored_vector.unsqueeze(0), original_vector.unsqueeze(0)).item()
                improvement = max(0.0, after_sim - before_sim) * (1.0 + self.recovery_level * 0.4)
                recon_improvements.append(improvement)
                
                if improvement > 0.1:
                    successes += 1
                
                if improvement > 0.2 and after_sim > 0.7:
                    evolved_thought = test_thought.copy()
                    
                    evolved_delta = min(1.0, test_thought.get('delta', 0.5) * 1.2)
                    evolved_reward = min(0.95, test_thought.get('reward', 0.5) + 0.15)
                    
                    evolved_thought.update({
                        "step": self.step_count,
                        "type": "evolved_insight",
                        "delta": round(evolved_delta, 4),
                        "reward": round(evolved_reward, 4),
                        "surprise": round(test_thought.get('surprise', 0.1) * 0.8, 4),
                        "snapshot": restored_vector.cpu().tolist(),
                        "snapshot_norm": float(torch.norm(restored_vector).item()),
                        "original_step": test_thought.get('step'),
                        "evolution_method": "recovery",
                        "improvement": round(improvement, 4),
                        "recovery_level": self.recovery_level,
                        "corruption_method": corruption_method,
                        "age": 0,
                        "max_age": self.max_age_positive,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    evolved_count = sum(1 for t in self.data_manager.thought_chains
                                      if t.get('type') == 'evolved_insight')
                    if evolved_count < 10:
                        self.data_manager.add_thought_chain(evolved_thought)
                        evolved_thoughts += 1
                        
                        self.recovered_thoughts_history.append({
                            'step': self.step_count,
                            'improvement': improvement,
                            'level': self.recovery_level,
                            'method': corruption_method
                        })
                        if len(self.recovered_thoughts_history) > self.max_recovered_thoughts:
                            self.recovered_thoughts_history.pop(0)
                       
            except Exception as e:
                continue
       
        if not recon_improvements:
            return {'bonus': 0.0, 'success_rate': 0.0, 'evolved_count': 0}
       
        avg_improvement = np.mean(recon_improvements)
        success_rate = successes / recon_attempts
       
        if success_rate >= 0.6:
            self.recovery_success_streak += 1
            self.recovery_failure_streak = 0
            if self.recovery_success_streak >= 3 and self.recovery_level < self.max_recovery_level:
                self.recovery_level += 1
                self.recovery_success_streak = 0
                logger.info(f"📈 Recovery curriculum: уровень повышен до {self.recovery_level} (успех {success_rate:.1%})")
        else:
            self.recovery_failure_streak += 1
            self.recovery_success_streak = 0
            if self.recovery_failure_streak >= 3 and self.recovery_level > 0:
                self.recovery_level -= 1
                self.recovery_failure_streak = 0
                logger.info(f"📉 Recovery curriculum: уровень понижен до {self.recovery_level} (неудачи {self.recovery_failure_streak})")
       
        base_bonus = max(0.0, avg_improvement * 0.8)
        level_multiplier = 1.0 + (self.recovery_level * 0.3)
        bonus = base_bonus * level_multiplier
       
        logger.info(f"🔧 Recovery curriculum [уровень {self.recovery_level}]: "
                    f"ср. улучшение = {avg_improvement:.3f}, "
                    f"успех = {success_rate:.1%}, "
                    f"бонус = {bonus:.3f}, "
                    f"эволюций = {evolved_thoughts}")
       
        return {
            'bonus': bonus,
            'success_rate': success_rate,
            'evolved_count': evolved_thoughts,
            'avg_improvement': avg_improvement,
            'recovery_level': self.recovery_level
        }
   
    def _maintain_notebook_balance(self):
        """🔥 УЛУЧШЕННЫЙ: Поддерживает баланс блокнота с корректировкой gaslight"""
        thoughts = self.data_manager.get_thoughts_slice()
       
        if len(thoughts) < 50:
            return
       
        positive_thoughts = [t for t in thoughts if t.get('type') == 'curiosity_insight']
        hard_negative_thoughts = [t for t in thoughts if t.get('type') == 'hard_negative']
        gaslight_thoughts = [t for t in thoughts if t.get('is_gaslight', False)]
       
        total_thoughts = len(thoughts)
        target_positive = int(total_thoughts * 0.6)
        target_hard = int(total_thoughts * 0.25)
        target_gaslight = int(total_thoughts * 0.15)
       
        current_positive = len(positive_thoughts)
        current_hard = len(hard_negative_thoughts)
        current_gaslight = len(gaslight_thoughts)
       
        positive_ratio = current_positive / total_thoughts
        hard_ratio = (current_hard + current_gaslight) / total_thoughts
        gaslight_ratio = current_gaslight / total_thoughts
       
        needs_balance = False
       
        if positive_ratio < 0.5 or positive_ratio > 0.7:
            needs_balance = True
       
        if hard_ratio < 0.3 or hard_ratio > 0.5:
            needs_balance = True
       
        if needs_balance and self.step_count % 100 == 0:
            logger.info(f"⚖️ Баланс блокнота: 👍{current_positive}({positive_ratio:.1%}) "
                       f"⚔️{current_hard} 🔥{current_gaslight} ({hard_ratio:.1%})")
           
            if gaslight_ratio < 0.1:
                self.gaslight_prob = min(0.3, self.gaslight_prob * 1.3)
                logger.info(f" 🔥 Увеличиваем gaslight_prob до {self.gaslight_prob:.2f}")
            elif gaslight_ratio > 0.2:
                self.gaslight_prob = max(0.05, self.gaslight_prob * 0.7)
                logger.info(f" 🔥 Уменьшаем gaslight_prob до {self.gaslight_prob:.2f}")
           
            if positive_ratio > 0.7:
                new_threshold = min(0.4, self.curiosity_threshold * 1.1)
                self.curiosity_threshold = new_threshold
                logger.info(f" ⬆️ Увеличиваем порог дельты до {new_threshold:.3f}")
               
            elif positive_ratio < 0.5:
                new_threshold = max(0.05, self.curiosity_threshold * 0.9)
                self.curiosity_threshold = new_threshold
                logger.info(f" ⬇️ Уменьшаем порог дельты до {new_threshold:.3f}")
           
            hard_negative_probability = getattr(self, 'hard_negative_probability', 0.9)
            if hard_ratio > 0.5:
                new_prob = max(0.3, hard_negative_probability * 0.8)
                self.hard_negative_probability = new_prob
                logger.info(f" ⬇️ Уменьшаем вероятность HN до {new_prob:.1%}")
               
            elif hard_ratio < 0.3:
                new_prob = min(0.99, hard_negative_probability * 1.2)
                self.hard_negative_probability = new_prob
                logger.info(f" ⬆️ Увеличиваем вероятность HN до {new_prob:.1%}")
   
    def _process_mamba_writer(self, candidates, context_vectors, mood_influence=0.0):
        """🔥 ИСПРАВЛЕННЫЙ Writer с поддержкой batch_size > 1"""
        try:
            model_device = next(self.parameters()).device
           
            if not candidates:
                if context_vectors.dim() == 3:
                    mamba_input = context_vectors[:, -1:, :]
                else:
                    mamba_input = context_vectors.unsqueeze(1)
            else:
                candidates = [c.to(model_device) if isinstance(c, torch.Tensor) else
                             torch.tensor(c, device=model_device) for c in candidates]
               
                stacked = torch.stack(candidates)
               
                if stacked.dim() == 3:
                    stacked = stacked.transpose(0, 1)
                    mamba_input = stacked
                else:
                    mamba_input = stacked.unsqueeze(0)
           
            mamba_input = mamba_input.to(model_device)
            # 🔥 ANTI-MUSH для кандидатов (нарезка из context_vectors)
            if context_vectors is not None and context_vectors.shape[1] > 0:
                num_candidates = min(5, context_vectors.shape[1])
                enriched_candidates = []
               
                for _ in range(num_candidates):
                    if self.training and context_vectors.shape[1] > 32:
                        window_size = 16
                        start_idx = torch.randint(0, max(1, context_vectors.shape[1] - window_size), (1,), device=model_device).item()
                        candidate_slice = context_vectors[:, start_idx:start_idx + window_size, :]
                        # Усредняем окно, чтобы получить четкую мысль [D]
                        candidate = candidate_slice.mean(dim=1).mean(dim=0)
                    else:
                        # Для коротких текстов берем случайный токен
                        idx = torch.randint(0, context_vectors.shape[1], (1,), device=model_device).item()
                        candidate = context_vectors[:, idx, :].mean(dim=0)
                   
                    enriched_candidates.append(candidate)
               
                if enriched_candidates:
                    enriched_stack = torch.stack(enriched_candidates) # [N, D]
                    if enriched_stack.dim() == 2:
                        enriched_stack = enriched_stack.unsqueeze(0) # [1, N, D]
                   
                    # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Расширяем до размера батча
                    if enriched_stack.size(0) == 1 and mamba_input.size(0) > 1:
                        enriched_stack = enriched_stack.expand(mamba_input.size(0), -1, -1)
                   
                    # Конкатенируем с основным входом Mamba
                    mamba_input = torch.cat([mamba_input, enriched_stack], dim=1)
                    #logger.info(f"🧬 Добавлено {len(enriched_candidates)} ANTI-MUSH кандидатов, форма: {mamba_input.shape}")
            if abs(mood_influence) > 0.1:
                activation_scale = 1.0 + mood_influence * 0.3
                mamba_input = mamba_input * activation_scale
            if mamba_input.dim() == 3:
                batch_norm = torch.norm(mamba_input, dim=-1).mean()
                if batch_norm > 3.0:
                    mamba_input = mamba_input / (batch_norm + 1e-8) * 1.0
            mamba_output, new_state, *_ = self.mamba_writer(
                mamba_input,
                hidden_state=self.mamba_writer_state
            )
           
            if mamba_output is not None:
                mamba_output = torch.nan_to_num(mamba_output, nan=0.0, posinf=1.0, neginf=-1.0)
                mamba_output = torch.clamp(mamba_output, min=-5.0, max=5.0)
           
            if new_state is not None:
                self.mamba_writer_state = new_state.detach()
               
            if self.mamba_writer_state is not None:
                if torch.isnan(self.mamba_writer_state).any() or torch.isinf(self.mamba_writer_state).any():
                    logger.warning("⚠ NaN/Inf in Mamba State! Resetting.")
                    self.mamba_writer_state = self.mamba_writer.reset_state(
                        batch_size=mamba_input.shape[0],
                        device=model_device
                    )
                else:
                    state_norm = torch.norm(self.mamba_writer_state)
                    if state_norm > 8.0:
                        scale = 5.0 / (state_norm + 1e-8)
                        self.mamba_writer_state = self.mamba_writer_state * scale
            last_token_output = mamba_output[:, -1, :]
           
            insight_vector = None
            if last_token_output.dim() > 1 and last_token_output.shape[0] > 1:
                insight_vector = last_token_output.mean(dim=0)
            else:
                insight_vector = last_token_output.squeeze(0)
           
            insight_final = None
            if insight_vector is not None:
                insight_norm = torch.norm(insight_vector, dim=-1, keepdim=True)
                normalized_insight = insight_vector / (insight_norm + getattr(self.config, 'snapshot_norm_epsilon', 1e-8))
               
                final_norm = torch.norm(normalized_insight)
                if abs(final_norm - 1.0) > 0.05:
                    normalized_insight = F.normalize(normalized_insight, dim=-1)
               
                insight_final = normalized_insight
           
            if insight_final is not None:
                insight_final = torch.nan_to_num(insight_final, nan=0.0)
            avg_input = None
            if mamba_input.dim() == 3:
                avg_input = mamba_input.mean(dim=[0, 1])
            else:
                avg_input = mamba_input.mean(dim=0)
           
            if insight_final is None or avg_input is None:
                self._last_writer_delta = 0.0
                # 🔥 Даже при ошибке сохраняем нулевое удивление
                self._last_surprise = 0.0
                self._surprise_for_hebb = 0.0
                return None
           
            insight_normalized = insight_final.flatten()
            avg_input = avg_input.flatten()
           
            avg_norm = torch.norm(avg_input)
            if avg_norm > 1e-8:
                avg_normalized = avg_input / avg_norm
            else:
                avg_normalized = avg_input
           
            mamba_delta = 0.0
           
            if torch.norm(insight_normalized).item() > 1e-8 and avg_norm > 1e-8:
                cosine_sim = F.cosine_similarity(
                    insight_normalized.unsqueeze(0),
                    avg_normalized.unsqueeze(0),
                    dim=1
                ).item()
               
                cosine_sim = max(-0.999, min(0.999, cosine_sim))
               
                angle = torch.acos(torch.tensor(cosine_sim)).item()
                norm_change = abs(torch.norm(insight_normalized).item() - avg_norm) / (avg_norm + 1e-8)
               
                angle_component = angle / 3.14159
                norm_component = min(norm_change, 1.0)
               
                mamba_delta = 0.7 * angle_component + 0.3 * norm_component
           
            mood_factor = 1.0 + mood_influence * 0.5
           
            if insight_final is not None:
                try:
                    was_training = self.predictor.training
                    
                    # ===========================================================
                    # 🔥 УЧИМ ПРЕДИКТОР В ПРОЦЕССЕ WRITER
                    # ===========================================================
                    if was_training:
                        self.predictor.train()
                        # ОТРЫВАЕМ ВХОД от градиентов Mamba
                        pred_input = insight_final.unsqueeze(0).detach()
                        predicted = self.predictor(pred_input).squeeze(0)
                        predicted = F.normalize(predicted, dim=-1)
                        
                        # Считаем метрики (detach обязателен)
                        metrics = self.curiosity.compute(predicted.detach(), insight_final.detach())
                        
                        # 🔥 СОХРАНЯЕМ LOSS ПРЕДИКТОРА
                        self._predictor_loss = F.mse_loss(predicted, insight_final.detach()) * 0.5
                        
                        # 🔥 ДОПОЛНИТЕЛЬНО: обновляем предсказание с градиентами
                        if hasattr(self, '_last_prediction'):
                            self._last_prediction = predicted.detach()
                    else:
                        self.predictor.eval()
                        with torch.no_grad():
                            pred_input = insight_final.unsqueeze(0)
                            predicted = self.predictor(pred_input).squeeze(0)
                            predicted = F.normalize(predicted, dim=-1)
                        metrics = self.curiosity.compute(predicted, insight_final)
                    
                    # ===========================================================
                    # 🔥 ИЗВЛЕКАЕМ SURPRISE И ДРУГИЕ МЕТРИКИ
                    # ===========================================================
                    surprise = metrics["surprise"]
                    state = metrics["state"]
                    motivation = metrics["motivation"] * mood_factor
                    arousal = metrics["arousal"]
                    habituation = metrics["habituation"]
                   
                    # 🔥🔥🔥 ГАРАНТИРОВАННОЕ СОХРАНЕНИЕ SURPRISE В НЕСКОЛЬКИХ МЕСТАХ
                    self._last_surprise = surprise
                    self._surprise_for_hebb = surprise # отдельный атрибут для Hebb
                    self._surprise_value = surprise # дополнительный атрибут для надежности
                   
                    # 🔥 Сохраняем также в виде тензора для обратной совместимости
                    if isinstance(surprise, torch.Tensor):
                        self._surprise_tensor = surprise.clone().detach()
                    else:
                        self._surprise_tensor = torch.tensor(surprise, device=model_device)
                   
                    self._curiosity_state = state
                    self._curiosity_motivation = motivation
                    self._curiosity_arousal = arousal
                    self._curiosity_habituation = habituation
                   
                    if was_training:
                        self.predictor.train()
                   
                    base_threshold = self.curiosity_threshold
                   
                    if state == "bored":
                        state_factor = 0.4
                    elif state == "overloaded" or state == "shocked":
                        state_factor = 1.3
                    elif state == "engaged":
                        state_factor = 0.7
                    else:
                        state_factor = 1.0
                   
                    adaptive_threshold = base_threshold * state_factor * mood_factor
                   
                    notebook_ratio = len(self.data_manager.thought_chains) / self.config.notebook_size
                    if notebook_ratio > 0.9:
                        adaptive_threshold *= 0.9
                    elif notebook_ratio < 0.3:
                        adaptive_threshold *= 0.5
                   
                    adaptive_threshold = max(0.04, min(0.30, adaptive_threshold))
                   
                    if self._curiosity_state == "overloaded":
                        motivation_boost = 1.0
                    else:
                        motivation_boost = 1.0 + motivation * 1.0
                   
                    effective_delta = mamba_delta * motivation_boost
                    effective_surprise = surprise * motivation_boost
                   
                    if not isinstance(effective_delta, (int, float)):
                        effective_delta = effective_delta.item() if hasattr(effective_delta, 'item') else float(effective_delta)
                    if not isinstance(surprise, (int, float)):
                        surprise = surprise.item() if hasattr(surprise, 'item') else float(surprise)
                    if not isinstance(motivation, (int, float)):
                        motivation = motivation.item() if hasattr(motivation, 'item') else float(motivation)
                   
                    mood_bonus = 0.1 if mood_influence > 0.3 else 0.0
                   
                    reward = min(0.95, max(0.05,
                        effective_delta * 0.6 +
                        effective_surprise * 0.4 +
                        motivation * 0.2 +
                        mood_bonus -
                        0.1
                    ))
                   
                    importance = min(1.0, max(0.0,
                        effective_surprise * 0.4 +
                        effective_delta * 0.3 +
                        motivation * 0.2 +
                        (1.0 if state == "engaged" else 0.5) * 0.1 +
                        (abs(mood_influence) * 0.1)
                    ))
                   
                    self.mamba_writer_ema = 0.95 * self.mamba_writer_ema + 0.05 * effective_delta
                    self.mamba_writer_ema = min(0.6, max(0.01, self.mamba_writer_ema))
                   
                    self._last_writer_delta = effective_delta
                   
                    # 🔥 ИСПОЛЬЗУЕМ СОХРАНЕННОЕ УДИВЛЕНИЕ
                    current_surprise = self._last_surprise
                   
                    if current_surprise < 0.1:
                        threshold_multiplier = 0.5
                        logger.debug(f"📉 Критически низкое удивление ({current_surprise:.3f}) -> порог записи снижен до x0.5")
                    elif current_surprise < 0.15:
                        threshold_multiplier = 0.7
                        logger.debug(f"📉 Низкое удивление ({current_surprise:.3f}) -> порог записи снижен до x0.7")
                    elif current_surprise > 0.3:
                        threshold_multiplier = 1.2
                        logger.debug(f"📈 Высокое удивление ({current_surprise:.3f}) -> порог записи повышен до x1.2")
                    else:
                        threshold_multiplier = 1.0
                   
                    effective_threshold = self.writer_insight_threshold * threshold_multiplier
                    effective_threshold = max(0.15, min(0.8, effective_threshold))
                   
                    adaptive_threshold = self.curiosity_threshold * state_factor * mood_factor
                   
                    record_every_n = max(200, 400 - int(self.mamba_writer_ema * 300))
                    steps_since_last = self.step_count - self._last_record_step
                    bored_condition = (state == "bored" and self.curiosity._bored_count > 5)
                    force_record = (
                        (self.sleep_cycles > 0 and self.step_count - self._last_record_step < 200) or
                        self.step_count % max(60, 180 - int(self.mamba_writer_ema * 120)) == 0 or
                        steps_since_last > 900
                    )
                    should_record = self._should_record_thought(
                        state, effective_delta, effective_surprise,
                        effective_threshold, adaptive_threshold,
                        bored_condition, force_record
                    )
                   
                    if should_record:
                        snapshot = insight_final.detach().cpu()
                        snapshot_np = snapshot.numpy()
                        snapshot_norm = np.linalg.norm(snapshot_np)
                        if abs(snapshot_norm - 1.0) > 0.05:
                            snapshot = torch.from_numpy(snapshot_np / (snapshot_norm + 1e-8))
                       
                        thought_data = {
                            "step": self.step_count,
                            "type": "curiosity_insight",
                            "delta": round(float(effective_delta), 4),
                            "surprise": round(float(surprise), 4),
                            "motivation": round(float(motivation), 4),
                            "state": state,
                            "arousal": round(float(arousal), 4),
                            "habituation": round(float(habituation), 4),
                            "reward": round(float(reward), 4),
                            "importance": round(float(importance), 4),
                            "mood_influence": round(float(mood_influence), 4),
                            "curiosity_threshold": round(float(self.curiosity_threshold), 4),
                            "snapshot": snapshot.tolist(),
                            "snapshot_norm": float(torch.norm(snapshot).item()),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                       
                        self.data_manager.add_thought_chain(thought_data)
                        self._last_record_step = self.step_count
                        try:
                            model_device = next(self.parameters()).device
                            query_vec = snapshot.to(model_device) if isinstance(snapshot, torch.Tensor) else \
                                        torch.tensor(snapshot, dtype=torch.float32, device=model_device)
                            query_vec = F.normalize(query_vec.view(-1), dim=-1)
                           
                            _, info = self.query_memory(query_vec, top_k=4)
                           
                            if info.get('found', 0) > 0:
                                logger.debug(f"🔍 Recall после записи: активировано {info['found']} слотов")
                        except Exception as e:
                            logger.debug(f"⚠ Recall после записи: {e}")
                        if effective_delta >= 1.0 or surprise >= 0.7:
                            try:
                                model_device = next(self.parameters()).device
                                query_vec = snapshot.to(model_device) if isinstance(snapshot, torch.Tensor) else \
                                            torch.tensor(snapshot, dtype=torch.float32, device=model_device)
                                query_vec = F.normalize(query_vec.view(-1), dim=-1)
                                context, info = self.query_memory(query_vec, top_k=3)
                                is_centroid_duplicate = False
                                max_sim = 0.0
                                if info.get('found', 0) > 0:
                                    if 'top_sims' in info and info['top_sims']:
                                        max_sim = max(info['top_sims'])
                                    elif context is not None:
                                        max_sim = F.cosine_similarity(query_vec.unsqueeze(0), context.unsqueeze(0)).item()
                                    if max_sim > 0.975:
                                        is_centroid_duplicate = True
                                if not is_centroid_duplicate:
                                    centroid_data = {
                                        'snapshot': snapshot.tolist() if isinstance(snapshot, torch.Tensor) else snapshot,
                                        'delta': float(effective_delta),
                                        'step': self.step_count,
                                        'type': 'curiosity_insight'
                                    }
                                    self._write_thought_to_centroid(centroid_data)
                                    logger.debug(f"⚡ Сверхсильный инсайт → центроид (Δ={effective_delta:.3f})")
                                else:
                                    logger.debug(f"🧹 Центроид: дубликат пропущен")
                            except Exception as e:
                                logger.warning(f"⚠ Ошибка записи в центроид: {e}")
                        if effective_delta >= 0.2:
                            hard_negative_probability = getattr(self, 'hard_negative_probability', 1.0)
                            mood_adjusted_prob = hard_negative_probability * (1.0 - abs(mood_influence) * 0.3)
                            if random.random() < mood_adjusted_prob:
                                negative_thought = self._generate_hard_negative(thought_data)
                                if negative_thought:
                                    thoughts = self.data_manager.get_thoughts_slice()
                                    hard_negative_count = sum(1 for t in thoughts if t.get('type') == 'hard_negative')
                                    max_ratio = getattr(self.config, 'max_hard_negative_ratio', 0.4)
                                    if (hard_negative_count + 1) / (len(thoughts) + 2) <= max_ratio:
                                        self.data_manager.add_thought_chain(negative_thought)
                                        logger.debug(f"⚔️ Добавлен Hard Negative (метод: {negative_thought.get('method', 'unknown')})")
                       
                        if self.step_count % 500 == 0:
                            self._update_config_based_on_logs()
                       
                        if state == "overloaded" or state == "shocked":
                            if self.data_manager.get_thought_count() > 75:
                                logger.info(f"🧠 {state} → принудительный сон")
                                self._check_and_trigger_sleep_async()
                   
                    # 🔥 ВОЗВРАЩАЕМ insight_final И ГАРАНТИРУЕМ, ЧТО SURPRISE СОХРАНЕН
                    return insight_final
                   
                except Exception as e:
                    logger.error(f"⚠ Curiosity error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 🔥 При ошибке тоже сохраняем нулевое удивление
                    self._last_surprise = 0.0
                    self._surprise_for_hebb = 0.0
                    self._surprise_value = 0.0
                    return None
            else:
                # 🔥 При отсутствии insight_final сохраняем нулевое удивление
                self._last_surprise = 0.0
                self._surprise_for_hebb = 0.0
                self._surprise_value = 0.0
                return None
               
        except Exception as e:
            logger.error(f"❌ Writer ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 🔥 При критической ошибке тоже сохраняем нулевое удивление
            self._last_surprise = 0.0
            self._surprise_for_hebb = 0.0
            self._surprise_value = 0.0
            return None
      
    def _update_config_based_on_logs(self):
        """🔥 АДАПТИВНЫЙ ГАЗЛАЙТИНГ"""
        thoughts = self.data_manager.get_thoughts_slice()
        gaslights = [t for t in thoughts if t.get('is_gaslight')]
       
        if len(gaslights) < 5:
            self.gaslight_prob = min(0.4, self.gaslight_prob + 0.05)
        elif len(gaslights) > 15:
            self.gaslight_prob = max(0.1, self.gaslight_prob - 0.02)
           
        if len(gaslights) < 10:
            logger.info(f"🧪 Режим накопления Gaslight: n={len(gaslights)}, prob={self.gaslight_prob:.2f}")
   
    def _clamp_states_norm(self, max_norm=10.0):
        try:
            if self.mamba_writer_state is not None:
                if isinstance(self.mamba_writer_state, torch.Tensor):
                    norm = torch.norm(self.mamba_writer_state).item()
                    if norm > max_norm:
                        scale = max_norm / (norm + 1e-8)
                        self.mamba_writer_state = self.mamba_writer_state * scale
           
            if self.mamba_reader_state is not None:
                if isinstance(self.mamba_reader_state, torch.Tensor):
                    norm = torch.norm(self.mamba_reader_state).item()
                    if norm > max_norm:
                        scale = max_norm / (norm + 1e-8)
                        self.mamba_reader_state = self.mamba_reader_state * scale
                           
        except Exception as e:
            logger.debug(f"⚠ Ошибка при ограничении состояний: {e}")
   
    def _restore_saved_states(self, saved_states):
        if saved_states['writer'] is not None:
            self.mamba_writer_state = saved_states['writer'] * 0.7
       
        if saved_states['reader'] is not None:
            self.mamba_reader_state = saved_states['reader'] * 0.8
       
        if not saved_states['training_mode']:
            self.mamba_writer.eval()
            self.predictor.eval()
   
    def _report_status_with_gpu_monitoring(self):
        try:
            if self.step_count % 200 != 0:
                return
           
            r_ema = getattr(self, 'mamba_reader_ema', 0.0)
            w_ema = getattr(self, 'mamba_writer_ema', 0.0)
            last_delta = getattr(self, '_last_writer_delta', 0.0)
            last_surprise = getattr(self, '_last_surprise', 0.0)
            insights = len(getattr(self.data_manager, 'thought_chains', []))
           
            thoughts = self.data_manager.get_thoughts_slice()
            hard_negatives = len([t for t in thoughts if t.get('type') == 'hard_negative'])
            positive_thoughts = len([t for t in thoughts if t.get('type') == 'curiosity_insight'])
            gaslights = len([t for t in thoughts if t.get('is_gaslight', False)])
           
            if w_ema > 0.3:
                w_status = "🔥"
            elif w_ema > 0.15:
                w_status = "🟢"
            elif w_ema > 0.05:
                w_status = "🟡"
            else:
                w_status = "⚪"
           
            if last_surprise > 0.3:
                s_status = "🤯"
            elif last_surprise > 0.2:
                s_status = "😲"
            elif last_surprise > 0.1:
                s_status = "🤔"
            else:
                s_status = "😐"
           
            notebook_size = getattr(self.config, 'notebook_size', 100)
            notebook_pct = int((insights / notebook_size) * 100) if notebook_size > 0 else 0
            if notebook_pct > 80:
                n_status = "🔴"
            elif notebook_pct > 50:
                n_status = "🟡"
            else:
                n_status = "🟢"
           
            logger.info(f"\n🧠 MAMBA v11.1 [шаг {self.step_count}]")
            logger.info(f"📊 Активность: W{w_status}{w_ema:.2f} Δ{last_delta:.3f}")
            logger.info(f"🎯 Удивление: {s_status}{last_surprise:.3f} Состояние: {getattr(self, '_curiosity_state', 'unknown')}")
            logger.info(f"📓 Блокнот: {n_status}{insights:3d}/{notebook_size} 💤 Циклов: {getattr(self, 'sleep_cycles', 0)}")
            logger.info(f"📊 Мысли: {positive_thoughts}👍 {hard_negatives}⚔️ {gaslights}🔥")
           
            logger.info(f"⚙️ Параметры: gaslight_prob={self.gaslight_prob:.2f}")
           
            # ===========================================================
            # 🔥 ОБНОВЛЁННЫЙ ВЫВОД МЕТА-СОСТОЯНИЯ (v3.0)
            # ===========================================================
            if hasattr(self, 'meta_predictor') and self.meta_predictor is not None:
                if hasattr(self.meta_predictor, '_last_meta') and self.meta_predictor._last_meta is not None:
                    meta = self.meta_predictor._last_meta
                   
                    # Получаем решения
                    decisions = meta.get("decision", ["act"])
                    if isinstance(decisions, torch.Tensor):
                        decisions = [decisions[0].item()] if decisions.numel() > 0 else ["act"]
                   
                    # Формируем строку с мета-информацией
                    meta_parts = []
                   
                    # Уверенность
                    confidence = meta.get('confidence', None)
                    if confidence is not None:
                        conf_val = confidence.mean().item() if isinstance(confidence, torch.Tensor) else confidence
                        meta_parts.append(f"уверен={conf_val:.3f}")
                   
                    # Сомнение
                    doubt = meta.get('doubt', None)
                    if doubt is not None:
                        doubt_val = doubt.mean().item() if isinstance(doubt, torch.Tensor) else doubt
                        meta_parts.append(f"сомнение={doubt_val:.3f}")
                   
                    # Любопытство
                    curiosity = meta.get('curiosity', None)
                    if curiosity is not None:
                        cur_val = curiosity.mean().item() if isinstance(curiosity, torch.Tensor) else curiosity
                        meta_parts.append(f"любопыт={cur_val:.3f}")
                   
                    # Решение (главное)
                    decision_emoji = {
                        "act": "⚡",
                        "recheck": "🔍",
                        "rethink": "🧠",
                        "deep_rethink": "💭",
                        "reflect": "🪞"
                    }
                    main_decision = decisions[0] if decisions else "act"
                    decision_icon = decision_emoji.get(main_decision, "❓")
                    meta_parts.append(f"решение={decision_icon}{main_decision}")
                   
                    # Дополнительные метрики для логирования (кратко)
                    readiness = meta.get('readiness', None)
                    if readiness is not None:
                        ready_val = readiness.mean().item() if isinstance(readiness, torch.Tensor) else readiness
                        meta_parts.append(f"готов={ready_val:.3f}")
                   
                    novelty = meta.get('novelty', None)
                    if novelty is not None:
                        nov_val = novelty.mean().item() if isinstance(novelty, torch.Tensor) else novelty
                        meta_parts.append(f"новизна={nov_val:.3f}")
                   
                    # Информация о рефлексии
                    reflection_steps = meta.get('reflection_steps', 0)
                    reflection_applied = meta.get('reflection_applied', False)
                    if reflection_applied:
                        meta_parts.append(f"🔄рефл={reflection_steps}")
                   
                    # Логируем основную мета-информацию
                    logger.info(f"🧠 Внутренний голос: {' | '.join(meta_parts)}")
                   
                    # Дополнительное логирование для отладки решений
                    if len(decisions) > 1 and self.step_count % 400 == 0:
                        unique_decisions = set(decisions[:10])
                        logger.info(f" 📊 Распределение решений: {unique_decisions}")
                   
                    # Логируем мета-loss, если есть
                    meta_loss = meta.get('meta_loss', None)
                    if meta_loss is not None and isinstance(meta_loss, torch.Tensor) and meta_loss.requires_grad:
                        logger.info(f" 📉 Мета-loss: {meta_loss.item():.6f}")
                   
                    # Логируем изменения после рефлексии (если есть)
                    confidence_after = meta.get('confidence_after', None)
                    doubt_after = meta.get('doubt_after', None)
                    if confidence_after is not None and doubt_after is not None:
                        logger.info(f" 📈 После рефлексии: уверен={confidence_after:.3f}, сомнение={doubt_after:.3f}")
           
            # Существующий блок для hard negatives
            if self.step_count % 100 == 0 and hard_negatives > 0:
                methods_count = {}
                avg_similarity = 0
                for t in thoughts:
                    if t.get('type') == 'hard_negative':
                        method = t.get('method', 'unknown')
                        methods_count[method] = methods_count.get(method, 0) + 1
                        avg_similarity += t.get('similarity_to_original', 0)
               
                if methods_count:
                    avg_similarity /= hard_negatives
                    methods_str = " ".join([f"{m}:{c}" for m, c in methods_count.items()])
                    logger.info(f"⚔️ Hard Negatives: {methods_str}, avg sim={avg_similarity:.3f}")
           
            # Дополнительное логирование для сна и обучения
            if hasattr(self, '_total_reflections') and self._total_reflections is not None:
                reflection_count = self._total_reflections.item() if isinstance(self._total_reflections, torch.Tensor) else self._total_reflections
                reflection_depth = getattr(self, '_reflection_depth_avg', 0)
                if reflection_count > 0:
                    logger.info(f"🔄 Рефлексия: {reflection_count} раз, средняя глубина={reflection_depth:.2f}")
           
        except Exception as e:
            logger.debug(f"⚠ Ошибка отчета: {e}")
            
    def _should_record_thought(self, state, effective_delta, effective_surprise,
                              effective_threshold, adaptive_threshold,
                              bored_condition, force_record):
        """🔥 ИСПРАВЛЕНИЕ 7: улучшенная логика записи мыслей"""
        if state in ["overloaded", "shocked"]:
            return False
        
        reasons = []
        
        # Проверяем delta (основной критерий)
        if effective_delta >= effective_threshold * 0.72:
            reasons.append("delta_good")
        else:
            # Если delta низкий - не записываем
            return False
        
        # Проверяем surprise (вторичный критерий)
        if effective_surprise > adaptive_threshold * 0.6:
            reasons.append("surprise")
        elif effective_delta >= effective_threshold * 1.2:
            # Высокий delta может компенсировать низкий surprise
            reasons.append("delta_high_compensation")
        
        # Дополнительные условия
        if state == "engaged":
            reasons.append("engaged")
        if bored_condition:
            reasons.append("bored")
        if force_record:
            reasons.append("forced")
        
        if reasons:
            logger.debug(f"📝 Запись мысли: {', '.join(reasons)} (Δ={effective_delta:.3f}, 🤯={effective_surprise:.3f})")
            return True
        
        return False
        
# ===================================================================
# МЕТОДЫ ГЕНЕРАЦИИ HARD NEGATIVES
# ===================================================================
    def _generate_hard_negative(self, positive_thought, preferred_method=None):
        """⚔️ Генерирует сложный негатив с учетом текущего баланса блокнота"""
        thoughts = self.data_manager.get_thoughts_slice()
        
        hard_negative_count = sum(1 for t in thoughts if t.get('type') == 'hard_negative')
        gaslight_count = sum(1 for t in thoughts if t.get('is_gaslight', False))
        total_hard = hard_negative_count + gaslight_count
        
        # Вычисление вероятностей
        base_probability = getattr(self, 'hard_negative_probability', 0.9)
        adaptive_probability = base_probability
        
        if len(thoughts) > 50:
            current_hard_ratio = total_hard / len(thoughts)
            
            if current_hard_ratio > 0.4:
                reduction_factor = 0.4 / current_hard_ratio
                adaptive_probability = base_probability * reduction_factor
                adaptive_probability = max(0.3, min(0.95, adaptive_probability))
                
                if self.step_count % 50 == 0:
                    logger.debug(f"⚖️ Баланс Hard: {total_hard}/{len(thoughts)}={current_hard_ratio:.2f} → prob={adaptive_probability:.3f}")
            
            elif current_hard_ratio < 0.2:
                boost_factor = 1.0 + (0.2 - current_hard_ratio)
                adaptive_probability = base_probability * boost_factor
                adaptive_probability = min(0.99, adaptive_probability)
        
        # Gaslight проверка
        gaslight_probability = getattr(self, 'gaslight_prob', 0.1)
        
        if gaslight_count / max(1, len(thoughts)) > 0.15:
            should_gaslight = False
            if self.step_count % 50 == 0:
                logger.debug(f"🔥 Пропускаем gaslight (достигнут лимит: {gaslight_count}/{len(thoughts)})")
        else:
            should_gaslight = (
                random.random() < gaslight_probability and
                positive_thought.get('reward', 0) > 0.8 and
                positive_thought.get('delta', 0) > 0.75 and
                positive_thought.get('type') == 'curiosity_insight'
            )
        
        if should_gaslight:
            try:
                positive_vector = torch.tensor(positive_thought['snapshot'], dtype=torch.float32)
                positive_vector = F.normalize(positive_vector, dim=-1)
                
                if random.random() < 0.6:
                    gaslight_vector, gaslight_thought = AdversarialInverterOptimized.create_gaslight_trap(
                        positive_vector, positive_thought
                    )
                    method = "direct_gaslight"
                else:
                    corruption_strength = 0.03 + random.random() * 0.07
                    gaslight_vector, gaslight_thought = AdversarialInverterOptimized.create_subtle_gaslight(
                        positive_vector, positive_thought, corruption_strength
                    )
                    method = "subtle_gaslight"
                
                gaslight_thought['method'] = method
                gaslight_thought['is_gaslight'] = True
                
                if self.step_count % 50 == 0:
                    logger.info(f"🔥 GASLIGHT TRAP ({method}): "
                               f"Δ={positive_thought.get('delta', 0):.1f}→{gaslight_thought.get('delta', 0):.1f}")
                
                return gaslight_thought
                
            except Exception as e:
                logger.warning(f"⚠ Ошибка генерации Gaslight Trap: {e}")
        
        if positive_thought.get('delta', 0) < 0.3:
            return None
        
        if random.random() > adaptive_probability:
            if self.step_count % 100 == 0:
                logger.debug(f"⚖️ Пропускаем Hard Negative по вероятности: {adaptive_probability:.3f}")
            return None
        
        # ===========================================================
        # 🔥  правильное определение available_methods
        # ===========================================================
        available_methods = ['lobotomy', 'twist']
        
        other_positives = [t for t in thoughts
                          if t.get('snapshot') is not None
                          and t.get('step', -1) != positive_thought.get('step', -1)]
        
        if len(other_positives) >= 1:
            available_methods.append('chimera')
        
        if not available_methods:
            logger.debug("⚠ Нет доступных методов для генерации Hard Negative")
            return None
        
        # Выбор метода
        hard_negatives = [t for t in thoughts if t.get('type') == 'hard_negative']
        
        if len(hard_negatives) > 10:
            existing_methods = {}
            for t in hard_negatives[-30:]:
                method = t.get('method', 'unknown')
                existing_methods[method] = existing_methods.get(method, 0) + 1
            
            total_existing = sum(existing_methods.values())
            
            method_weights = {}
            for method in available_methods:
                count = existing_methods.get(method, 0)
                frequency = count / max(1, total_existing)
                
                if method == 'lobotomy' and frequency < 0.2:
                    method_weights[method] = 2.0
                elif method == 'chimera' and frequency < 0.25:
                    method_weights[method] = 1.7
                else:
                    method_weights[method] = 1.5 - (frequency * 0.8)
            
            total_weight = sum(method_weights.values())
            if total_weight > 0:
                r = random.random() * total_weight
                cumulative = 0
                selected_method = random.choice(available_methods)
                for method, weight in method_weights.items():
                    cumulative += weight
                    if r <= cumulative:
                        selected_method = method
                        break
                
                if self.step_count % 50 == 0:
                    logger.debug(f"🎲 Выбран метод: {selected_method} (веса: {method_weights})")
            else:
                selected_method = random.choice(available_methods)
        else:
            selected_method = random.choice(available_methods)
        
        if preferred_method and preferred_method in available_methods:
            method = preferred_method
            logger.debug(f"🎯 Используем предпочтительный метод: {method}")
        else:
            method = selected_method
        
        try:
            positive_vector = torch.tensor(positive_thought['snapshot'], dtype=torch.float32)
            positive_vector = F.normalize(positive_vector, dim=-1)
            
            if method == 'chimera':
                if len(other_positives) >= 1:
                    other_thought = random.choice(other_positives)
                    try:
                        other_vector = torch.tensor(other_thought["snapshot"],
                                                  dtype=torch.float32)
                        other_vector = F.normalize(other_vector, dim=-1)
                    except:
                        other_vector = -positive_vector * 0.5
                else:
                    other_vector = -positive_vector * 0.5
                
                intensity = 0.15 + random.random() * 0.25
                negative_vector = AdversarialInverterOptimized.create_chimera(
                    positive_vector, other_vector, intensity=intensity
                )
                
            elif method == 'lobotomy':
                original_delta = positive_thought.get('delta', 0.5)
                
                lobotomy_history = [t for t in self.data_manager.get_thoughts_slice()
                                  if t.get('method') == 'lobotomy']
                
                if len(lobotomy_history) > 10:
                    recent_similarities = [t.get('similarity_to_original', 0)
                                          for t in lobotomy_history[-10:]]
                    avg_recent_sim = np.mean(recent_similarities)
                    
                    if avg_recent_sim > -0.2:
                        original_delta = min(1.0, original_delta * 1.1)
                        logger.debug(f"🧠 Увеличиваем сложность lobotomy (средняя схожесть={avg_recent_sim:.3f})")
                    elif avg_recent_sim < -0.7:
                        original_delta = max(0.1, original_delta * 0.9)
                        logger.debug(f"🧠 Уменьшаем сложность lobotomy (средняя схожесть={avg_recent_sim:.3f})")
                
                negative_vector, lobotomy_metadata = AdversarialInverterOptimized.adaptive_lobotomy(
                    positive_vector,
                    original_delta=original_delta
                )
                
                logger.debug(f"🧠 Lobotomy: Δ={original_delta:.3f} → сложность={lobotomy_metadata['complexity']}, "
                            f"inversion={lobotomy_metadata['inversion_strength']:.1f}")
                
            else:  # twist
                # ===========================================================
                # 🔥 исправленная обработка twist для разных размерностей
                # ===========================================================
                D = positive_vector.shape[-1]
                
                if D >= 8:
                    # Для больших размерностей - разбиваем на чанки
                    num_chunks = min(8, D // 4)
                    chunk_size = D // num_chunks
                    
                    chunks = []
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i < num_chunks - 1 else D
                        chunks.append(positive_vector[..., start:end])
                    
                    random.shuffle(chunks)
                    twisted = torch.cat(chunks, dim=-1)
                    
                elif D >= 4:
                    # Для средних размерностей - фиксированное число чанков
                    chunk_size = D // 4
                    chunks = []
                    for i in range(4):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i < 3 else D
                        chunks.append(positive_vector[..., start:end])
                    
                    # Проверяем, что все чанки имеют совместимую размерность
                    chunk_lengths = [c.shape[-1] for c in chunks]
                    if len(set(chunk_lengths)) == 1 or all(l == chunk_lengths[0] for l in chunk_lengths):
                        # Все чанки одинаковой длины - можно переставлять
                        permutation_pattern = random.choice([
                            [0, 2, 1, 3],
                            [1, 0, 3, 2],
                            [2, 3, 0, 1],
                            [3, 1, 2, 0]
                        ])
                        twisted_chunks = [chunks[i] for i in permutation_pattern]
                        twisted = torch.cat(twisted_chunks, dim=-1)
                    else:
                        # Чанки разной длины - используем случайную перестановку индексов
                        indices = torch.randperm(D)
                        twisted = positive_vector[..., indices]
                else:
                    # Для маленьких размерностей - случайная перестановка
                    indices = torch.randperm(D)
                    twisted = positive_vector[..., indices]
                
                complexity = 0.3 + random.random() * 0.2
                negative_vector = torch.lerp(positive_vector, twisted, complexity)
            
            # Проверка схожести и коррекция
            similarity = F.cosine_similarity(
                positive_vector.unsqueeze(0),
                negative_vector.unsqueeze(0)
            ).item()
            
            # ===========================================================
            # 🔥 оптимизированное копирование (без deepcopy)
            # ===========================================================
            negative_thought = {}
            negative_thought.update(positive_thought)  # копируем поля
            
            if method == 'chimera':
                reward_factor = 0.3
                delta_factor = 0.4
                state = "confused_chimera"
                method_description = f"Химера (intensity={intensity:.2f})"
            elif method == 'lobotomy':
                reward_factor = 0.4
                delta_factor = 0.3
                state = "empty_shell"
                method_description = f"Лоботомия"
                
                if 'lobotomy_metadata' in locals():
                    negative_thought.update({
                        "lobotomy_complexity": lobotomy_metadata.get('complexity', 'unknown'),
                        "lobotomy_inversion_strength": round(lobotomy_metadata.get('inversion_strength', 1.0), 2),
                        "lobotomy_suppression_ratio": round(lobotomy_metadata.get('suppression_ratio', 0.1), 3),
                        "estimated_complexity": round(lobotomy_metadata.get('estimated_similarity', 0.7) * 2.0, 3)
                    })
            else:  # twist
                reward_factor = 0.35
                delta_factor = 0.5
                state = "logical_fallacy"
                method_description = f"Логический вывих (complexity={complexity:.2f})"
            
            original_reward = positive_thought.get('reward', 0.5)
            original_delta = positive_thought.get('delta', 1.0)
            original_surprise = positive_thought.get('surprise', 0.1)
            
            negative_vector_normalized = F.normalize(negative_vector, dim=-1)
            
            negative_thought.update({
                "type": "hard_negative",
                "method": method,
                "method_description": method_description,
                "similarity_to_original": round(similarity, 4),
                "reward": round(original_reward * reward_factor, 4),
                "delta": round(original_delta * delta_factor, 4),
                "surprise": round(original_surprise * 0.7, 4),
                "state": state,
                "snapshot": negative_vector_normalized.tolist(),
                "snapshot_norm": float(torch.norm(negative_vector_normalized).item()),
                "original_step": positive_thought.get('step'),
                "original_reward": round(original_reward, 4),
                "original_delta": round(original_delta, 4),
                "original_state": positive_thought.get('state', 'unknown'),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "age": 0
            })
            
            max_total_hard = int(len(thoughts) * 0.4)
            
            if total_hard >= max_total_hard:
                logger.debug(f"⚖️ Достигнут лимит Hard Negatives: {total_hard} >= {max_total_hard}")
                return None
            
            return negative_thought
            
        except Exception as e:
            logger.warning(f"⚠ Ошибка генерации Hard Negative ({method if 'method' in locals() else 'unknown'}): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
            
    def _deduplicate_notebook(self, thoughts, similarity_threshold=0.88):
        """🔥 КОНСЕРВАТИВНАЯ ДЕДУПЛИКАЦИЯ МЫСЛЕЙ"""
        if not thoughts or len(thoughts) < 2:
            return thoughts
       
        # 🔥 ОПТИМИЗАЦИЯ: ограничиваем размер для дедупликации
        MAX_DEDUP_SIZE = 500
        if len(thoughts) > MAX_DEDUP_SIZE:
            # Берём только последние MAX_DEDUP_SIZE мыслей для дедупликации
            thoughts_to_dedup = thoughts[-MAX_DEDUP_SIZE:]
            logger.debug(f"🧹 Дедупликация ограничена {MAX_DEDUP_SIZE} мыслями (всего {len(thoughts)})")
        else:
            thoughts_to_dedup = thoughts
       
        positive_thoughts = [t for t in thoughts_to_dedup if t.get('type') == 'curiosity_insight'
                           and not t.get('is_hard_negative') and not t.get('is_gaslight')]
       
        if len(positive_thoughts) < 2:
            return thoughts
       
        vectors = []
        valid_indices = []
       
        for i, t in enumerate(positive_thoughts):
            if 'snapshot' in t:
                try:
                    if isinstance(t['snapshot'], list):
                        v = torch.tensor(t['snapshot'], dtype=torch.float32)
                    else:
                        v = t['snapshot'].clone().detach()
                    vectors.append(F.normalize(v, dim=-1))
                    valid_indices.append(i)
                except:
                    continue
       
        if len(vectors) < 2:
            return thoughts
       
        vectors = torch.stack(vectors)
        sim_matrix = torch.mm(vectors, vectors.t())
       
        max_removals = int(len(positive_thoughts) * 0.3)
       
        duplicates_to_remove = set()
        n = len(vectors)
       
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j].item()
                if sim > similarity_threshold:
                    pairs.append((sim, i, j))
       
        pairs.sort(reverse=True)
       
        for sim, i, j in pairs:
            if len(duplicates_to_remove) >= max_removals:
                break
            if i in duplicates_to_remove or j in duplicates_to_remove:
                continue
               
            t_i = positive_thoughts[valid_indices[i]]
            t_j = positive_thoughts[valid_indices[j]]
           
            score_i = t_i.get('delta', 0) * 0.7 + t_i.get('age', 0) * 0.03
            score_j = t_j.get('delta', 0) * 0.7 + t_j.get('age', 0) * 0.03
           
            if score_i >= score_j:
                duplicates_to_remove.add(j)
            else:
                duplicates_to_remove.add(i)
       
        if duplicates_to_remove:
            logger.info(f"🧹 Дедупликация: удалено {len(duplicates_to_remove)} из {len(positive_thoughts)} "
                       f"(найдено пар: {len(pairs)}, лимит: {max_removals})")
           
            # Находим ID мыслей для удаления из ОРИГИНАЛЬНОГО списка
            thoughts_to_remove_ids = set()
            for idx in duplicates_to_remove:
                original_idx = valid_indices[idx]
                # Находим оригинальную мысль по индексу в positive_thoughts
                target_thought = positive_thoughts[original_idx]
                # Ищем её в исходном списке thoughts (по step или по id)
                for t in thoughts:
                    if t.get('step') == target_thought.get('step') and t.get('timestamp') == target_thought.get('timestamp'):
                        thoughts_to_remove_ids.add(id(t))
                        break
           
            cleaned_thoughts = [t for t in thoughts if id(t) not in thoughts_to_remove_ids]
           
            logger.info(f"🧹 После дедупликации: {len(thoughts)} → {len(cleaned_thoughts)} мыслей")
            return cleaned_thoughts
       
        return thoughts
        
    def _intelligent_pruning(self, thoughts, keep_count, positive_thoughts, hard_negative_thoughts, gaslight_thoughts):
        """🔥 ИНТЕЛЛЕКТУАЛЬНАЯ ОЧИСТКА С ПРОРАБОТАННОЙ СИСТЕМОЙ СТАРЕНИЯ"""
        target_positive = int(keep_count * 0.6)
        target_hard_negative = int(keep_count * 0.25)
        target_gaslight = keep_count - target_positive - target_hard_negative
       
        logger.info(f"🧹 ЦЕЛЕВОЕ РАСПРЕДЕЛЕНИЕ: 👍{target_positive} | ⚔️{target_hard_negative} | 🔥{target_gaslight}")
       
        # ===========================================================
        # Отбор positive мыслей
        # ===========================================================
        keep_positive = []
        if positive_thoughts and target_positive > 0:
            sorted_positive = sorted(
                positive_thoughts,
                key=lambda x: self._get_thought_score(x, 'positive'),
                reverse=True
            )
           
            filtered_positive = []
            for t in sorted_positive:
                delta = t.get('delta', 0)
                max_age = self.max_age_positive
               
                if delta > 0.8 and t.get('surprise', 0) > 0.4:
                    max_age = int(max_age * 1.5)
                elif delta < 0.4:
                    max_age = int(max_age * 0.5)
               
                if t.get('age', 0) <= max_age:
                    filtered_positive.append(t)
           
            keep_positive = filtered_positive[:target_positive]
            logger.info(f"🏆 Отобрано элиты: {len(keep_positive)} из {len(filtered_positive)} (возрастной фильтр)")
       
        # ===========================================================
        # Отбор hard negative мыслей
        # ===========================================================
        keep_hard_negative = []
        if hard_negative_thoughts and target_hard_negative > 0:
            recent_hard_negatives = []
            for t in hard_negative_thoughts:
                similarity = t.get('similarity_to_original', 0.5)
                max_age = t.get('max_age', int(similarity * self.max_age_hard_negative))
                if t.get('age', 0) < max_age:
                    recent_hard_negatives.append(t)
           
            if recent_hard_negatives:
                sorted_hard = sorted(
                    recent_hard_negatives,
                    key=lambda x: self._get_thought_score(x, 'hard_negative'),
                    reverse=True
                )
               
                keep_hard_negative = sorted_hard[:target_hard_negative]
       
        # ===========================================================
        # Отбор gaslight мыслей
        # ===========================================================
        keep_gaslight = []
        if gaslight_thoughts and target_gaslight > 0:
            recent_gaslight = []
            for t in gaslight_thoughts:
                dissonance = abs(t.get('original_reward', 0.9) - t.get('reward', -0.9))
                max_age = t.get('max_age', int(dissonance * self.max_age_gaslight))
                if t.get('age', 0) < max_age:
                    recent_gaslight.append(t)
           
            if recent_gaslight:
                sorted_gaslight = sorted(
                    recent_gaslight,
                    key=lambda x: self._get_thought_score(x, 'gaslight'),
                    reverse=True
                )
               
                keep_gaslight = sorted_gaslight[:target_gaslight]
       
        # ===========================================================
        # Гарантия минимального количества каждого типа
        # ===========================================================
        MIN_PER_TYPE = 5
       
        if len(keep_positive) < MIN_PER_TYPE and len(positive_thoughts) >= MIN_PER_TYPE:
            remaining = [t for t in positive_thoughts if t not in keep_positive]
            if remaining:
                additional = sorted(remaining,
                                  key=lambda x: x.get('delta', 0),
                                  reverse=True)[:MIN_PER_TYPE - len(keep_positive)]
                keep_positive.extend(additional)
       
        if len(keep_hard_negative) < MIN_PER_TYPE and len(hard_negative_thoughts) >= MIN_PER_TYPE:
            remaining = [t for t in hard_negative_thoughts if t not in keep_hard_negative]
            if remaining:
                additional = sorted(remaining,
                                  key=lambda x: x.get('similarity_to_original', 0),
                                  reverse=True)[:MIN_PER_TYPE - len(keep_hard_negative)]
                keep_hard_negative.extend(additional)
       
        if len(keep_gaslight) < MIN_PER_TYPE and len(gaslight_thoughts) >= MIN_PER_TYPE:
            remaining = [t for t in gaslight_thoughts if t not in keep_gaslight]
            if remaining:
                additional = sorted(remaining,
                                  key=lambda x: abs(x.get('original_reward', 0.9) - x.get('reward', -0.9)),
                                  reverse=True)[:MIN_PER_TYPE - len(keep_gaslight)]
                keep_gaslight.extend(additional)
       
        # ===========================================================
        # 🔥 безопасная дедупликация по step
        # ===========================================================
        all_candidates = keep_positive + keep_hard_negative + keep_gaslight
       
        if len(all_candidates) < keep_count * 0.5:
            logger.warning(f"⚠ Мало отобранных мыслей: {len(all_candidates)}/{keep_count}")
            
            all_candidates = []
            
            if positive_thoughts:
                best_pos = sorted(positive_thoughts,
                                key=lambda x: self._get_thought_score(x, 'positive'),
                                reverse=True)[:keep_count//3]
                all_candidates.extend(best_pos)
            
            if hard_negative_thoughts:
                best_hn = sorted(hard_negative_thoughts,
                               key=lambda x: self._get_thought_score(x, 'hard_negative'),
                               reverse=True)[:keep_count//3]
                all_candidates.extend(best_hn)
            
            if gaslight_thoughts:
                best_gl = sorted(gaslight_thoughts,
                               key=lambda x: self._get_thought_score(x, 'gaslight'),
                               reverse=True)[:keep_count//3]
                all_candidates.extend(best_gl)
            
            # 🔥 ИСПРАВЛЕНИЕ 9: дедупликация по step (более надёжно)
            unique_by_step = {}
            for t in all_candidates:
                step_key = t.get('step')
                if step_key is not None:
                    # Если step есть, используем его как ключ
                    if step_key not in unique_by_step:
                        unique_by_step[step_key] = t
                else:
                    # Если step нет, используем tuple из основных полей
                    fallback_key = (t.get('type'), t.get('timestamp'), t.get('snapshot_norm'))
                    if fallback_key not in unique_by_step:
                        unique_by_step[fallback_key] = t
            
            all_candidates = list(unique_by_step.values())
            sorted_thoughts = all_candidates[:keep_count]
        else:
            # 🔥 ИСПРАВЛЕНИЕ 9: дедупликация и для основного случая
            unique_by_step = {}
            for t in all_candidates:
                step_key = t.get('step')
                if step_key is not None:
                    if step_key not in unique_by_step:
                        unique_by_step[step_key] = t
                else:
                    fallback_key = (t.get('type'), t.get('timestamp'), t.get('snapshot_norm'))
                    if fallback_key not in unique_by_step:
                        unique_by_step[fallback_key] = t
            
            sorted_thoughts = list(unique_by_step.values())
            
            # Если после дедупликации всё ещё нужно больше мыслей, добавляем из резерва
            if len(sorted_thoughts) < keep_count:
                remaining_count = keep_count - len(sorted_thoughts)
                all_thoughts = positive_thoughts + hard_negative_thoughts + gaslight_thoughts
                existing_steps = {t.get('step') for t in sorted_thoughts if t.get('step') is not None}
                
                for t in all_thoughts:
                    if len(sorted_thoughts) >= keep_count:
                        break
                    step = t.get('step')
                    if step is None or step not in existing_steps:
                        sorted_thoughts.append(t)
                        if step is not None:
                            existing_steps.add(step)
        
        return sorted_thoughts
        
    def _log_detailed_cleaning_stats(self, positive_thoughts, hard_negative_thoughts, gaslight_thoughts):
        try:
            logger.info("📊 " + "═" * 70)
            logger.info("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ОЧИСТКИ МЫСЛЕЙ")
            logger.info("📊 " + "═" * 70)
           
            all_thoughts = positive_thoughts + hard_negative_thoughts + gaslight_thoughts
           
            if not all_thoughts:
                logger.info("📊 Нет мыслей для анализа")
                return
           
            age_groups = {
                'младенцы (0-2)': 0,
                'юные (3-5)': 0,
                'зрелые (6-10)': 0,
                'ветераны (11+)': 0
            }
           
            for t in all_thoughts:
                age = t.get('age', 0)
                if age <= 2:
                    age_groups['младенцы (0-2)'] += 1
                elif age <= 5:
                    age_groups['юные (3-5)'] += 1
                elif age <= 10:
                    age_groups['зрелые (6-10)'] += 1
                else:
                    age_groups['ветераны (11+)'] += 1
           
            logger.info("📊 ВОЗРАСТНОЙ СОСТАВ ВСЕХ МЫСЛЕЙ:")
            for group, count in age_groups.items():
                percentage = (count / len(all_thoughts)) * 100 if len(all_thoughts) > 0 else 0
                logger.info(f" {group}: {count} мыслей ({percentage:.1f}%)")
           
            if positive_thoughts:
                logger.info("📊 " + "─" * 60)
                logger.info("📊 АНАЛИЗ ЭЛИТНЫХ МЫСЛЕЙ:")
               
                top_positive = sorted(positive_thoughts,
                                     key=lambda x: x.get('delta', 0),
                                     reverse=True)[:5]
               
                for i, thought in enumerate(top_positive):
                    logger.info(f" #{i+1}: Δ={thought.get('delta',0):.3f} "
                              f"🤯={thought.get('surprise',0):.3f} "
                              f"🎯={thought.get('reward',0):.3f} "
                              f"возраст={thought.get('age',0)} "
                              f"состояние={thought.get('state','unknown')}")
               
                deltas = [t.get('delta', 0) for t in positive_thoughts]
                surprises = [t.get('surprise', 0) for t in positive_thoughts]
                rewards = [t.get('reward', 0) for t in positive_thoughts]
               
                logger.info(f" 📈 СРЕДНИЕ: Δ={np.mean(deltas):.3f} "
                          f"🤯={np.mean(surprises):.3f} "
                          f"🎯={np.mean(rewards):.3f}")
               
                elite_count = sum(1 for t in positive_thoughts if t.get('delta', 0) >= 0.6)
                elite_percent = (elite_count / len(positive_thoughts)) * 100
                logger.info(f" 🏆 ЭЛИТА (Δ≥0.6): {elite_count}/{len(positive_thoughts)} ({elite_percent:.1f}%)")
           
            if hard_negative_thoughts:
                logger.info("📊 " + "─" * 60)
                logger.info("📊 АНАЛИЗ HARD NEGATIVES:")
               
                methods_count = {}
                for t in hard_negative_thoughts:
                    method = t.get('method', 'unknown')
                    methods_count[method] = methods_count.get(method, 0) + 1
               
                if methods_count:
                    methods_str = " | ".join([f"{m}:{c}" for m, c in methods_count.items()])
                    logger.info(f" 🔧 Методы: {methods_str}")
           
            logger.info("📊 " + "═" * 70)
        except Exception as e:
            logger.debug(f"⚠ Ошибка в _log_detailed_cleaning_stats: {e}")
            
    def _get_thought_score(self, thought, thought_type):
        """🔥 УЛУЧШЕННАЯ СИСТЕМА БАЛЛОВ С УЧЕТОМ ВОЗРАСТА И СЛОЖНОСТИ"""
        base_score = 0.0
       
        if thought_type == 'positive':
            delta = thought.get('delta', 0)
            surprise = thought.get('surprise', 0)
            max_age = self.max_age_positive
           
            if delta > 0.8 and surprise > 0.4:
                max_age = int(max_age * 1.5)
            elif delta < 0.4:
                max_age = int(max_age * 0.5)
           
            current_age = thought.get('age', 0)
            age_factor = max(0.3, 1.0 - (current_age / max(max_age, 1)))
           
            motivation = thought.get('motivation', 0)
           
            base_score = (
                delta * 2.5 +
                surprise * 2.0 +
                motivation * 1.0 +
                (1.0 / (current_age + 1)) * 0.5
            )
           
            if delta < 0.6:
                base_score *= 0.5
           
            if surprise > 0.4:
                base_score *= 1.5
           
            state = thought.get('state', 'neutral')
            if state == "engaged":
                base_score *= 1.2
            elif state == "shocked":
                base_score *= 1.5
           
            if current_age > max_age:
                base_score *= 0.1
           
            return base_score * age_factor
           
        elif thought_type == 'hard_negative':
            similarity = thought.get('similarity_to_original', 0)
           
            max_age = thought.get('max_age', int(similarity * self.max_age_hard_negative))
            current_age = thought.get('age', 0)
           
            if current_age > max_age:
                return 0.0
           
            base_score = (
                similarity * 2.0 +
                abs(thought.get('reward', 0) - 0.5) +
                (max_age - current_age) / max(max_age, 1) * 0.5
            )
           
            method = thought.get('method', 'unknown')
            if method == 'chimera':
                base_score *= 1.2
           
            method_counts = {}
            thoughts = self.data_manager.get_thoughts_slice()
            hard_negatives = [t for t in thoughts if t.get('type') == 'hard_negative']
            for hn in hard_negatives:
                m = hn.get('method', 'unknown')
                method_counts[m] = method_counts.get(m, 0) + 1
           
            total_hn = len(hard_negatives)
            if total_hn > 0 and method in method_counts:
                method_freq = method_counts[method] / total_hn
                diversity_bonus = (1.0 - method_freq) * 0.3
                base_score += diversity_bonus
           
            return base_score
               
        elif thought_type == 'gaslight':
            original_reward = thought.get('original_reward', 0.9)
            gaslight_reward = thought.get('reward', -0.9)
            dissonance_strength = abs(original_reward - gaslight_reward)
           
            max_age = thought.get('max_age', int(dissonance_strength * self.max_age_gaslight))
            current_age = thought.get('age', 0)
           
            if current_age > max_age:
                return 0.0
           
            base_score = (
                dissonance_strength * 3.0 +
                (max_age - current_age) / max(max_age, 1) * 1.0
            )
           
            method = thought.get('gaslight_method', 'direct')
            if method == 'subtle_gaslight':
                base_score *= 1.3
           
            return base_score
       
        return base_score
   
    def _log_controller_influence(self, applied_controls):
        """🔥 Логирование влияния контроллера на работу памяти"""
        thoughts = self.data_manager.get_thoughts_slice()
       
        if len(thoughts) < 5:
            return
       
        recent_thoughts = thoughts[-5:]
       
        avg_mood_influence = np.mean([t.get('mood_influence', 0) for t in recent_thoughts])
        avg_curiosity_threshold = np.mean([t.get('curiosity_threshold', self.curiosity_threshold)
                                          for t in recent_thoughts])
       
        logger.info(f"🎛️ Влияние контроллера: mood={avg_mood_influence:.3f}, " +
                   f"curiosity_threshold={avg_curiosity_threshold:.3f}")
       
        positive_thoughts = [t for t in recent_thoughts if t.get('type') == 'curiosity_insight']
        if positive_thoughts:
            avg_delta = np.mean([t.get('delta', 0) for t in positive_thoughts])
            avg_surprise = np.mean([t.get('surprise', 0) for t in positive_thoughts])
           
            logger.info(f"📊 Эффективность: Δ={avg_delta:.3f}, 🤯={avg_surprise:.3f}")
           
            if abs(avg_mood_influence) > 0.1:
                if avg_mood_influence > 0 and avg_delta > 0.4:
                    logger.info(f"📈 Позитивное настроение → высокое качество мыслей!")
                elif avg_mood_influence < 0 and avg_delta < 0.3:
                    logger.info(f"📉 Негативное настроение → снижение качества мыслей")
                    
# ===================================================================
# SAVE / LOAD
# ===================================================================
    def save_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        import os
        import torch
        serializable_thoughts = []
        for thought in self.data_manager.thought_chains:
            serialized = thought.copy()
            if 'snapshot' in serialized and isinstance(serialized['snapshot'], torch.Tensor):
                serialized['snapshot'] = serialized['snapshot'].cpu().tolist()
            if 'snapshot_norm' in serialized and isinstance(serialized['snapshot_norm'], torch.Tensor):
                serialized['snapshot_norm'] = serialized['snapshot_norm'].cpu().item()
            serializable_thoughts.append(serialized)
        
        cpu_centroid_memory = self.centroid_memory.cpu()
        centroid_state = cpu_centroid_memory.state_dict_custom()
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'mamba_writer_state': self.mamba_writer_state.cpu() if self.mamba_writer_state is not None else None,
            'mamba_reader_state': self.mamba_reader_state.cpu() if self.mamba_reader_state is not None else None,
            'curiosity_state': self.curiosity.to_dict(),
            'step_count': self.step_count,
            'sleep_cycles': self.sleep_cycles,
            'training_losses': getattr(self, 'training_losses', []),
            'mamba_writer_ema': self.mamba_writer_ema,
            'mamba_reader_ema': self.mamba_reader_ema,
            'thought_chains': serializable_thoughts,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            'writer_insight_threshold': self.writer_insight_threshold,
            'max_age_hard_negative': self.max_age_hard_negative,
            'max_age_gaslight': self.max_age_gaslight,
            'max_age_positive': self.max_age_positive,
            'recovery_level': self.recovery_level,
            'recovery_success_streak': self.recovery_success_streak,
            'recovery_failure_streak': self.recovery_failure_streak,
            'gaslight_prob': self.gaslight_prob,
            'hard_negative_stats': self.hard_negative_stats,
            'recovered_thoughts_history': self.recovered_thoughts_history,
            'centroid_memory_state': centroid_state,
            'device_type': str(self.config.device),
            'recovery_history': {
                'level': self.recovery_level,
                'success_streak': self.recovery_success_streak,
                'failure_streak': self.recovery_failure_streak,
                'recovered_thoughts': self.recovered_thoughts_history,
            },
            'control_weights': {k: v.item() for k, v in self.control_weights.items()},
            'hard_negative_stats': self.hard_negative_stats,
            'config_updates': self.config_updates,
            #  Сохраняем текущее состояние контрастного лосса
            'contrastive_loss_margin': self.contrastive_loss_fn._current_margin.item() if hasattr(self, 'contrastive_loss_fn') else None,
            'memory_gate': self.memory_gate.item() if hasattr(self, 'memory_gate') else None,  # для обратной совместимости
            
            # ===========================================================
            # 🔥 MOTIVATION MODULE + GATE STATE
            # ===========================================================
            'motivation_module_state': self.motivation_module.state_dict() if hasattr(self, 'motivation_module') else None,
            'gate_state': {
                'memory_gate_logit': self.memory_gate_logit.data.cpu().clone() if hasattr(self, 'memory_gate_logit') else None,
                'memory_gate_ema': self.memory_gate_ema.cpu().clone() if hasattr(self, 'memory_gate_ema') else None,
                'gate_readiness_weight': self.gate_readiness_weight.data.cpu().clone() if hasattr(self, 'gate_readiness_weight') else None,
                'gate_curiosity_weight': self.gate_curiosity_weight.data.cpu().clone() if hasattr(self, 'gate_curiosity_weight') else None,
                'gate_confidence_weight': self.gate_confidence_weight.data.cpu().clone() if hasattr(self, 'gate_confidence_weight') else None,
                'gate_motivation_weight': self.gate_motivation_weight.data.cpu().clone() if hasattr(self, 'gate_motivation_weight') else None,
                'gate_aux_coef': self.gate_aux_coef.data.cpu().clone() if hasattr(self, 'gate_aux_coef') else None,
            },
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
       
        self.centroid_memory = self.centroid_memory.to(self.config.device)
       
        logger.info(f"💾 Checkpoint сохранён: {checkpoint_path} (центроидная память включена)")
        if 'motivation_module_state' in checkpoint and checkpoint['motivation_module_state']:
            logger.info(f"🔥 MotivationModule сохранён")
        if 'gate_state' in checkpoint:
            logger.info(f"🚪 Memory Gate состояние сохранено")
        

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        import os
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint не найден: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        device = self.config.device
        
        # Загружаем основное состояние модели
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Mamba состояния
        if 'mamba_writer_state' in checkpoint and checkpoint['mamba_writer_state'] is not None:
            self.mamba_writer_state = checkpoint['mamba_writer_state'].to(device)
        if 'mamba_reader_state' in checkpoint and checkpoint['mamba_reader_state'] is not None:
            self.mamba_reader_state = checkpoint['mamba_reader_state'].to(device)
        
        # Curiosity
        if 'curiosity_state' in checkpoint:
            self.curiosity.from_dict(checkpoint['curiosity_state'])
        
        # Счётчики
        self.step_count = checkpoint.get('step_count', 0)
        self.sleep_cycles = checkpoint.get('sleep_cycles', 0)
        self.training_losses = checkpoint.get('training_losses', [])
        self.mamba_writer_ema = checkpoint.get('mamba_writer_ema', 0.05)
        self.mamba_reader_ema = checkpoint.get('mamba_reader_ema', 0.05)
        
        # Centroid memory
        if 'centroid_memory_state' in checkpoint:
            try:
                centroid_state = checkpoint['centroid_memory_state']
               
                self.centroid_memory.load_state_dict_custom(centroid_state)
               
                self.centroid_memory = self.centroid_memory.to(device)
               
                if hasattr(self.centroid_memory, 'validate_cache_shape'):
                    self.centroid_memory.validate_cache_shape()
                    logger.debug("🔍 Кэш центроидной памяти валидирован")
                else:
                    self.centroid_memory._cache_dirty = True
                    logger.debug("🔍 Кэш центроидной памяти помечен как грязный")
               
                stats = self.centroid_memory.get_stats()
                logger.info(f"✅ CentroidMemoryManager загружен: {stats['active_slots']}/{stats['total_slots']} слотов")
               
                if hasattr(self.centroid_memory, 'linker_status'):
                    self.centroid_memory.linker_status()
                   
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки центроидной памяти: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠ В чекпоинте нет centroid_memory_state")
        
        # Thought chains
        if 'thought_chains' in checkpoint:
            thought_chains = checkpoint['thought_chains']
            self.data_manager.thought_chains = thought_chains
            logger.info(f"✅ Загружено {len(thought_chains)} мыслей из checkpoint")
        else:
            self.data_manager.thought_chains = []
            logger.warning("⚠ В чекпоинте нет thought_chains")
        
        # Параметры
        self.writer_insight_threshold = checkpoint.get('writer_insight_threshold', 0.5)
        self.max_age_hard_negative = checkpoint.get('max_age_hard_negative', 25)
        self.max_age_gaslight = checkpoint.get('max_age_gaslight', 15)
        self.max_age_positive = checkpoint.get('max_age_positive', 20)
        self.recovery_level = checkpoint.get('recovery_level', 0)
        self.recovery_success_streak = checkpoint.get('recovery_success_streak', 0)
        self.recovery_failure_streak = checkpoint.get('recovery_failure_streak', 0)
        self.gaslight_prob = checkpoint.get('gaslight_prob', 0.25)
        self.hard_negative_stats = checkpoint.get('hard_negative_stats', {})
        self.recovered_thoughts_history = checkpoint.get('recovered_thoughts_history', [])
       
        # Контрастный loss
        if hasattr(self, 'contrastive_loss_fn') and 'contrastive_loss_margin' in checkpoint and checkpoint['contrastive_loss_margin'] is not None:
            self.contrastive_loss_fn._current_margin.fill_(checkpoint['contrastive_loss_margin'])
            logger.info(f"📊 Восстановлен margin контрастного лосса: {checkpoint['contrastive_loss_margin']:.4f}")
        
        # Optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
        # Controller optimizer
        if 'controller_optimizer_state_dict' in checkpoint and hasattr(self, 'personality_core') and self.personality_core is not None:
            try:
                opt_state = checkpoint['controller_optimizer_state_dict']
               
                device = next(self.personality_core.controller.parameters()).device
                self.personality_core._ensure_optimizers(device)
               
                self.personality_core.controller_optimizer.load_state_dict(opt_state)
               
                for param_group in self.personality_core.controller_optimizer.param_groups:
                    for param in param_group['params']:
                        if param in self.personality_core.controller_optimizer.state:
                            state = self.personality_core.controller_optimizer.state[param]
                            for key, value in state.items():
                                if isinstance(value, torch.Tensor) and value.device != device:
                                    state[key] = value.to(device)
               
                logger.info(f"✅ Состояние оптимизатора контроллера загружено и перемещено на {device}")
            except Exception as e:
                logger.warning(f"⚠ Ошибка загрузки оптимизатора контроллера: {e}")
       
        # Scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # ===========================================================
        # 🔥 ЗАГРУЗКА MOTIVATION MODULE + GATE
        # ===========================================================

        if 'motivation_module_state' in checkpoint and checkpoint['motivation_module_state'] is not None:
            if hasattr(self, 'motivation_module'):
                try:
                    self.motivation_module.load_state_dict(checkpoint['motivation_module_state'], strict=False)
                    self.motivation_module = self.motivation_module.to(device)
                    logger.info("✅ MotivationModule загружен")
                except Exception as e:
                    logger.warning(f"⚠ Ошибка загрузки MotivationModule: {e}")
                    # Создаём новый при ошибке
                    if not hasattr(self, 'motivation_module'):
                        self.motivation_module = MotivationModule(self.config)
                        self.motivation_module = self.motivation_module.to(device)
                        logger.info("🔥 Создан новый MotivationModule (ошибка загрузки)")
            else:
                logger.warning("⚠ MotivationModule не найден в модели, но есть в чекпоинте. Создаём новый.")
                self.motivation_module = MotivationModule(self.config)
                self.motivation_module = self.motivation_module.to(device)
        else:
            # 🔥 ИСПРАВЛЕНИЕ 15: создаём MotivationModule если его нет в чекпоинте
            if not hasattr(self, 'motivation_module'):
                self.motivation_module = MotivationModule(self.config)
                self.motivation_module = self.motivation_module.to(device)
                logger.info("🔥 Создан новый MotivationModule (не было в чекпоинте)")
        
        # ===========================================================
        # 🔥 ИСПРАВЛЕНИЕ: безопасная конвертация memory_gate
        # ===========================================================
        if 'memory_gate' in checkpoint and checkpoint['memory_gate'] is not None:
            if hasattr(self, 'memory_gate_logit'):
                old_gate = checkpoint['memory_gate']
                # 🔥 Безопасная конвертация
                if isinstance(old_gate, (int, float)):
                    old_gate = float(old_gate)
                    # Защита от граничных значений
                    if old_gate <= 0.0:
                        old_gate = 0.05
                    elif old_gate >= 1.0:
                        old_gate = 0.95
                    
                    # Безопасное вычисление logit
                    old_gate_clipped = np.clip(old_gate, 0.01, 0.99)
                    old_logit = np.log(old_gate_clipped / (1.0 - old_gate_clipped + 1e-8))
                    self.memory_gate_logit.data.fill_(old_logit)
                    if hasattr(self, 'memory_gate_ema'):
                        self.memory_gate_ema.fill_(old_gate)
                    logger.info(f"🔄 Конвертирован старый memory_gate ({old_gate:.4f}) в новый формат")
                else:
                    logger.warning(f"⚠ Неожиданный тип memory_gate: {type(old_gate)}")
            elif hasattr(self, 'memory_gate'):
                self.memory_gate.data.fill_(checkpoint['memory_gate'])
                logger.info(f"🔙 Загружен старый memory_gate: {self.memory_gate.item():.4f}")
        
        # Загрузка Gate состояния (новый формат)
        if 'gate_state' in checkpoint:
            gs = checkpoint['gate_state']
            device = next(self.parameters()).device
            
            # Загружаем logit (основной параметр)
            if 'memory_gate_logit' in gs and gs['memory_gate_logit'] is not None:
                if hasattr(self, 'memory_gate_logit'):
                    self.memory_gate_logit.data.copy_(gs['memory_gate_logit'].to(device))
                    logger.info(f"✅ Загружен memory_gate_logit: {self.memory_gate_logit.item():.4f}")
            
            # Загружаем EMA
            if 'memory_gate_ema' in gs and gs['memory_gate_ema'] is not None:
                if hasattr(self, 'memory_gate_ema'):
                    self.memory_gate_ema.copy_(gs['memory_gate_ema'].to(device))
                    logger.info(f"✅ Загружена memory_gate_ema: {self.memory_gate_ema.item():.4f}")
            
            # Загружаем веса для метрик
            gate_weight_keys = [
                'gate_readiness_weight',
                'gate_curiosity_weight', 
                'gate_confidence_weight',
                'gate_motivation_weight',
                'gate_aux_coef'
            ]
            
            for key in gate_weight_keys:
                if key in gs and gs[key] is not None and hasattr(self, key):
                    getattr(self, key).data.copy_(gs[key].to(device))
                    logger.info(f"✅ Загружен {key}: {getattr(self, key).item():.4f}")
            
            logger.info("✅ Dynamic Memory Gate успешно загружен")
        
        logger.info(f"✅ Checkpoint загружен: {checkpoint_path}")
        logger.info(f"📊 Step: {self.step_count}, Sleep cycles: {self.sleep_cycles}")
        return True
 
# ===================================================================
# 🔥 MOTIVATION MODULE — Внутренняя система мотивации v1.0
# ===================================================================
class MotivationModule(nn.Module):
    """
    Простая, но эффективная система внутренних наград.
    Влияет на write_gate, memory_gate и скорость обучения.
    """
    def __init__(self, config):
        super().__init__()
        
        self.dim = config.n_embd
        
        # Оценщик качества текущего состояния
        self.evaluator = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Весовые коэффициенты разных видов награды
        self.novelty_weight     = nn.Parameter(torch.tensor(0.45))
        self.consistency_weight = nn.Parameter(torch.tensor(0.30))
        self.curiosity_weight   = nn.Parameter(torch.tensor(0.25))
        
        # EMA мотивации (для стабильности)
        self.register_buffer('motivation_ema', torch.tensor(0.5))
        
        logger.info("🔥 MotivationModule активирован")

    def forward(self, hidden_state: torch.Tensor, surprise: float = 0.0, curiosity_level: float = 0.5):
        """
        hidden_state: [batch, dim] или [batch, seq, dim]
        surprise и curiosity_level — из CuriositySystem / MetaPredictor
        """
        # Усредняем по последовательности
        if hidden_state.dim() == 3:
            x = hidden_state.mean(dim=1)
        else:
            x = hidden_state
            
        quality_score = self.evaluator(x).squeeze(-1)          # [batch]
        
        # 🔥 нормализация весов
        weights = torch.softmax(torch.stack([
            self.novelty_weight, 
            self.consistency_weight, 
            self.curiosity_weight
        ]), dim=0)
        
        w_nov = weights[0]
        w_cons = weights[1]
        w_cur = weights[2]
        
        motivation = (
            surprise * w_nov +
            quality_score * w_cons +
            curiosity_level * w_cur
        )
        
        # EMA для плавности
        with torch.no_grad():
            self.motivation_ema.mul_(0.92).add_(motivation.mean() * 0.08)
        
        motivation = torch.clamp(motivation.mean(), 0.0, 1.0)
        
        self._last_motivation = motivation.detach()
        
        return motivation
 
# ===================================================================
# Если нужно — можно добавить тестовый запуск
# ===================================================================
if __name__ == "__main__":
    logger.info("🧠 memory_heads.py v11.1 с исправлениями для batch>1 и персистентным лоссом загружен успешно")
