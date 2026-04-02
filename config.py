# -*- coding: utf-8 -*-
# config.py - ThaliaConfig v9.1.1 (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)
from transformers import GPT2Config
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import json

logger = logging.getLogger('ThaliaConfig')

class ThaliaConfig(GPT2Config):
    model_type = "thalia"
    
    def __init__(
        self,
        # БАЗОВЫЕ GPT-2 ПАРАМЕТРЫ
        vocab_size: int = 50257,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.15,
        embd_pdrop: float = 0.2,
        attn_pdrop: float = 0.15,
        layer_norm_epsilon: float = 1e-05,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: Optional[int] = None,
        
        # ПАРАМЕТРЫ ПАМЯТИ (группировка)
        slot_size: Optional[int] = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        reader_ema_alpha: float = 0.95,
        writer_ema_alpha: float = 0.98,
        writer_insight_threshold: float = 0.5,
        writer_quality_threshold: float = 0.12,
        
        # БЛОКНОТ
        notebook_size: int = 300,
        notebook_save_path: str = "thought_chains.json",
        min_thoughts_for_sleep: int = 50,
        min_thoughts_for_emergency_sleep: int = 100,
        min_quality_thoughts: int = 30,
        clear_keep_last: int = 40,
        
        # ПАРАМЕТРЫ СНА
        sleep_lr: float = 6e-5,
        sleep_epochs: int = 10,
        sleep_max_epochs: int = 15,
        sleep_patience: int = 8,
        sleep_batch_size: int = 8,
        sleep_grad_clip: float = 0.8,
        sleep_weight_decay: float = 0.0,
        sleep_noise_scale: float = 0.08,
        sleep_state_norm_max: float = 10.0,
        sleep_loss_weights: Optional[Dict[str, float]] = None,
        
        # ЛЮБОПЫТСТВО
        curiosity_threshold: float = 0.12,
        predictor_coef: float = 0.15,
        use_combo_loss: bool = True,
        curiosity_beta: float = 0.95,
        curiosity_habituation_decay: float = 0.985,
        curiosity_arousal_decay: float = 0.88,
        curiosity_trace_decay: float = 0.97,
        curiosity_weight_multiplier: float = 3.0,
        
        # НОРМАЛИЗАЦИЯ
        state_norm_threshold: float = 8.0,
        snapshot_norm_epsilon: float = 1e-8,
        
        # ЧАСТОТА ОБНОВЛЕНИЙ
        synchronous_sleep: bool = True,
        memory_update_freq: int = 2,
        sleep_check_freq: int = 50,
        status_report_freq: int = 100,
        log_freq: int = 100,
        verbose_logs: bool = True,
        
        # РЕДКО ИСПОЛЬЗУЕМЫЕ (можно оставить для совместимости)
        duplicate_threshold: float = 0.85,
        buffer_size: int = 2000,
        scheduler_T_0: int = 10,
        scheduler_eta_min: float = 1e-5,
        
        # 🔥 КИБЕР-ПСИХИКА
        use_psyche_core: bool = True,
        use_living_layer: bool = True,
        psyche_coef: float = 0.03,
        psyche_influence_weight: float = 0.35,
        psyche_loss_weights: Optional[Dict[str, float]] = None,
        
        # 🔥 НЕЙРОКОНТРОЛЛЕР
        controller_lr: float = 0.001,
        controller_input_dim: Optional[int] = None,
        controller_output_dim: int = 10,
        controller_hidden_dim: int = 32,
        controller_gamma: float = 0.95,
        controller_lambda: float = 0.95,
        controller_clip_epsilon: float = 0.2,
        controller_value_coef: float = 0.5,
        controller_entropy_coef: float = 0.01,
        controller_exploration_init: float = 1.0,
        controller_output_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        
        # 🔥 ХАРАКТЕР ПСИХИКИ
        trait_names: Optional[List[str]] = None,
        drive_names: Optional[List[str]] = None,
        initial_trait_values: Optional[Dict[str, float]] = None,
        initial_drive_values: Optional[Dict[str, float]] = None,
        
        # 🔥 СИСТЕМА ЦЕЛЕЙ
        goal_system_enabled: bool = True,
        num_goals: int = 5,
        
        # 🔥 ПАРАМЕТРЫ СТАБИЛЬНОСТИ
        stability_threshold: float = 0.35,
        overload_recovery_rate: float = 0.15,
        radicalization_max: float = 2.5,
        
        # 🔥 УЛУЧШЕННАЯ ПАМЯТЬ
        consolidation_ratio: float = 0.6,
        min_consolidate: int = 8,
        importance_weighting: bool = True,
        enhanced_forgetting: bool = True,
        
        # 🔥 СИСТЕМА СТАРЕНИЯ
        age_decay_factor: float = 0.1,
        max_age_hard_negative: int = 25,
        max_age_gaslight: int = 15,
        max_age_positive: int = 20,
        
        # 🔥 HARD NEGATIVES
        hard_negative_delta_threshold: float = 0.1,
        hard_negative_probability: float = 0.90,
        
        # 🔥 РАЗДЕЛЬНЫЕ ПАРАМЕТРЫ СХОЖЕСТИ (ВАЖНО!)
        min_similarity_chimera: float = 0.8,
        max_similarity_chimera: float = 0.95,
        min_similarity_twist: float = 0.4,
        max_similarity_twist: float = 0.9,
        min_similarity_lobotomy: float = -0.7,
        max_similarity_lobotomy: float = -0.2,
        min_similarity_gaslight: float = 0.1,
        max_similarity_gaslight: float = 0.3,
        
        shared_exchange_dim: int = 768,
        max_hard_negative_ratio: float = 0.5,
        gaslight_prob: float = 0.2,
        
        # 🔥 КОНТРАСТНЫЙ СОН
        use_contrastive_sleep: bool = True,
        contrastive_margin: float = 0.6,
        contrastive_alpha: float = 0.8,
        contrastive_weight: float = 0.3,
        adaptive_margin: bool = True,
        contrastive_min_positive: int = 10,
        contrastive_min_negative: int = 60,
        contrastive_margin_growth: float = 1.05,
        contrastive_min_margin: float = 0.1,
        contrastive_max_margin: float = 2.0,
        contrastive_keep_ratio: float = 0.3,
        contrastive_temperature: float = 0.05,
        
        # 🔥 СИМБИОЗ MAMBA-TRANSFORMER
        bidirectional_exchange: bool = True,
        exchange_strength: float = 0.4,
        max_experience_bank_size: int = 100,
        alignment_coef: float = 0.08,

        # 🔥 NEURAL LINKER
        enable_linker: bool = True,
        linker_lr: float = 0.001,
        linker_weight_decay: float = 0.01,
        linker_hidden_dim: Optional[int] = None,
        linker_batch_size: int = 32,
        linker_train_frequency: int = 5,
        
        # 🔥 ОСАДОЧНАЯ ПАМЯТЬ
        num_sedimentary_slots: int = 512,
        core_slots_count: int = 8,
        sedimentary_dim: Optional[int] = None,
        hebb_lr_base: float = 0.05,
        hebb_decay_base: float = 0.002,
        veteran_age_threshold: int = 7,
        veteran_delta_threshold: float = 0.65,
        cos_sim_threshold: float = 0.85,
        cluster_threshold_mult_range: Tuple[float, float] = (0.8, 1.2),
        
        # 🔥 HEBB-ПАМЯТЬ
        use_hebb_layers: bool = True,
        hebb_num_slots: int = 64,
        hebb_tau_update: float = 0.08,
        
        # 🔥 ДОБАВЛЕННЫЕ ПАРАМЕТРЫ ДЛЯ CentroidMemoryManager
        merge_threshold: float = 0.96,
        split_variance_threshold: float = 0.3,
        eviction_utility_threshold: float = 0.01,
        
        # ===========================================================
        # 🔥 МЕТА-КОГНИТИВНЫЙ ПРЕДИКТОР
        # ===========================================================
        
        # Размерность предиктора
        predictor_hidden_dim: int = 128,
        
        # Параметры истории
        max_batch_size: int = 64,
        meta_history_len: int = 20,
        
        # Параметры рефлексии
        max_reflection_steps: int = 2,
        reflection_temperature: float = 1.2,
        reflection_early_stop: float = 0.7,
        
        # Коэффициенты потерь
        meta_loss_coef: float = 0.05,
        uncertainty_weight_max: float = 3.0,
        
        # 🔥 ВЕРСИЯ
        thalia_version: str = "9.1.1",
        
        **kwargs
    ):
        # Вычисляем производные параметры
        if slot_size is None:
            slot_size = n_embd
        
        if sedimentary_dim is None:
            sedimentary_dim = slot_size
        
        if linker_hidden_dim is None:
            linker_hidden_dim = max(16, slot_size // 2)
        
        # Очистка kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
        
        # Инициализация родителя
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **clean_kwargs
        )
        
        # ПАМЯТЬ
        self.slot_size = slot_size
        
        # MAMBA
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_inner_dim = self.slot_size * self.mamba_expand
        
        # READER/WRITER
        self.reader_ema_alpha = reader_ema_alpha
        self.writer_ema_alpha = writer_ema_alpha
        self.writer_insight_threshold = writer_insight_threshold
        self.writer_quality_threshold = writer_quality_threshold
        
        # БЛОКНОТ
        self.notebook_size = notebook_size
        self.notebook_save_path = notebook_save_path
        self.min_thoughts_for_sleep = min_thoughts_for_sleep
        self.min_thoughts_for_emergency_sleep = min_thoughts_for_emergency_sleep
        self.min_quality_thoughts = min_quality_thoughts
        self.clear_keep_last = clear_keep_last
        
        # СОН
        self.sleep_lr = sleep_lr
        self.sleep_epochs = sleep_epochs
        self.sleep_max_epochs = sleep_max_epochs
        self.sleep_patience = sleep_patience
        self.sleep_batch_size = sleep_batch_size
        self.sleep_grad_clip = sleep_grad_clip
        self.sleep_weight_decay = sleep_weight_decay
        self.sleep_noise_scale = sleep_noise_scale
        self.sleep_state_norm_max = sleep_state_norm_max
        self.sleep_loss_weights = sleep_loss_weights or {
            'contrastive': 1.0,
            'reconstruction': 0.1,
            'curiosity': 0.1,
            'recovery': 0.2
        }
        
        # ЛЮБОПЫТСТВО
        self.curiosity_threshold = curiosity_threshold
        self.predictor_coef = predictor_coef
        self.use_combo_loss = use_combo_loss
        self.curiosity_beta = curiosity_beta
        self.curiosity_habituation_decay = curiosity_habituation_decay
        self.curiosity_arousal_decay = curiosity_arousal_decay
        self.curiosity_trace_decay = curiosity_trace_decay
        self.curiosity_weight_multiplier = curiosity_weight_multiplier
        
        # НОРМАЛИЗАЦИЯ
        self.state_norm_threshold = state_norm_threshold
        self.snapshot_norm_epsilon = snapshot_norm_epsilon
        
        # ЧАСТОТА
        self.synchronous_sleep = synchronous_sleep
        self.memory_update_freq = memory_update_freq
        self.sleep_check_freq = sleep_check_freq
        self.status_report_freq = status_report_freq
        self.log_freq = log_freq
        self.verbose_logs = verbose_logs
        
        # РЕДКО ИСПОЛЬЗУЕМЫЕ
        self.duplicate_threshold = duplicate_threshold
        self.buffer_size = buffer_size
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_eta_min = scheduler_eta_min
        
        # 🔥 ПСИХИКА
        self.use_psyche_core = use_psyche_core
        self.use_living_layer = use_living_layer
        self.psyche_coef = psyche_coef
        self.psyche_influence_weight = psyche_influence_weight
        
        self.psyche_loss_weights = psyche_loss_weights or {
            'range': 0.1,
            'extreme': 0.02,
            'tension': 0.06,
            'mood': 0.03,
            'opp': 0.008,
            'stability': 0.05,
            'change': 0.15,
            'diversity': 0.03,
            'efficacy': 0.1
        }
        
        # 🔥 НЕЙРОКОНТРОЛЛЕР
        self.controller_lr = controller_lr
        self.controller_input_dim = controller_input_dim
        self.controller_output_dim = controller_output_dim
        self.controller_hidden_dim = controller_hidden_dim
        self.controller_gamma = controller_gamma
        self.controller_lambda = controller_lambda
        self.controller_clip_epsilon = controller_clip_epsilon
        self.controller_value_coef = controller_value_coef
        self.controller_entropy_coef = controller_entropy_coef
        self.controller_exploration_init = controller_exploration_init
        
        # 🔥 ОБНОВЛЕННЫЕ RANGES
        self.controller_output_ranges = controller_output_ranges or {
            "curiosity_threshold": (0.05, 0.40),
            "recall_threshold_mod": (0.5, 1.5),
            "write_intensity": (0.5, 2.5),
            "noise_mod": (0.8, 2.0),
            "negative_sensitivity": (0.5, 1.5),
            "exploration_bonus": (0.0, 0.4),
            "learning_rate_mult": (0.7, 1.5),
            "hebb_write_gate": (0.0, 1.5),
            "hebb_lr_mult": (0.3, 2.0),
            "stability_factor": (0.1, 0.5),
        }
        
        # 🔥 ХАРАКТЕР
        self.trait_names = trait_names or [
            "curiosity", "depth", "empathy", "confidence", "rigidity",
            "creativity", "reflection", "ethics", "persistence",
            "arrogance", "order"
        ]
        self.drive_names = drive_names or [
            "novelty", "fatigue", "meaning", "competence", "identity", 
            "social", "share"
        ]
        
        self.initial_trait_values = initial_trait_values or {}
        self.initial_drive_values = initial_drive_values or {}
        
        # 🔥 ЦЕЛИ
        self.goal_system_enabled = goal_system_enabled
        self.num_goals = num_goals
        
        # 🔥 СТАБИЛЬНОСТЬ
        self.stability_threshold = stability_threshold
        self.overload_recovery_rate = overload_recovery_rate
        self.radicalization_max = radicalization_max
        
        # 🔥 УЛУЧШЕННАЯ ПАМЯТЬ
        self.consolidation_ratio = consolidation_ratio
        self.min_consolidate = min_consolidate
        self.importance_weighting = importance_weighting
        self.enhanced_forgetting = enhanced_forgetting
        
        # 🔥 СТАРЕНИЕ
        self.age_decay_factor = age_decay_factor
        self.max_age_hard_negative = max_age_hard_negative
        self.max_age_gaslight = max_age_gaslight
        self.max_age_positive = max_age_positive
        
        # 🔥 HARD NEGATIVES
        self.hard_negative_delta_threshold = hard_negative_delta_threshold
        self.hard_negative_probability = hard_negative_probability
        
        # 🔥 РАЗДЕЛЬНЫЕ ПАРАМЕТРЫ СХОЖЕСТИ
        self.min_similarity_chimera = min_similarity_chimera
        self.max_similarity_chimera = max_similarity_chimera
        self.min_similarity_twist = min_similarity_twist
        self.max_similarity_twist = max_similarity_twist
        self.min_similarity_lobotomy = min_similarity_lobotomy
        self.max_similarity_lobotomy = max_similarity_lobotomy
        self.min_similarity_gaslight = min_similarity_gaslight
        self.max_similarity_gaslight = max_similarity_gaslight
        
        self.shared_exchange_dim = shared_exchange_dim
        self.max_hard_negative_ratio = max_hard_negative_ratio
        self.gaslight_prob = gaslight_prob
        
        # 🔥 КОНТРАСТНЫЙ СОН
        self.use_contrastive_sleep = use_contrastive_sleep
        self.contrastive_margin = contrastive_margin
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_weight = contrastive_weight
        self.adaptive_margin = adaptive_margin
        self.contrastive_min_positive = contrastive_min_positive
        self.contrastive_min_negative = contrastive_min_negative
        self.contrastive_margin_growth = contrastive_margin_growth
        self.contrastive_min_margin = contrastive_min_margin
        self.contrastive_max_margin = contrastive_max_margin
        self.contrastive_keep_ratio = contrastive_keep_ratio
        self.contrastive_temperature = contrastive_temperature
        
        # 🔥 СИМБИОЗ
        self.bidirectional_exchange = bidirectional_exchange
        self.exchange_strength = exchange_strength
        self.max_experience_bank_size = max_experience_bank_size
        self.alignment_coef = alignment_coef

        # 🔥 LINKER
        self.enable_linker = enable_linker
        self.linker_lr = linker_lr
        self.linker_weight_decay = linker_weight_decay
        self.linker_hidden_dim = linker_hidden_dim
        self.linker_batch_size = linker_batch_size
        self.linker_train_frequency = linker_train_frequency
        
        # 🔥 ОСАДОЧНАЯ ПАМЯТЬ
        self.num_sedimentary_slots = num_sedimentary_slots
        self.core_slots_count = core_slots_count
        self.sedimentary_dim = sedimentary_dim
        self.hebb_lr_base = hebb_lr_base
        self.hebb_decay_base = hebb_decay_base
        self.veteran_age_threshold = veteran_age_threshold
        self.veteran_delta_threshold = veteran_delta_threshold
        self.cos_sim_threshold = cos_sim_threshold
        self.cluster_threshold_mult_range = cluster_threshold_mult_range
        
        # 🔥 HEBB
        self.use_hebb_layers = use_hebb_layers
        self.hebb_num_slots = hebb_num_slots
        self.hebb_tau_update = hebb_tau_update
        
        # 🔥 ДОБАВЛЕННЫЕ ПАРАМЕТРЫ
        self.merge_threshold = merge_threshold
        self.split_variance_threshold = split_variance_threshold
        self.eviction_utility_threshold = eviction_utility_threshold
        
        # ===========================================================
        # 🔥 МЕТА-КОГНИТИВНЫЙ ПРЕДИКТОР
        # ===========================================================
        self.predictor_hidden_dim = predictor_hidden_dim
        self.max_batch_size = max_batch_size
        self.meta_history_len = meta_history_len
        self.max_reflection_steps = max_reflection_steps
        self.reflection_temperature = reflection_temperature
        self.reflection_early_stop = reflection_early_stop
        self.meta_loss_coef = meta_loss_coef
        self.uncertainty_weight_max = uncertainty_weight_max
        
        # 🔥 ВЕРСИЯ
        self.thalia_version = thalia_version
        
        # Валидация
        self._validate_config()
    
    def _validate_config(self):
        """Компактная валидация"""
        try:
            # Базовые проверки
            assert self.slot_size > 0, "slot_size > 0"
            assert self.num_sedimentary_slots > 0, "num_sedimentary_slots > 0"
            
            # Размерности
            assert self.sedimentary_dim > 0, "sedimentary_dim > 0"
            assert self.shared_exchange_dim > 0, "shared_exchange_dim > 0"
            assert self.shared_exchange_dim <= self.n_embd, "shared_exchange_dim ≤ n_embd"
            
            # Пороги схожести
            assert self.min_similarity_chimera < self.max_similarity_chimera, "Chimera min < max"
            assert self.max_similarity_lobotomy <= 0.0, "Lobotomy max ≤ 0.0"
            
            # Количество слотов
            assert self.core_slots_count <= self.num_sedimentary_slots, "core_slots ≤ total"
            
            # Вероятности
            assert 0 <= self.hard_negative_probability <= 1, "hard_negative_probability in [0,1]"
            assert 0 <= self.gaslight_prob <= 0.5, "gaslight_prob in [0,0.5]"
            
            # Добавленные параметры
            assert 0 <= self.merge_threshold <= 1, "merge_threshold in [0,1]"
            assert self.split_variance_threshold > 0, "split_variance_threshold > 0"
            assert self.eviction_utility_threshold >= 0, "eviction_utility_threshold >= 0"
            
            # Мета-когнитивные параметры
            assert self.max_reflection_steps >= 0, "max_reflection_steps >= 0"
            assert 0 < self.reflection_temperature <= 2.0, "reflection_temperature in (0,2]"
            assert 0 < self.reflection_early_stop <= 1.0, "reflection_early_stop in (0,1]"
            assert 0 <= self.meta_loss_coef <= 1.0, "meta_loss_coef in [0,1]"
            assert self.uncertainty_weight_max >= 1.0, "uncertainty_weight_max >= 1.0"
            
        except AssertionError as e:
            logger.warning(f"⚠ Конфигурация требует внимания: {e}")
        
        logger.info(f"✅ ThaliaConfig v{self.thalia_version} загружен")
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        output = super().to_dict()
        
        # Только реально используемые параметры
        used_params = [
            'slot_size', 'num_sedimentary_slots', 'mamba_d_state',
            'writer_insight_threshold', 'notebook_size',
            'sleep_lr', 'sleep_epochs', 'sleep_batch_size',
            'curiosity_threshold', 'curiosity_beta',
            'use_psyche_core', 'psyche_coef', 'psyche_influence_weight',
            'controller_lr', 'controller_output_dim', 'controller_hidden_dim',
            'controller_gamma', 'controller_lambda', 'controller_clip_epsilon',
            'trait_names', 'drive_names', 'num_goals',
            'hard_negative_probability', 'gaslight_prob',
            'min_similarity_chimera', 'max_similarity_chimera',
            'min_similarity_lobotomy', 'max_similarity_lobotomy',
            'min_similarity_twist', 'max_similarity_twist',
            'min_similarity_gaslight', 'max_similarity_gaslight',
            'use_contrastive_sleep', 'contrastive_margin',
            'contrastive_min_negative', 'contrastive_temperature',
            'bidirectional_exchange', 'exchange_strength', 'max_experience_bank_size',
            'enable_linker', 'linker_lr', 'linker_batch_size',
            'core_slots_count', 'cos_sim_threshold',
            'merge_threshold', 'split_variance_threshold', 'eviction_utility_threshold',
            'buffer_size',
            'veteran_age_threshold', 'veteran_delta_threshold',
            'use_hebb_layers', 'hebb_num_slots', 'hebb_tau_update',
            'max_age_hard_negative', 'max_age_gaslight', 'max_age_positive',
            'predictor_hidden_dim', 'max_batch_size', 'meta_history_len',
            'max_reflection_steps', 'reflection_temperature', 'reflection_early_stop',
            'meta_loss_coef', 'uncertainty_weight_max',
            'thalia_version'
        ]
        
        for param in used_params:
            if hasattr(self, param):
                output[param] = getattr(self, param)
        
        if hasattr(self, 'controller_output_ranges'):
            output['controller_output_ranges'] = self.controller_output_ranges
        
        return output
    
    def save_pretrained(self, save_directory: str):
        """Сохранение конфига"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Конфиг v{self.thalia_version} сохранен в {config_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "ThaliaConfig":
        """Загрузка из словаря"""
        return cls(**{**config_dict, **kwargs})
    
    @classmethod
    def from_json_file(cls, json_file: str) -> "ThaliaConfig":
        """Загрузка из JSON файла"""
        import os
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Конфиг не найден: {json_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @property
    def device(self):
        """Для совместимости"""
        return getattr(self, '_device', None)
    
    @device.setter
    def device(self, value):
        self._device = value
    
    def __str__(self) -> str:
        """Краткое представление"""
        return (
            f"ThaliaConfig v{self.thalia_version}: "
            f"n_embd={self.n_embd}, n_layer={self.n_layer}, "
            f"psyche={'✅' if self.use_psyche_core else '❌'}, "
            f"hebb={'✅' if self.use_hebb_layers else '❌'}, "
            f"sedimentary={self.num_sedimentary_slots} slots, "
            f"controller={self.controller_output_dim} outputs, "
            f"meta={'✅' if self.max_reflection_steps > 0 else '❌'}"
        )
    
    @classmethod
    def get_default_config(cls) -> "ThaliaConfig":
        """Конфигурация по умолчанию"""
        return cls()
    
    @classmethod
    def get_small_config(cls) -> "ThaliaConfig":
        """Для тестов"""
        return cls(
            n_embd=256,
            n_layer=4,
            n_head=4,
            slot_size=128,
            notebook_size=50,
            num_sedimentary_slots=30,
            core_slots_count=3,
            use_hebb_layers=True,
            hebb_num_slots=16,
            controller_output_dim=5,
            max_reflection_steps=1,
            predictor_hidden_dim=64,
            meta_history_len=10
        )
    
    @classmethod
    def get_large_config(cls) -> "ThaliaConfig":
        """Для продакшена"""
        return cls(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            slot_size=512,
            notebook_size=500,
            num_sedimentary_slots=1024,
            core_slots_count=16,
            use_hebb_layers=True,
            hebb_num_slots=128,
            controller_output_dim=10,
            max_reflection_steps=3,
            predictor_hidden_dim=256,
            meta_history_len=32,
            reflection_temperature=1.0,
            meta_loss_coef=0.08
        )