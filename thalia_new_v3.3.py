# -*- coding: utf-8 -*-
import sys
import os
os.environ["TRITON_DISABLE"] = "1" 
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import argparse 
# КРИТИЧЕСКИ ВАЖНО: Очистка кэша ПЕРЕД импортами
modules_to_clear = [k for k in sys.modules.keys() if 'thalia' in k.lower() or 'modeling' in k.lower()]
for mod in modules_to_clear:
    del sys.modules[mod]
    print(f"Очищен кэш: {mod}")

# Добавляем текущую папку в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ОЧИСТКА КЭША ПЕРЕД ВСЕМИ ИМПОРТАМИ (существующий код)
if 'modeling_thalia' in sys.modules:
    del sys.modules['modeling_thalia']
import torch
torch.autograd.set_detect_anomaly(False)
print("🔍 Anomaly detection ENABLED - будет показана точная строка ошибки")
import json
import time
import re
import platform
import numpy as np
import glob
import hashlib
from typing import Optional
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
import logging
from contextlib import contextmanager
import shutil
import pickle
import locale
import unicodedata
from functools import partial 
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Adam 
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None 
import optuna.visualization as vis
try:
    import plotly
except ImportError:
    plotly = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    import pynvml
except ImportError:
    pynvml = None
import pandas as pd
import yaml
import random
import traceback
import zipfile
import math
import optuna

# ПРАВИЛЬНЫЙ ИМПОРТ Thalia ИЗ ЛОКАЛЬНОГО ФАЙЛА
try:
    from modeling_thalia import Thalia, ThaliaConfig
    THALIA_AVAILABLE = True
    print("🤖 Thalia загружена из modeling_thalia.py")
except ImportError as e:
    THALIA_AVAILABLE = False
    print(f"🤖 Thalia import failed: {e}")
    # Fallback to package if available
    try:
        from thalia import Thalia, ThaliaConfig
        THALIA_AVAILABLE = True
        print("🤖 Thalia загружена из пакета thalia")
    except ImportError:
        print("Thalia не установлена. Для использования установите пакет thalia")

# --- Lion Optimizer ---
try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None
    print("WARNING: Lion optimizer not available. Please install: pip install lion-pytorch")

# Добавлен для padding
from torch.nn.utils.rnn import pad_sequence

@dataclass
class TrainingConfig:

# ===================================================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ===================================================================

    epochs: int = 3 # Количество эпох обучения
    learning_rate: float = 1e-5 # Скорость обучения
    batch_size: Union[int, str] = 4 # Размер батча ("auto" для автоопределения)
    warmup_steps: int = 100 # Шагов для прогрева learning rate
    weight_decay: float = 0.01 # Вес для decay регуляризации
    gradient_clip_val: float = 0.5 # Максимальное значение градиента (обрезка)
    gradient_accumulation_steps: int = 4 # Накопление градиентов перед обновлением
    lr_scheduler_type: str = "cosine" # Тип шедулера обучения (linear/plateau)
    
# ===================================================================
# ОСНОВНЫЕ ПАРАМЕТРЫ ПУТЕЙ
# ===================================================================   
 
    model_path: Optional[str] = None # Путь к предобученной модели
    dataset_path: Optional[str] = None # Путь к датасету для обучения
    output_dir: str = "output" # Директория для сохранения результатов
    validation_dataset_path: Optional[str] = None # Путь к валидационному датасету
    recent_datasets: List[str] = field(default_factory=list) # История недавних датасетов
    
# ===================================================================
# ДИФФЕРЕНЦИАЛЬНЫЕ LEARNING RATES
# ===================================================================  
  
    use_differential_lr: bool = False # Использовать разные LR для разных слоев
    lr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "embeddings": 0.1, # Множитель для embedding слоев
        "bottom_layers": 0.2, # Множитель для нижних слоев (0-33%)
        "middle_layers": 0.5, # Множитель для средних слоев (33-66%)
        "top_layers": 1.0, # Множитель для верхних слоев (66-100%)
        "output_layers": 1.5 # Множитель для выходных слоев
    })
    
# ===================================================================
# ПАРАМЕТРЫ МОДЕЛИ И ТОКЕНИЗАЦИИ
# =================================================================== 
    
    max_length: int = 256 # Максимальная длина последовательности
    loss_type: str = "ForCausalLMLoss" # Тип функции потерь
    min_token_length: int = 5 # Минимальная длина токенов
    
# ===================================================================
# ПАРАМЕТРЫ ГЕНЕРАЦИИ 
# ===================================================================  
   
    temperature: float = 0.9 # Температура для сэмплирования (0.1-2.0)
    top_k: int = 50 # Top-k sampling (0-1000)
    top_p: float = 0.9 # Top-p (nucleus) sampling (0.0-1.0)
    repetition_penalty: float = 1.1 # Штраф за повторения (1.0-2.0)
    generation_max_length: int = 200 # Максимальная длина генерации
    
# ===================================================================
# ПАРАМЕТРЫ ДАТАСЕТА И ФОРМАТА
# ===================================================================
     
    validation_split_ratio: float = 0.2 # Доля данных для валидации
    dataset_format: str = "jsonl" # Формат датасета
    input_keys: List[str] = field(default_factory=lambda: ["input"]) # Ключи входных данных
    output_keys: List[str] = field(default_factory=lambda: ["output"]) # Ключи выходных данных
    dialog_roles: List[str] = field(default_factory=lambda: ["system", "user", "assistant"]) # Роли в диалоге
    strict_roles: bool = False # Строгая проверка ролей
    
# ===================================================================
# ДИАЛОГОВЫЕ ПАРАМЕТРЫ
# =================================================================== 
    
    dialog_separator: str = "\n" # Разделитель реплик в диалоге
    
# ===================================================================
# ОПТИМИЗАЦИЯ И УСКОРЕНИЕ
# ===================================================================
    
    optimizer_type: str = "adamw" # Тип оптимизатора (adamw/lion)
    use_mixed_precision: bool = True # Использовать mixed precision (AMP)
    use_bfloat16: bool = True # Использовать bfloat16 вместо float16
    
# ===================================================================
# ПАРАМЕТРЫ ОЧИСТКИ ТЕКСТА
# =================================================================== 
   
    normalize_unicode: bool = True # Нормализовать Unicode
    auto_clean_dataset: bool = True # Автоматическая очистка датасета
    
# ===================================================================
# ПАРАМЕТРЫ РАЗБИЕНИЯ НА ЧАНКИ
# ===================================================================   
 
    chunk_continuation_marker: str = " [>]" # Маркер продолжения
    chunk_end_marker: str = " [end]" # Маркер окончания
    show_chunk_markers: bool = True # Показывать ли маркеры
    adjust_chunk_boundaries: bool = True # Корректировать границы предложений
    max_boundary_adjustment: int = 100 # Макс. символов для поиска границы
    split_long_texts: bool = True # Разбивать длинные тексты
    overlap_ratio: float = 0.05 # Коэффициент перекрытия при разбиении текста
    
# ===================================================================
# ПАРАМЕТРЫ ЛОГИРОВАНИЯ И СОХРАНЕНИЯ
# ===================================================================  
  
    save_steps: int = 500 # Сохранять чекпоинт каждые N шагов
    eval_steps: int = 100 # Выполнять оценку каждые N шагов
    logging_steps: int = 20 # Логировать каждые N шагов
    max_checkpoints: int = 3 # Максимальное количество чекпоинтов
    log_level: str = "INFO" # Уровень логирования
    verbose: bool = False # Подробный вывод
    log_gpu_steps: int = 100 # Логировать GPU статистику каждые N шагов
    log_gradients_steps: int = 200 # Логировать градиенты каждые N шагов
    
# ===================================================================
# ПАРАМЕТРЫ МОНИТОРИНГА И КОНТРОЛЯ
# ===================================================================   
   
    early_stopping_patience: int = 3 # Терпение для ранней остановки
    overfitting_threshold: float = 0.1 # Порог для детектора переобучения
    cache_clear_steps: int = 50 # Очищать кеш CUDA каждые N шагов

# ===================================================================
# SEQUENCE PACKING
# ===================================================================
      
    sequence_packing: bool = False  # Включить упаковку нескольких файлов
    packing_max_tokens: int = 512   # Макс. токенов в одной последовательности
    packing_min_items: int = 2       # Мин. документов в упаковке
    packing_separator: str = "[EOS]" # Разделитель между документами
    auto_clean_text: bool = True     # Автоматическая очистка текста
    packing_continue_learning: bool = True  # Продолжать обучение на нескольких файлах
    
# ===================================================================
# ПРОЧИЕ ПАРАМЕТРЫ
# =================================================================== 
     
    use_progress_bar: bool = True # Показывать прогресс-бар
    num_workers: int = 0 # Количество workers для DataLoader
    frozen_layers: List[str] = field(default_factory=list) # Замороженные слои модели
    eval_prompt: str = "Thalia, раскажи, что нового ты узнала и чему ты научилась, за время обучения?" # Промпт для оценки генерации
    
# ===================================================================
# ПАРАМЕТРЫ ВАЛИДАЦИИ
# ===================================================================  
    
    validation_mode: str = "each_epoch" # "each_epoch", "final_only", "disabled"
    validation_interval_epochs: int = 1 # Интервал между валидациями (если mode=each_epoch)
    save_validation_generations: bool = True # Сохранять ли сгенерированные тексты при валидации
    
    def __post_init__(self):
        """Валидация и приведение типов параметров"""
        # Приведение целочисленных параметров
        self.epochs = int(self.epochs)
        self.warmup_steps = int(self.warmup_steps)
        self.max_length = int(self.max_length)
        self.generation_max_length = int(self.generation_max_length)
        self.top_k = int(self.top_k)
        self.min_token_length = int(self.min_token_length)
        self.cache_clear_steps = int(self.cache_clear_steps)
        self.log_gpu_steps = int(self.log_gpu_steps)
        self.log_gradients_steps = int(self.log_gradients_steps)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        
        # Приведение вещественных параметров
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.gradient_clip_val = float(self.gradient_clip_val)
        self.temperature = float(self.temperature)
        self.top_p = float(self.top_p)
        self.repetition_penalty = float(self.repetition_penalty)
        self.validation_split_ratio = float(self.validation_split_ratio)
        self.overfitting_threshold = float(self.overfitting_threshold)
        self.overlap_ratio = float(self.overlap_ratio)
        
        # Приведение булевых параметров
        self.split_long_texts = bool(self.split_long_texts)
        self.strict_roles = bool(self.strict_roles)
        self.auto_clean_dataset = bool(self.auto_clean_dataset)
        self.use_mixed_precision = bool(self.use_mixed_precision)
        self.use_bfloat16 = bool(self.use_bfloat16)
        self.use_progress_bar = bool(self.use_progress_bar)
        
        # Обработка строковых параметров
        self.optimizer_type = str(self.optimizer_type).lower()
        
        # Обработка batch_size
        if isinstance(self.batch_size, str) and self.batch_size.lower() != "auto":
            try:
                self.batch_size = int(self.batch_size)
            except ValueError:
                print(f"Warning: Invalid batch_size value '{self.batch_size}'. Defaulting to 'auto'.")
                self.batch_size = "auto"
        
        self.output_dir = str(self.output_dir)
        
        # Валидация параметров генерации
        self.temperature = max(0.1, min(float(self.temperature), 2.0))
        self.top_k = max(0, min(int(self.top_k), 1000))
        self.top_p = max(0.0, min(float(self.top_p), 1.0))
        self.repetition_penalty = max(1.0, min(float(self.repetition_penalty), 2.0))
        self.generation_max_length = max(10, min(int(self.generation_max_length), 2048))

        # Приведение новых параметров
        self.packing_max_tokens = int(self.packing_max_tokens)
        self.packing_min_items = int(self.packing_min_items)
        self.sequence_packing = bool(self.sequence_packing)
        self.auto_clean_text = bool(self.auto_clean_text)
        
        # ФИКС: Приведение параметров валидации + валидация validation_mode
        self.validation_mode = str(self.validation_mode).lower()
        self.validation_interval_epochs = int(self.validation_interval_epochs)
        self.save_validation_generations = bool(self.save_validation_generations)
        
        # Валидация значений
        valid_modes = ["each_epoch", "final_only", "disabled"]
        if self.validation_mode not in valid_modes:
            self.validation_mode = "each_epoch"
            print(f"Warning: Неверный validation_mode. Установлен 'each_epoch'. Допустимые значения: {valid_modes}") # Замена logger на print
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}. Using defaults.") # Без logger
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {} # Прямой вызов, импорт глобальный
                valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
                config_data = {k: v for k, v in config_data.items() if k in valid_fields}
            return cls(**config_data)
        except ImportError:
            print(f"PyYAML not installed. Install with 'pip install pyyaml'. Using defaults.")
            return cls()
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}. Using defaults. Error: {e}")
            return cls()
    
    def save_yaml(self, config_path: str) -> None:
        if not config_path:
            config_path = "config.yaml"
        try:
            config_dir = os.path.dirname(config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            config_dict = asdict(self)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Error saving config: {e}")

# ===================================================================
# КЛАСС LOGGER
# =================================================================== 

class Logger:
    """Класс для логирования и вывода информации в консоль с цветами"""
    def __init__(self, log_dir: str, log_level: str = "INFO"):
        """
        Инициализация логгера
        Args:
            log_dir: директория для сохранения логов
            log_level: уровень логирования (INFO, WARNING, ERROR, etc.)
        """
        self.console = Console()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Настройка базового логирования
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "training.log", encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        # Настройка консольного вывода
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        def filter_loss_type(record):
            return "loss_type" not in record.getMessage().lower()
        console_handler.addFilter(filter_loss_type)
        self.logger.addHandler(console_handler)
        # ДОБАВЬ ЭТО: Отдельный handler для ошибок
        error_handler = logging.FileHandler(self.log_dir / "errors.txt", mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s\n---'))
        self.logger.addHandler(error_handler)

# ===================================================================
# МЕТОДЫ ЛОГИРОВАНИЯ УРОВНЕЙ
# ===================================================================   
  
    def debug(self, message: str, style: str = "blue") -> None:
        """Отладочное сообщение"""
        self.logger.debug(message)
        if self.logger.getEffectiveLevel() <= logging.DEBUG: # Показываем только если уровень DEBUG
            self.console.print(f"[DEBUG] {message}", style=style)
            
    def info(self, message: str, style: str = "green") -> None:
        """Информационное сообщение"""
        self.logger.info(message)        
        self.console.print(f"[INFO] {message}", style=style)
        
    def success(self, message: str) -> None:
        """Сообщение об успешном выполнении"""
        self.console.print(f"[SUCCESS] {message}", style="green")
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Предупреждение"""
        self.console.print(f"[WARNING] {message}", style="yellow")
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Ошибка"""
        self.console.print(f"[ERROR] {message}", style="red")
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Критическая ошибка с защитой от разметки"""
        # Экранируем текст, чтобы [tensor] не ломал цвета
        safe_message = escape(str(message))
        
        self.console.print(f"[bold red][CRITICAL][/bold red] {safe_message}")
        
        # В файл пишем оригинальное сообщение (там разметка не страшна)
        if hasattr(self, 'logger'):
            self.logger.critical(message)
        
# ===================================================================
# МЕТОДЫ ВИЗУАЛИЗАЦИИ И ФОРМАТИРОВАНИЯ
# ===================================================================   
 
    def panel(self, content: str, title: str, style: str = "green") -> None:
        """Вывод информации в панели"""
        self.console.print(Panel(content, title=title, style=style))
        
    def table(self, data: Dict[str, Any], title: str = "") -> None:
        """
        Вывод данных в виде таблицы
        Args: data: данные для отображения
            title: заголовок таблицы
        """
        table = Table(title=title, style="blue")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="magenta")
        for key, value in data.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            table.add_row(key, str(value))
        self.console.print(table)
        
    def progress_bar(self, iterable, desc: str = "Processing", total: int = None):
        return tqdm(iterable, desc=desc, total=total, disable=not self.config.use_progress_bar)
        
# ===================================================================
# СЛУЖЕБНЫЕ МЕТОДЫ
# ===================================================================

    def set_level(self, log_level: str) -> None:
        """
        Установка уровня логирования
        Args:
            log_level: новый уровень логирования
        """
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
    def get_log_file(self) -> Path:
        """
        Получение пути к файлу лога
        Returns:
            Path объект с путем к файлу
        """
        return self.log_dir / "training.log"
        
    def clear_logs(self) -> None:
        """
        Очистка файла логов
        """
        log_file = self.get_log_file()
        if log_file.exists():
            log_file.unlink()
        self.info("Логи очищены")

# ===================================================================
# МЕТОДЫ ДЛЯ ОТЛАДКИ
# ===================================================================

    def debug_memory(self) -> None:
        """
        Логирование информации о памяти (для отладки)
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
    def traceback(self, error: Exception) -> None:
        """
        Логирование traceback ошибки
        Args:
            error: исключение для логирования
        """
        self.error(f"Exception: {error}")
        self.error(traceback.format_exc())

# ===================================================================
# КЛАСС DATASETVALIDATOR
# ===================================================================

class DatasetValidator:
    """
    УМНЫЙ валидатор, который распознает форматы данных (JSONL, TXT, CSV) и сохраняет их структуру.
    """
    SUPPORTED_FORMATS = [
        {
            "name": "Dialog",
            "required_keys": ["system", "user", "assistant"],
            "target_key": "assistant"
        },
        {
            "name": "Instruction",
            "required_keys": ["instruction", "output"],
            "optional_keys": ["input"],
            "target_key": "output"
        },       
        {
            "name": "Q&A",
            "required_keys": ["question", "answer"],
            "optional_keys": ["context"],
            "target_key": "answer"
        },
        {
            "name": "Completion",
            "required_keys": ["prompt", "completion"],
            "target_key": "completion"
        },
        {
            "name": "GEC",
            "required_keys": ["incorrect", "correct"],
            "optional_keys": ["input_text", "target_text"],
            "target_key": "correct"
        },
        # 🔥 НОВЫЙ ФОРМАТ: Prediction 
        {
            "name": "Prediction",
            "required_keys": ["model", "question_id", "ground_truth", "prediction", "is_correct"],
            "optional_keys": ["parse_ok", "format_fallback_used", "error"],
            "target_key": "prediction",  # Что генерировать
            "prompt_keys": ["model", "question_id", "ground_truth"]  # Что использовать как промпт
        },
        {
            "name": "Plain Text",
            "required_keys": ["text"],
            "target_key": "text"
        },
    ]
    
    def __init__(self, config: TrainingConfig, tokenizer, logger: Logger = None, trainer_formatter=None):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger or Logger("logs", "WARNING")
        # trainer_formatter — ожидается callable(item) -> str (передаётся из ModelTrainer)
        self.trainer_formatter = trainer_formatter
        # Если передан callable, используем его; иначе оставляем None и применяем локальную реализацию
        self._external_formatter = trainer_formatter if callable(trainer_formatter) else None
        
    def _get_formatted_text(self, item: Dict) -> str:
        """
        Универсальная функция форматирования БЕЗ ключей и БЕЗ сепораторов.
        """
        # Пробуем внешний форматтер
        if callable(self._external_formatter):
            try:
                res = self._external_formatter(item)
                if isinstance(res, str):
                    return res
            except Exception as e:
                self.logger.warning(f"Внешний форматтер упал: {e}. Применяю локальный форматтер.")
        
        # Локальная реализация
        try:
            # 🔥 НОВЫЙ ФОРМАТ: Prediction
            if all(key in item for key in ["model", "question_id", "ground_truth", "prediction"]):
                # Формируем текст для обучения: контекст + вопрос + правильный ответ + предсказание
                context = f"Model: {item.get('model', '')}"
                question = f"Question: {item.get('question_id', '')}"
                ground_truth = f"Ground truth: {item.get('ground_truth', '')}"
                prediction = f"Prediction: {item.get('prediction', '')}"
                
                # Добавляем дополнительные поля если есть
                extras = []
                if 'is_correct' in item:
                    extras.append(f"Correct: {item['is_correct']}")
                if 'error' in item and item['error']:
                    extras.append(f"Error: {item['error']}")
                
                all_parts = [context, question, ground_truth, prediction] + extras
                return " ".join(all_parts)
            
            # Диалог
            elif "system" in item and "user" in item and "assistant" in item:
                return (
                    f"{item.get('system','')} "
                    f"{item.get('user','')} "
                    f"{item.get('assistant','')}"
                )
            
            # Instruction
            elif "instruction" in item and "output" in item:
                if "input" in item and item["input"]:
                    return (
                        f"{item['instruction']} "
                        f"{item['input']} "
                        f"{item['output']}"
                    )
                else:
                    return (
                        f"{item['instruction']} "
                        f"{item['output']}"
                    )
            
            # Q&A
            elif "question" in item and "answer" in item:
                if "context" in item and item["context"]:
                    return (
                        f"{item['context']} "
                        f"{item['question']} "
                        f"{item['answer']}"
                    )
                else:
                    return (
                        f"{item['question']} "
                        f"{item['answer']}"
                    )
            
            # Completion
            elif "prompt" in item and "completion" in item:
                return f"{item['prompt']}{item['completion']}"
            
            # GEC
            elif "incorrect" in item and "correct" in item:
                return f"{item['incorrect']} {item['correct']}"
            
            # Plain Text
            elif "text" in item:
                return item["text"]
            
            else:
                # В крайнем случае сконкатенируем все строковые поля через пробел
                parts = []
                for k in ("system", "user", "assistant", "instruction", "input", "output", 
                         "prompt", "completion", "text", "model", "question_id", 
                         "ground_truth", "prediction"):
                    if k in item and isinstance(item[k], str):
                        parts.append(item[k])
                    elif k in item and isinstance(item[k], (int, float)):
                        parts.append(str(item[k]))
                
                if parts:
                    return " ".join(parts)
        
        except Exception as e:
            self.logger.warning(f"Ошибка при локальном форматировании item: {e}")
        
        return ""
        
    def _format_item_for_training(self, item: Dict) -> str:
        return self._get_formatted_text(item)         
        
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
        # ✅ Логирование размера датасета
        file_size_gb = os.path.getsize(dataset_path) / (1024**3)
        self.logger.info(f"📚 Датасет: {dataset_path} (размер: {file_size_gb:.2f} GB)")
        stats = {
            'total_samples': 0, 'valid_samples': 0, 'skipped_samples': 0, 'avg_length': 0.0,
            'issues': [], 'processed_items': [], 'format_counts': defaultdict(int),
            'chunked_samples': 0, 'duplicates_skipped': 0
        }
        # Detect format
        ext = os.path.splitext(dataset_path)[1].lower()
        if self.config.dataset_format == "auto":
            if ext == '.jsonl':
                format_type = 'json'
            elif ext == '.txt':
                format_type = 'text'
            elif ext == '.csv':
                format_type = 'csv'
            else:
                format_type = 'json' # Default
                self.logger.warning(f"Неизвестное расширение {ext}, используем 'json'")
        else:
            format_type = self.config.dataset_format
        # ✅ Логирование detect'а
        self.logger.info(f"🔍 Формат датасета: {format_type} (расширение: {ext})")
        token_lengths = []
        seen_hashes = set()
        try:
            if format_type == 'json':
                # ✅ JSONL обработка
                total_lines = sum(1 for _ in open(dataset_path, 'r', encoding='utf-8'))
                stats['total_samples'] = total_lines # Устанавливаем заранее, без +=1 в цикле
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    with tqdm(total=total_lines, desc="Валидация JSONL", disable=not self.config.use_progress_bar) as pbar:
                        for line_num, line in enumerate(f, 1):
                            try:
                                item = json.loads(line.strip())
                                text_hash = hashlib.md5(str(item).encode()).hexdigest()
                                if text_hash in seen_hashes:
                                    stats['duplicates_skipped'] += 1
                                    stats['issues'].append(f"Строка {line_num}: Дубликат")
                                    pbar.update(1)
                                    continue
                                seen_hashes.add(text_hash)
                                processed_items, detected_format, was_chunked = self._validate_and_process_item(item)
                                if processed_items and detected_format:
                                    # 🔥 ИСПРАВЛЕНИЕ: valid_samples увеличиваем на количество чанков
                                    stats['valid_samples'] += len(processed_items)
                                    stats['processed_items'].extend(processed_items)
                                    stats['format_counts'][detected_format['name']] += len(processed_items)
                                    if was_chunked:
                                        stats['chunked_samples'] += 1
                                    # вњ… Batched token count
                                    if len(processed_items) > 0:
                                        texts = [self._get_formatted_text(pi) for pi in processed_items]
                                        encoded = self.tokenizer(texts, truncation=False, return_tensors=None)
                                        for t in encoded['input_ids']:
                                            token_lengths.append(len(t))
                                else:
                                    stats['skipped_samples'] += 1
                                    stats['issues'].append(f"Строка {line_num}: Нераспознанный формат: {str(item)[:50]}...")
                            except json.JSONDecodeError as e:
                                stats['skipped_samples'] += 1
                                stats['issues'].append(f"РЎС‚СЂРѕРєР° {line_num}: JSON РѕС€РёР±РєР°: {e}")
                            except Exception as e:
                                stats['skipped_samples'] += 1
                                stats['issues'].append(f"РЎС‚СЂРѕРєР° {line_num}: РћР±С‰Р°СЏ РѕС€РёР±РєР°: {e}")
                            pbar.update(1)
            elif format_type == 'text':
                # ✅ TXT обработка
                file_size_gb = os.path.getsize(dataset_path) / (1024**3)
                if file_size_gb > 1.0:
                    self.logger.info("📦 Большой TXT: Streaming + sample 10k")
                    if load_dataset is None:
                        self.logger.warning("datasets not installed, fallback to full load")
                        # Fallback to full load
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f if line.strip()]
                        total_samples = len(lines)
                    else:
                        ds = load_dataset('text', data_files=dataset_path, streaming=True)
                        sample_ds = list(ds.take(10000)) # List for iterate
                        lines = [ex['text'].strip() for ex in sample_ds if ex['text'].strip()]
                        total_samples = len(lines) # Approx
                else:
                    # Full load for small
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    total_samples = len(lines)
                if total_samples == 0: 
                    self.logger.warning("TXT файл пустой!")
                    return stats
                stats['total_samples'] = total_samples
                with tqdm(total=total_samples, desc="Валидация TXT", disable=not self.config.use_progress_bar) as pbar:
                    for line_num, line in enumerate(lines, 1):
                        item = {"text": line}
                        text_hash = hashlib.md5(line.encode()).hexdigest()
                        if text_hash in seen_hashes:
                            stats['duplicates_skipped'] += 1
                            pbar.update(1)
                            continue
                        seen_hashes.add(text_hash)
                        processed_items, detected_format, was_chunked = self._validate_and_process_item(item)
                        if processed_items:
                            # 🔥 ИСПРАВЛЕНИЕ: valid_samples увеличиваем на количество чанков
                            stats['valid_samples'] += len(processed_items)
                            stats['processed_items'].extend(processed_items)
                            stats['format_counts']['Plain Text'] += len(processed_items)
                            if was_chunked:
                                stats['chunked_samples'] += 1
                            # Batch token lengths
                            texts = [self._get_formatted_text(pi) for pi in processed_items]
                            encoded = self.tokenizer(texts, truncation=False, return_tensors=None)
                            for t in encoded['input_ids']:
                                token_lengths.append(len(t))
                        else:
                            stats['skipped_samples'] += 1
                            stats['issues'].append(f"Строка {line_num}: Слишком короткая: '{line[:50]}...'")
                        pbar.update(1)
            elif format_type == 'csv':
                # ✅ УЛУЧШЕННАЯ CSV обработка
                if load_dataset is None:
                    self.logger.error("datasets not installed for CSV support. Install: pip install datasets")
                    return stats
                
                self.logger.info(f"📊 Загрузка CSV: {dataset_path}")
                ds = load_dataset('csv', data_files=dataset_path)
                
                # Показываем первые несколько строк для отладки
                self.logger.info(f"📋 Пример CSV строки: {ds['train'][0]}")
                self.logger.info(f"📋 Колонки: {ds['train'].column_names}")
                
                total_samples = len(ds['train'])
                if total_samples == 0:
                    self.logger.warning("CSV файл пустой!")
                    return stats
                
                stats['total_samples'] = total_samples
                
                with tqdm(total=total_samples, desc="Валидация CSV", disable=not self.config.use_progress_bar) as pbar:
                    for i, example in enumerate(ds['train']):
                        # 🔥 ИСПРАВЛЕНО: Используем улучшенный _map_csv_to_item
                        item = self._map_csv_to_item(example)
                        
                        # Хеш для проверки дубликатов
                        text_hash = hashlib.md5(str(item).encode()).hexdigest()
                        if text_hash in seen_hashes:
                            stats['duplicates_skipped'] += 1
                            pbar.update(1)
                            continue
                        seen_hashes.add(text_hash)
                        
                        # Валидация и обработка
                        processed_items, detected_format, was_chunked = self._validate_and_process_item(item)
                        
                        if processed_items:
                            # 🔥 ИСПРАВЛЕНИЕ: valid_samples увеличиваем на количество чанков
                            stats['valid_samples'] += len(processed_items)
                            stats['processed_items'].extend(processed_items)
                            
                            # Определяем формат
                            if detected_format:
                                stats['format_counts'][detected_format['name']] += len(processed_items)
                            else:
                                # Если формат не определен, но есть 'text' поле
                                if 'text' in item:
                                    stats['format_counts']['Plain Text'] += len(processed_items)
                                else:
                                    stats['format_counts']['Other'] += len(processed_items)
                            
                            if was_chunked:
                                stats['chunked_samples'] += 1
                            
                            # Подсчет токенов
                            texts = [self._get_formatted_text(pi) for pi in processed_items]
                            encoded = self.tokenizer(texts, truncation=False, return_tensors=None)
                            for t in encoded['input_ids']:
                                token_lengths.append(len(t))
                        else:
                            stats['skipped_samples'] += 1
                            stats['issues'].append(f"Строка {i}: Невалидный формат: {str(example)[:100]}...")
                        
                        pbar.update(1)
                        
                        if (i + 1) % 1000 == 0:
                            self.logger.info(f"📊 Обработано {i+1}/{total_samples} строк CSV")
            # Calc avg_length
            if token_lengths:
                stats['avg_length'] = np.mean(token_lengths)
                self.logger.info(f"📏 Средняя длина: {stats['avg_length']:.2f} токенов")
            if stats['duplicates_skipped'] > 0:
                self.logger.warning(f"⚠️ Пропущено дубликатов: {stats['duplicates_skipped']}")
            if stats['chunked_samples'] > 0:
                self.logger.info(f"🔪 Разбито чанков: {stats['chunked_samples']}")
            self._print_dataset_stats(stats)
            return stats
        except Exception as e:
            self.logger.error(f"Критическая ошибка валидации: {e}")
            traceback.print_exc()
            return stats
            
    def _map_csv_to_item(self, example):
        """
        Правильное преобразование CSV строки в словарь для обучения.
        """
        # Преобразуем в обычный dict если это Dataset объект
        if hasattr(example, 'items'):
            example_dict = dict(example)
        else:
            example_dict = example
        
        self.logger.debug(f"CSV строка: {example_dict}")
        
        # 🔥 СЛУЧАЙ 1: Уже есть все нужные поля (как в вашем CSV)
        required_fields = ['model', 'question_id', 'ground_truth', 'prediction', 'is_correct']
        if all(field in example_dict for field in required_fields):
            return {
                "model": str(example_dict.get('model', '')),
                "question_id": str(example_dict.get('question_id', '')),
                "ground_truth": str(example_dict.get('ground_truth', '')),
                "prediction": str(example_dict.get('prediction', '')),
                "is_correct": int(example_dict.get('is_correct', 0)),
                "parse_ok": int(example_dict.get('parse_ok', 0)) if 'parse_ok' in example_dict else 0,
                "format_fallback_used": int(example_dict.get('format_fallback_used', 0)) if 'format_fallback_used' in example_dict else 0,
                "error": str(example_dict.get('error', '')) if example_dict.get('error') is not None else ""
            }
        
        # Случай 2: Есть колонки 'prompt' и 'completion'
        if 'prompt' in example_dict and 'completion' in example_dict:
            return {
                "prompt": str(example_dict['prompt']),
                "completion": str(example_dict['completion'])
            }
        
        # Случай 3: Есть колонки 'question' и 'answer'
        if 'question' in example_dict and 'answer' in example_dict:
            result = {
                "question": str(example_dict['question']),
                "answer": str(example_dict['answer'])
            }
            if 'context' in example_dict:
                result['context'] = str(example_dict['context'])
            return result
        
        # Случай 4: Есть колонки 'instruction' и 'output'
        if 'instruction' in example_dict and 'output' in example_dict:
            result = {
                "instruction": str(example_dict['instruction']),
                "output": str(example_dict['output'])
            }
            if 'input' in example_dict:
                result['input'] = str(example_dict['input'])
            return result
        
        # Случай 5: Есть колонка 'text'
        if 'text' in example_dict:
            return {"text": str(example_dict['text'])}
        
        # Случай 6: Все остальные колонки - создаем text из всех полей
        text_parts = []
        for key, value in example_dict.items():
            if value is not None and str(value).strip():
                text_parts.append(f"{key}: {value}")
        
        if text_parts:
            return {"text": " | ".join(text_parts)}
        
        # Если ничего не подошло
        return example_dict

            
    def _split_long_text(self, text: str, max_tokens: int, overlap_ratio: float = 0.1) -> List[Dict]:
        """Разбивает длинный текст на чанки с перекрытием"""
        if not self.tokenizer or not text or not text.strip():
            return [{"text": text}]
        
        # Токенизируем
        tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [{"text": text}]
        
        # 🔥 ИСПРАВЛЕНО: Проверяем валидность параметров
        overlap_ratio = max(0.0, min(overlap_ratio, 0.5))  # Ограничиваем 0-0.5
        overlap_tokens = max(1, int(max_tokens * overlap_ratio))
        step_size = max(1, max_tokens - overlap_tokens)
        
        chunks = []
        total_tokens_after_split = 0
        
        for i in range(0, len(tokens), step_size):
            end_idx = min(i + max_tokens, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            
            if len(chunk_tokens) == 0:
                continue
            
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            is_last = (end_idx >= len(tokens))
            
            # Корректировка границ для промежуточных чанков
            if not is_last and self.config.adjust_chunk_boundaries:
                chunk_text = self._adjust_chunk_boundaries(chunk_text)
            
            # Создаем метаданные для чанка
            chunk_data = {
                "text": chunk_text,
                "tokens": chunk_tokens,
                "metadata": {
                    "chunk": f"{len(chunks) + 1}",
                    "continuation": not is_last,
                    "original_length": len(tokens),
                    "is_last": is_last
                }
            }
            
            # Добавляем маркеры в текст для визуального разделения
            if self.config.show_chunk_markers:
                if not is_last:
                    chunk_data["text_with_marker"] = chunk_text + self.config.chunk_continuation_marker
                else:
                    chunk_data["text_with_marker"] = chunk_text + self.config.chunk_end_marker
            
            chunks.append(chunk_data)
            
            total_tokens_after_split += len(chunk_tokens)
            
            if end_idx >= len(tokens):
                break
        
        # Проверка потери токенов
        if abs(total_tokens_after_split - len(tokens)) > len(tokens) * 0.1:
            self.logger.warning(f"Потеря токенов при разбиении: {len(tokens)} → {total_tokens_after_split}")
        
        return chunks
        
    def _split_user_text_with_overlap(self, original_item: Dict, system_text: str,
                                       user_text: str, assistant_text: str,
                                       max_tokens: int, overlap_ratio: float) -> List[Dict]:
        """
        🔥 ИСПРАВЛЕННОЕ разбиение длинного user_text на чанки.
        
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Промежуточные чанки НЕ используются для обучения,
        только последний чанк с полным ответом. Это предотвращает обучение модели
        преждевременной остановке (генерации EOS после неполного ввода).
        """
        if not self.tokenizer:
            return [original_item]
        
        # Формируем полную строку для одного диалога
        system_tokens = self.tokenizer.encode(system_text or "", add_special_tokens=False)
        user_tokens = self.tokenizer.encode(user_text or "", add_special_tokens=False)
        assistant_tokens = self.tokenizer.encode(assistant_text or "", add_special_tokens=False)
        
        # Проверяем, нужно ли разбиение
        reserved = len(system_tokens) + len(assistant_tokens)
        user_chunk_capacity = max_tokens - len(system_tokens) - len(assistant_tokens)
        
        if len(user_tokens) <= user_chunk_capacity or user_chunk_capacity <= 0:
            return [original_item]
        
        overlap = max(1, int(max_tokens * overlap_ratio))
        step = max(1, user_chunk_capacity - overlap)
        
        chunks = []
        total_user_len = len(user_tokens)
        
        # Разбиваем user_tokens на окна с перекрытием
        window_starts = list(range(0, total_user_len, step))
        
        for i, start in enumerate(window_starts):
            end = min(start + user_chunk_capacity, total_user_len)
            user_chunk_tokens = user_tokens[start:end]
            
            if len(user_chunk_tokens) == 0:
                continue
            
            user_chunk_text = self.tokenizer.decode(user_chunk_tokens, skip_special_tokens=True)
            is_last = (end >= total_user_len)
            
            # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: 
            # Промежуточные чанки НЕ ДОБАВЛЯЕМ в обучение
            # Только последний чанк с полным ответом
            if not is_last:
                # Пропускаем промежуточный чанк - он не для обучения
                # Модель не должна учиться на "User: часть текста → Assistant: [EOS]"
                self.logger.debug(f"Пропуск промежуточного чанка {i+1}/{len(window_starts)}")
                continue
            
            # Только последний чанк включаем в обучение
            chunk_item = original_item.copy()
            
            # System только в первом чанке (для последнего чанка system уже есть)
            chunk_item['user'] = user_chunk_text
            chunk_item['assistant'] = assistant_text
            
            # Метаданные для отслеживания
            chunk_item['metadata'] = {
                "chunk": f"{i+1}/{len(window_starts)}",
                "is_last": True,
                "chunk_start": start,
                "chunk_end": end,
                "total_chunks": len(window_starts),
                "skipped_intermediate": len(window_starts) - 1
            }
            
            chunks.append(chunk_item)
            self.logger.info(f"📦 Создан обучающий чанк (последний из {len(window_starts)}): "
                            f"user_len={len(user_chunk_tokens)}/{len(user_tokens)} токенов")
        
        # Если по какой-то причине не создано ни одного чанка (is_last никогда не был True)
        if not chunks:
            # Fallback: берём последний чанк принудительно
            last_start = window_starts[-1] if window_starts else 0
            last_end = min(last_start + user_chunk_capacity, total_user_len)
            last_chunk_tokens = user_tokens[last_start:last_end]
            last_chunk_text = self.tokenizer.decode(last_chunk_tokens, skip_special_tokens=True)
            
            chunk_item = original_item.copy()
            chunk_item['user'] = last_chunk_text
            chunk_item['assistant'] = assistant_text
            chunk_item['metadata'] = {"is_last": True, "forced_fallback": True}
            chunks.append(chunk_item)
            self.logger.warning(f"⚠️ Принудительное создание последнего чанка (fallback)")
        
        return chunks
        
    def _validate_and_process_item(self, item: Dict) -> Tuple[Optional[List[Dict]], Optional[Dict], bool]:
        """
        Валидация и обработка элемента данных.
        """
        if not isinstance(item, dict):
            return None, None, False

        detected_format = None
        
        # ШАГ 1: Определяем формат
        for fmt in self.SUPPORTED_FORMATS:
            required = fmt.get("required_keys", [])
            if all(key in item for key in required):
                detected_format = fmt
                self.logger.debug(f"✅ Распознан формат: {fmt['name']} для элемента с ключами {list(item.keys())}")
                break

        # Если формат не распознан
        if not detected_format:
            self.logger.debug(f"❌ Формат не распознан для элемента с ключами: {list(item.keys())}")
            return None, None, False

        # ШАГ 2: Проверяем, что есть текст для обучения
        full_text = self._get_formatted_text(item)
        if not full_text or not full_text.strip():
            self.logger.debug(f"Пустой текст после форматирования для формата {detected_format['name']}")
            return None, None, False

        # ШАГ 3: Проверяем длину и необходимость разбиения
        if self.tokenizer:
            try:
                all_tokens = self.tokenizer.encode(full_text, truncation=False, add_special_tokens=False)
                
                # Если всё помещается - возвращаем как есть
                if len(all_tokens) <= self.config.max_length:
                    return [item], detected_format, False
                
                # Нужна разбивка
                self.logger.debug(f"Текст слишком длинный ({len(all_tokens)} > {self.config.max_length}), разбиваем...")
                
                # 🔥 ДЛЯ ДИАЛОГОВОГО ФОРМАТА используем специальное разбиение
                if detected_format['name'] == 'Dialog':
                    system_text = item.get('system', '')
                    user_text = item.get('user', '')
                    assistant_text = item.get('assistant', '')
                    
                    # Используем исправленный метод _split_user_text_with_overlap
                    chunks = self._split_user_text_with_overlap(
                        item, system_text, user_text, assistant_text,
                        self.config.max_length, self.config.overlap_ratio
                    )
                    return chunks, detected_format, True
                
                # Для других форматов - общее разбиение
                # Ищем самое длинное поле для разбиения
                longest_field = None
                max_len = 0
                
                # Для Prediction формата разбиваем prediction
                if detected_format['name'] == 'Prediction' and 'prediction' in item:
                    longest_field = 'prediction'
                else:
                    # Ищем среди всех строковых полей
                    for key in item:
                        if isinstance(item[key], str):
                            try:
                                tokens = self.tokenizer.encode(item[key], truncation=False, add_special_tokens=False)
                                if len(tokens) > max_len:
                                    max_len = len(tokens)
                                    longest_field = key
                            except:
                                continue
                
                if longest_field and longest_field in item:
                    long_text = item[longest_field]
                    chunks = self._split_long_text(long_text, self.config.max_length, self.config.overlap_ratio)
                    
                    if len(chunks) > 1:
                        result_chunks = []
                        for i, chunk_data in enumerate(chunks):
                            new_item = item.copy()
                            new_item[longest_field] = chunk_data["text"]
                            if "metadata" in chunk_data:
                                new_item.setdefault("metadata", {}).update(chunk_data["metadata"])
                            result_chunks.append(new_item)
                        
                        return result_chunks, detected_format, True
                
                return [item], detected_format, False
                
            except Exception as e:
                self.logger.debug(f"Ошибка при токенизации: {e}")
                return [item], detected_format, False
        
        return [item], detected_format, False
        
    def _debug_token_count(self, item: Dict, fmt: Dict):
        """Отладочная информация о токенах"""
        if not self.config.verbose:
            return
        token_info = {}
        for key in fmt["required_keys"]:
            if key in item and isinstance(item[key], str):
                tokens = self.tokenizer.encode(item[key], truncation=False, add_special_tokens=False)
                token_info[key] = len(tokens)
        full_text = self._format_item_for_training(item)
        all_tokens = self.tokenizer.encode(full_text, truncation=False, add_special_tokens=False)
        token_info["total"] = len(all_tokens)
        self.logger.info(f"Токены по полям: {token_info}")
        
    def _adjust_chunk_boundaries(self, text: str, max_adjustment_chars: int = 100) -> str:

        import re
        if not text or len(text) < 10: # слишком короткий текст не корректируем
            return text
        # Ищем все границы предложений в последних max_adjustment_chars символах…
        search_region = text[-max_adjustment_chars:] if len(text) > max_adjustment_chars else text
        sentence_ends = []
        # Ищем . !  которые likely являются концами предложений
        for match in re.finditer(r'(?<=[.!?])\s+', search_region):
            # Позиция относительно всего текста
            global_pos = len(text) - len(search_region) + match.start()
            sentence_ends.append(global_pos)
        # Также ищем переносы строк как потенциальные границы
        for match in re.finditer(r'\n+', search_region):
            global_pos = len(text) - len(search_region) + match.start()
            sentence_ends.append(global_pos)
        if sentence_ends:
            # Берём самую позднюю границу (ближе к концу текста)
            best_boundary = max(sentence_ends)
            # Убедимся что это хорошая граница (не слишком близко к началу)
            if best_boundary > len(text) * 0.7: # хотя бы 70% текста сохраняем
                return text[:best_boundary].strip()
        # Если не нашли хорошую границу, пробуем найти любую точку near the end
        dot_pos = text.rfind('.')
        question_pos = text.rfind('?')
        exclamation_pos = text.rfind('!')
        newline_pos = text.rfind('\n')
        # Выбираем самую позднюю подходящую границу
        potential_boundaries = [pos for pos in [dot_pos, question_pos, exclamation_pos, newline_pos] if pos != -1]
        if potential_boundaries:
            best_boundary = max(potential_boundaries)
            if best_boundary > len(text) * 0.6: # хотя бы 60% текста сохраняем
                return text[:best_boundary].strip()
        return text # если ничего не нашли, оставляем как есть
        
    def _print_dataset_stats(self, stats: Dict):
        """Выводит статистику датасета в виде таблицы."""
        table = Table(title="Статистика валидации датасета", style="blue")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="magenta")
        table.add_row("Всего строк", str(stats['total_samples']))
        table.add_row("Валидных записей (после чанкинга)", str(stats['valid_samples']))
        table.add_row("Пропущено/ошибок", str(stats['skipped_samples']))
        table.add_row("Разбито на чанков", str(stats['chunked_samples']))
        table.add_row("Средняя длина (токенов)", f"{stats['avg_length']:.2f}")
        self.logger.console.print(table)
        if stats['format_counts']:
            format_table = Table(title="Распределение форматов", style="green")
            format_table.add_column("Формат", style="cyan")
            format_table.add_column("Количество", style="magenta")
            for name, count in stats['format_counts'].items():
                format_table.add_row(name, str(count))
            self.logger.console.print(format_table)
        if stats['processed_items']:
            samples_text = ""
            for i, item in enumerate(stats['processed_items'][:3]):
                samples_text += f"Пример {i+1}:\n{json.dumps(item, ensure_ascii=False, indent=2)}\n"
            self.logger.console.print(Panel(
                samples_text.strip(),
                title="Примеры валидных данных (без изменений)",
                border_style="green"
            ))
        if stats['issues']:
            issues_text = "\n".join(stats['issues'][:5]) + ("\n..." if len(stats['issues']) > 5 else "")
            self.logger.console.print(Panel(
                issues_text,
                title="Проблемы в датасете",
                border_style="red"
            ))
            
# ===================================================================
# КЛАСС MODELTRAINER
# ===================================================================

class ModelTrainer:
   
    def __init__(self, config: TrainingConfig, logger: Logger, console=None):
        self.config = config
        self.logger = logger
        self.console = console
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True 
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        self.last_memory = None 
        self.training_stats = []
        self.best_loss = float('inf')
        self.current_epoch_loss = 0.0
        self.current_epoch_steps = 0
        self.global_step = 0
        self.gradient_accumulation_steps_counter = 0
        self.dataset_validated = False
        self.train_items = [] 
        self.val_items = [] 
        self.dataset_validation_stats = None
        self.total_layers: Optional[int] = None 
        os.makedirs(config.output_dir, exist_ok=True)
        if self.config.use_mixed_precision and self.device.type != 'cuda':
            self.config.use_mixed_precision = False
            self.logger.warning("Mixed precision is only available with CUDA. Disabling.")
        if self.config.use_bfloat16 and self.device.type != 'cuda':
            self.config.use_bfloat16 = False
            self.logger.warning("BFloat16 is only available with CUDA. Disabling.")
 
# ===================================================================
# ВЛОЖЕННЫЙ КЛАСС TEXTDATASET
# ===================================================================    
    
    class TextDataset(Dataset):
        def __init__(self, items: List[Dict], tokenizer, config: TrainingConfig, logger: Logger, formatter):
            self.items = items
            self.tokenizer = tokenizer
            self.config = config
            self.logger = logger
            self.formatter = formatter
        def __len__(self) -> int:
            return len(self.items)
            
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.items):
            idx = len(self.items) - 1

        try:
            item = self.items[idx]
            
            # 🔥 ИСПРАВЛЕНИЕ: Поддержка sequence packing с правильными тензорами
            if self.config.sequence_packing and 'input_ids' in item:
                # Данные уже токенизированы и упакованы
                if isinstance(item['input_ids'], torch.Tensor):
                    input_ids = item['input_ids'].clone().detach()
                else:
                    input_ids = torch.as_tensor(item['input_ids'], dtype=torch.long)
                
                # Получаем labels или создаем копию input_ids
                if 'labels' in item:
                    if isinstance(item['labels'], torch.Tensor):
                        labels = item['labels'].clone().detach()
                    else:
                        labels = torch.as_tensor(item['labels'], dtype=torch.long)
                else:
                    labels = input_ids.clone()
                
                # Проверка на пустые последовательности
                if input_ids.shape[0] == 0:
                    raise ValueError("Empty input_ids after sequence packing")
                
                # attention mask (1 для всех реальных токенов)
                attention_mask = torch.ones_like(input_ids)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
            
            # 1. Получаем промпт и ответ от форматтера
            prompt_text, answer_text = self.formatter(item)
            
            # 2. Получаем EOS токен с проверкой
            eos_token = self.tokenizer.eos_token
            if not eos_token:
                eos_token = "</s>"
                self.logger.warning("eos_token не найден, используем '</s>'")
            
            # 3. Токенизируем промпт (без добавления special tokens)
            prompt_encoded = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_length // 2  # Оставляем место для ответа
            )
            prompt_ids = prompt_encoded['input_ids']
            
            # 4. Токенизируем ответ с EOS
            answer_with_eos = answer_text + eos_token
            answer_encoded = self.tokenizer(
                answer_with_eos,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_length - len(prompt_ids)
            )
            answer_ids = answer_encoded['input_ids']
            
            # 5. Объединяем и проверяем длину
            total_len = len(prompt_ids) + len(answer_ids)
            
            # 🔥 ИСПРАВЛЕНО: Умная обрезка с сохранением контекста
            if total_len > self.config.max_length:
                # Вариант 1: Обрезаем ответ, сохраняя весь промпт
                max_answer_len = self.config.max_length - len(prompt_ids)
                if max_answer_len > 50:  # Оставляем хотя бы 50 токенов для ответа
                    answer_ids = answer_ids[:max_answer_len]
                    prompt_ids = prompt_ids  # сохраняем весь промпт
                else:
                    # Вариант 2: Обрезаем промпт, но сохраняем его конец (где важная информация)
                    # и оставляем достаточно места для ответа
                    min_answer_len = min(100, self.config.max_length // 3)
                    max_prompt_len = self.config.max_length - min_answer_len
                    
                    if len(prompt_ids) > max_prompt_len:
                        # Сохраняем конец промпта (последние max_prompt_len токенов)
                        prompt_ids = prompt_ids[-max_prompt_len:]
                    
                    # Обрезаем ответ если всё ещё не влезает
                    remaining = self.config.max_length - len(prompt_ids)
                    if remaining > 0:
                        answer_ids = answer_ids[:remaining]
            
            # 6. Собираем финальные тензоры
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids  # Промпт не учим, только ответ
            
            # Преобразуем в тензоры
            input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long)
            labels_tensor = torch.as_tensor(labels, dtype=torch.long)
            
            # Проверка на пустые последовательности
            if input_ids_tensor.shape[0] == 0:
                raise ValueError("Empty input_ids after processing")
            
            # attention mask (1 для всех реальных токенов)
            attention_mask = torch.ones_like(input_ids_tensor)
            
            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask,
                'labels': labels_tensor
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing item {idx}: {e}")
            # Возвращаем безопасный dummy
            eos_token = self.tokenizer.eos_token or "</s>"
            dummy_text = f"Пустой {eos_token}"
            encoded = self.tokenizer(dummy_text, return_tensors='pt', 
                                    max_length=10, truncation=True)
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': torch.ones_like(encoded['input_ids'].squeeze(0)),
                'labels': encoded['input_ids'].squeeze(0)
            }
 
# ===================================================================
# МЕТОДЫ ЗАГРУЗКИ И ИНИЦИАЛИЗАЦИИ
# ===================================================================

    def load_model(self, model_path: str = None) -> bool:
        model_path = model_path or self.config.model_path
        if not model_path:
            self.logger.error("Model path is not specified.")
            return False
        
        self.model = None
        self.tokenizer = None
        
        try:
            self.logger.info(f"Пытаемся загрузить токенизатор из: {model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.logger.success(f"Токенизатор успешно загружен")
            except Exception as tokenizer_error:
                self.logger.warning(f"Ошибка загрузки локального токенизатора, пробуем запасной gpt2...")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # 🔥 ИСПРАВЛЕНО: Проверяем и устанавливаем специальные токены
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("pad_token установлен как eos_token")
            
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "</s>"
                self.logger.warning("eos_token не найден, установлен '</s>'")
            
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = "<s>"
                self.logger.info("bos_token установлен как '<s>'")
            
            self.logger.info(f"Токены: BOS={self.tokenizer.bos_token}, "
                            f"EOS={self.tokenizer.eos_token}, "
                            f"PAD={self.tokenizer.pad_token}")
            
            self.logger.info("Пытаемся загрузить модель...")
            try:
                config = AutoConfig.from_pretrained(model_path)
                model_type = getattr(config, 'model_type', '').lower()
                self.logger.info(f"Определен тип модели: {model_type}")
                
                if model_type == 'thalia':
                    self.logger.info("🔍 Обнаружена модель Thalia, загружаем через кастомный класс...")
                    # Важно: Thalia должна быть импортирована в скрипте!
                    self.model = Thalia.from_pretrained(model_path).to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
                    
            except Exception as model_error:
                self.logger.error(f"Ошибка типа модели, пробуем AutoModel: {model_error}")
                self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

            # --- ИСПРАВЛЕННЫЙ БЛОК RESIZE EMBEDDINGS ---
            try:
                tokenizer_vocab_size = len(self.tokenizer)
                model_vocab_size = self.model.config.vocab_size
                
                if tokenizer_vocab_size > model_vocab_size:
                    self.logger.info(f"Расширяем эмбеддинги: {model_vocab_size} -> {tokenizer_vocab_size}")
                    self.model.resize_token_embeddings(tokenizer_vocab_size)
                elif tokenizer_vocab_size < model_vocab_size:
                    # Если разница небольшая (наш Padding 50264 vs 50257), не режем!
                    diff = model_vocab_size - tokenizer_vocab_size
                    if diff <= 64:
                        self.logger.success(f"Сохраняем оптимизированный Padding модели: {model_vocab_size}")
                    else:
                        self.logger.info(f"Обрезка лишних эмбеддингов: {model_vocab_size} -> {tokenizer_vocab_size}")
                        self.model.resize_token_embeddings(tokenizer_vocab_size)
            except Exception as resize_error:
                self.logger.warning(f"Ошибка при проверке размера эмбеддингов: {resize_error}")

            self.logger.success(f"✅ Модель полностью готова!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка загрузки: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
       
    def _apply_frozen_layers_from_config(self):
        """Применяет настройки заморозки из конфига с поддержкой except: префикса"""
        if not self.config.frozen_layers:
            return
        # Проверяем есть ли инвертированные настройки (с префиксом except:)
        except_patterns = []
        normal_patterns = []
        for pattern in self.config.frozen_layers:
            if isinstance(pattern, str) and pattern.startswith('except:'):
                except_patterns.append(pattern.replace('except:', '').strip())
            else:
                normal_patterns.append(pattern)
        # Применяем соответствующий режим
        if except_patterns:
            self.logger.info(f"Применяем инвертированную заморозку из конфига: все кроме {except_patterns}")
            success = self.freeze_layers(freeze_all_except=except_patterns)
            if not success:
                self.logger.warning("Не удалось применить настройки заморозки из конфига")
        elif normal_patterns:
            self.logger.info(f"Применяем обычную заморозку из конфига: {normal_patterns}")
            success = self.freeze_layers(layer_patterns=normal_patterns)
            if not success:
                self.logger.warning("Не удалось применить настройки заморозки из конфига")
                
    def freeze_layers(self, layer_patterns: List[str] = None, 
                      freeze_all_except: List[str] = None, 
                      exact_matches: List[str] = None) -> bool:
        """
        Умная заморозка слоев с поддержкой Thalia компонентов
        """
        if not self.model:
            self.logger.error("Модель не загружена для заморозки слоев.")
            return False
        
        all_params = list(self.model.named_parameters())
        available_layers = {name for name, _ in all_params}
        
        # 🔥 АВТООПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ
        is_thalia = any('personality_core' in name for name in available_layers)
        
        if is_thalia:
            self.logger.info("🔍 Обнаружена архитектура Thalia")
            
            # 🔥 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ О КОМПОНЕНТАХ
            thalia_components = {
                "🧠 Transformer": sum(1 for n in available_layers if 'transformer.' in n),
                "🎭 Personality Core": sum(1 for n in available_layers if 'personality_core.' in n),
                "💫 Living Layer": sum(1 for n in available_layers if 'living_layer.' in n),
                "🧠 Mamba": sum(1 for n in available_layers if 'adaptive_memory.' in n),
                "🔄 Experience Exchange": sum(1 for n in available_layers if 'experience_exchange.' in n),
                "📦 Другие": sum(1 for n in available_layers if not any(
                    x in n for x in ['transformer.', 'personality_core.', 
                                    'living_layer.', 'adaptive_memory.', 
                                    'experience_exchange.'])
                )
            }
            
            self.logger.info("📊 Компоненты Thalia:")
            for comp, count in thalia_components.items():
                if count > 0:
                    self.logger.info(f"   {comp}: {count} слоев")
        
        # Режим: Заморозить все, кроме указанных
        if freeze_all_except is not None:
            self.logger.info(f"Режим 'заморозить все кроме': {freeze_all_except}")
            
            # 🔥 УЛУЧШЕННАЯ ПРОВЕРКА ПАТТЕРНОВ
            valid_exceptions = []
            invalid_exceptions = []
            
            for pattern in freeze_all_except:
                found_layers = [layer_name for layer_name in available_layers if pattern in layer_name]
                if found_layers:
                    valid_exceptions.append(pattern)
                    # Логируем найденные слои
                    if len(found_layers) <= 5:  # Не слишком много
                        for layer in found_layers[:3]:
                            self.logger.debug(f"     • {layer}")
                        if len(found_layers) > 3:
                            self.logger.debug(f"     • ... и еще {len(found_layers) - 3} слоев")
                else:
                    invalid_exceptions.append(pattern)
            
            # Добавляем exact_matches если они есть
            if exact_matches:
                for exact_name in exact_matches:
                    if exact_name in available_layers:
                        valid_exceptions.append(exact_name)
                    else:
                        invalid_exceptions.append(exact_name)
            
            if invalid_exceptions:
                self.logger.warning(f"Шаблоны исключений не найдены: {invalid_exceptions}")
                
                # 🔥 ПОДСКАЗКИ ДЛЯ THALIA
                if is_thalia:
                    self.logger.info("💡 Доступные компоненты Thalia:")
                    thalia_patterns = [
                        ("personality_core", "🎭 Ядро личности"),
                        ("adaptive_memory", "🧠 Mamba память"),
                        ("living_layer", "💫 Living Layer"),
                        ("experience_exchange", "🔄 Обмен опытом")
                    ]
                    for pattern, desc in thalia_patterns:
                        count = sum(1 for n in available_layers if pattern in n)
                        if count > 0:
                            self.logger.info(f"   • '{pattern}' - {desc} ({count} слоев)")
            
            if not valid_exceptions:
                self.logger.error("Ни один из шаблонов исключений не найден в модели!")
                return False
            
            # Замораживаем все, кроме валидных исключений
            frozen_count = 0
            trained_count = 0
            
            # 🔥 ГРУППИРОВКА ПО КОМПОНЕНТАМ ДЛЯ ЛОГА
            component_stats = {}
            
            for name, param in all_params:
                # Проверяем частичные совпадения
                matches_pattern = any(pattern in name for pattern in valid_exceptions)
                # Проверяем точные совпадения
                matches_exact = name in exact_matches if exact_matches else False
                
                if matches_pattern or matches_exact:
                    param.requires_grad = True  # Оставляем обучаемым
                    trained_count += 1
                    
                    # Статистика по компонентам
                    comp_name = self._get_component_name(name)
                    component_stats[comp_name] = component_stats.get(comp_name, 0) + 1
                else:
                    param.requires_grad = False  # Замораживаем
                    frozen_count += 1
            
            # 🔥 ДЕТАЛЬНЫЙ ЛОГ
            self.logger.success(
                f"✅ Заморозка применена: "
                f"Заморожено {frozen_count} параметров, "
                f"Обучается {trained_count} параметров"
            )
            
            if component_stats:
                self.logger.info("📊 Обучаемые компоненты:")
                for comp, count in sorted(component_stats.items(), key=lambda x: -x[1]):
                    percentage = count / trained_count * 100
                    self.logger.info(f"   • {comp}: {count} слоев ({percentage:.1f}%)")
            
            return True
        # Режим: Заморозить только указанные
        if layer_patterns:
            self.logger.info(f"Режим 'заморозить по шаблонам': {layer_patterns}")
            # Находим валидные шаблоны
            valid_patterns = []
            invalid_patterns = []
            for pattern in layer_patterns:
                found = any(pattern in layer_name for layer_name in available_layers)
                if found:
                    valid_patterns.append(pattern)
                else:
                    invalid_patterns.append(pattern)
            if invalid_patterns:
                self.logger.warning(f"Шаблоны не найдены: {invalid_patterns}")
            if not valid_patterns:
                self.logger.error("Ни один шаблон не соответствует слоям модели!")
                # Показываем доступные слои для помощи
                self.logger.info("Доступные слои для примера:")
                sample_layers = list(available_layers)
                for i, layer_name in enumerate(sample_layers[:5]):
                    self.logger.info(f" • {layer_name}")
                if len(sample_layers) > 5:
                    self.logger.info(f" • ... и ещё {len(sample_layers) - 5} слоёв")
                return False
            # Замораживаем только соответствующие шаблонам
            frozen_count = 0
            trained_count = 0
            for name, param in all_params:
                if any(pattern in name for pattern in valid_patterns):
                    param.requires_grad = False # Замораживаем
                    frozen_count += 1
                else:
                    param.requires_grad = True # Оставляем обучаемым
                    trained_count += 1
            self.logger.success(
                f"Режим 'заморозить по шаблонам': "
                f"Заморожено {frozen_count} параметров, "
                f"Обучается {trained_count} параметров. "
                f"Шаблоны: {valid_patterns}"
            )
            return True
        self.logger.error("Не указаны шаблоны для заморозки")
        return False
        
    def _get_component_name(self, layer_name: str) -> str:
        """Определяет имя компонента Thalia по имени слоя"""
        if 'transformer.' in layer_name:
            if 'wte' in layer_name or 'wpe' in layer_name:
                return "🧠 Embeddings"
            elif 'h.' in layer_name:
                try:
                    layer_num = int(layer_name.split('.h.')[1].split('.')[0])
                    total = self._detect_total_layers()
                    if layer_num < total // 3:
                        return "🧠 Transformer (нижние)"
                    elif layer_num < 2 * total // 3:
                        return "🧠 Transformer (средние)"
                    else:
                        return "🧠 Transformer (верхние)"
                except:
                    return "🧠 Transformer"
            elif 'ln_f' in layer_name:
                return "🧠 Output norm"
        elif 'personality_core.' in layer_name:
            return "🎭 Personality Core"
        elif 'living_layer.' in layer_name:
            return "💫 Living Layer"
        elif 'adaptive_memory.' in layer_name:
            return "🧠 Mamba Memory"
        elif 'experience_exchange.' in layer_name:
            return "🔄 Experience Exchange"
        elif 'memory_system.' in layer_name:
            return "📦 Memory System"
        elif 'lm_head' in layer_name:
            return "🧠 LM Head"
        else:
            return "📦 Другие"        
        
# ===================================================================
# МЕТОД ДЛЯ АДАПТИВНЫХ ПРЕСЕТОВ
# ===================================================================

    def get_adaptive_presets(self) -> List[Dict]:
        """Возвращает пресеты заморозки с поддержкой Thalia"""
        # ⚠️ ОПРЕДЕЛЯЕМ АРХИТЕКТУРУ
        is_thalia = hasattr(self.model, 'personality_core') and hasattr(self.model, 'adaptive_memory')
        
        if is_thalia:
            return self._get_thalia_presets()
        else:
            return self._get_legacy_presets()

    def _get_thalia_presets(self) -> List[Dict]:
        """Пресеты для архитектуры Thalia"""
        # Базовые группы для Thalia
        total_transformer_layers = self._detect_total_layers()
        transformer_middle = total_transformer_layers // 2
        
        # 🔥 КЛЮЧЕВЫЕ КОМПОНЕНТЫ THALIA
        components = {
            "transformer_bottom": [f"transformer.h.{i}" for i in range(transformer_middle)],
            "transformer_top": [f"transformer.h.{i}" for i in range(transformer_middle, total_transformer_layers)],
            "embeddings": ["transformer.wte", "transformer.wpe"],
            "output": ["transformer.ln_f", "lm_head"],
            
            # 🔥 НОВЫЕ КОМПОНЕНТЫ
            "personality_core": ["personality_core"],
            "living_layer": ["living_layer"],
            "adaptive_memory": ["adaptive_memory"],
            "experience_exchange": ["experience_exchange"],
            "memory_system": ["memory_system"]  # Если есть старая память
        }
        
        presets = [
            # 🔥 1. ТОЛЬКО ПАМЯТЬ MAMBA (ultra-safe)
            {
                "name": "🧠 Только Mamba память",
                "desc": "Заморожено всё кроме Mamba Heads - минимальный риск",
                "mode": "freeze_all_except",
                "patterns": components["adaptive_memory"],
                "best_for": "Настройка памяти без влияния на личность",
                "risk_level": "🟢 Очень низкий",
                "estimated_trainable": "3-5%",
                "thalia_specific": True
            },
            
            # 🔥 2. ПАМЯТЬ + ОБМЕН ОПЫТОМ (рекомендуемый)
            {
                "name": "🚀 Память + обмен (РЕКОМЕНДУЕМЫЙ)",
                "desc": "Обучаем Mamba и Experience Exchange для быстрой адаптации",
                "mode": "freeze_all_except",
                "patterns": components["adaptive_memory"] + components["experience_exchange"],
                "best_for": "Быстрая адаптация к новым данным",
                "risk_level": "🟡 Средний",
                "estimated_trainable": "8-12%",
                "thalia_specific": True
            },
            
            # 🔥 3. ЛИЧНОСТЬ + ПАМЯТЬ
            {
                "name": "🎭 Личность и память",
                "desc": "Обучаем Personality Core и Mamba для развития характера",
                "mode": "freeze_all_except",
                "patterns": components["personality_core"] + components["adaptive_memory"],
                "best_for": "Развитие характера с сохранением памяти",
                "risk_level": "🟡 Средний",
                "estimated_trainable": "10-15%",
                "thalia_specific": True
            },
            
            # 🔥 4. TOP ТРАНСФОРМЕР + ПАМЯТЬ (аналог старого)
            {
                "name": "🤖 Интеллект + память",
                "desc": "Верхние слои трансформера + Mamba для интеллектуального развития",
                "mode": "freeze_all_except",
                "patterns": components["transformer_top"] + components["adaptive_memory"],
                "best_for": "Развитие интеллекта с адаптивной памятью",
                "risk_level": "🟠 Высокий",
                "estimated_trainable": "15-20%",
                "thalia_specific": False
            },
            
            # 🔥 5. LIVING LAYER АКТИВАЦИЯ
            {
                "name": "💫 Living Layer активация",
                "desc": "Активируем Living Layer для нелинейной адаптации",
                "mode": "freeze_all_except",
                "patterns": components["living_layer"] + components["adaptive_memory"],
                "best_for": "Нелинейная адаптация поведения",
                "risk_level": "🟡 Средний",
                "estimated_trainable": "7-10%",
                "thalia_specific": True
            },
            
            # 🔥 6. ПОЛНАЯ АКТИВАЦИЯ Thalia
            {
                "name": "⚡ Полная Thalia (ОПАСНО)",
                "desc": "Все компоненты Thalia активны - максимальная адаптация",
                "mode": "freeze_all_except",
                "patterns": (components["transformer_top"] + 
                            components["personality_core"] + 
                            components["living_layer"] + 
                            components["adaptive_memory"] + 
                            components["experience_exchange"]),
                "best_for": "Полное переобучение Thalia",
                "risk_level": "🔴 Очень высокий",
                "estimated_trainable": "30-40%",
                "thalia_specific": True,
                "warning": "Может вызвать катастрофическое забывание!"
            },
            
            # 🔥 7. ПОСТЕПЕННАЯ АКТИВАЦИЯ (progressive)
            {
                "name": "📅 Прогрессивная активация",
                "desc": "Постепенная активация компонентов Thalia",
                "mode": "freeze_all_except",
                "patterns": components["adaptive_memory"],  # Начинаем с памяти
                "is_progressive": True,
                "stages": [
                    {
                        "name": "Этап 1: Память",
                        "patterns": components["adaptive_memory"],
                        "unlock_at": "loss < 2.0"
                    },
                    {
                        "name": "Этап 2: Обмен опытом",
                        "patterns": components["adaptive_memory"] + components["experience_exchange"],
                        "unlock_at": "epoch > 2"
                    },
                    {
                        "name": "Этап 3: Личность",
                        "patterns": (components["adaptive_memory"] + 
                                    components["experience_exchange"] + 
                                    components["personality_core"]),
                        "unlock_at": "epoch > 5"
                    },
                    {
                        "name": "Этап 4: Living Layer",
                        "patterns": (components["adaptive_memory"] + 
                                    components["experience_exchange"] + 
                                    components["personality_core"] + 
                                    components["living_layer"]),
                        "unlock_at": "epoch > 10"
                    }
                ],
                "best_for": "Медленная безопасная адаптация",
                "risk_level": "🟢 Низкий",
                "estimated_trainable": "3% → 25%",
                "thalia_specific": True
            }
        ]
        
        return presets

    def _get_legacy_presets(self) -> List[Dict]:
        """Пресеты для старой архитектуры (оригинальный код)"""
        if self.total_layers is None:
            self.total_layers = self._detect_total_layers()
        total_layers = self.total_layers
        
        # Базовые группы
        groups = {
            "embeddings": ["transformer.wte", "transformer.wpe"],
            "bottom_layers": [f"transformer.h.{i}" for i in range(6)],
            "top_layers": [f"transformer.h.{i}" for i in range(6, 12)],
            "output": ["transformer.ln_f"]
        }
        
        presets = [
            # ... существующие пресеты ...
            {
                "name": "🟩🪛 Safe Embed + Final",
                "desc": "Только embed и final norm — минимальный риск.",
                "mode": "freeze_all_except",
                "patterns": groups["embeddings"] + groups["output"],
                "best_for": "Стиль/токены, без forgetting.",
                "risk_level": "🟢 Низкий",
                "estimated_trainable": "5%"
            },

        ]
        
        return presets
        
    def get_layer_statistics(self) -> Optional[Dict[str, Any]]:
        """Возвращает статистику слоёв модели"""
        if not self.model:
            return None
        layers_info = []
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            layers_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'params': param_count,
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            })
        return {
            'layers': layers_info,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
    def _auto_detect_output_layers(self, last_n_layers: int = 2) -> List[str]:
        """Автоматическое определение последних N слоёв модели с поддержкой Thalia"""
        if not self.model:
            return []
        # Для Thalia моделей
        model_type = str(type(self.model)).lower()
        if "thalia" in model_type:
            # Thalia имеет другую структуру - возвращаем общие паттерны
            return ["transformer.h", "lm_head"]
        # Собираем все слои с номерами
        layered_params = []
        for name, param in self.model.named_parameters():
            # Ищем паттерны с цифрами (номера слоёв)
            if re.search(r'\.[0-9]+\.', name):
                layered_params.append((name, param))
        if not layered_params:
            self.logger.warning("Не найдены слои с номерами для автоматического определения")
            return []
        # Извлекаем номера слоёв
        layer_numbers = set()
        for name, _ in layered_params:
            match = re.search(r'\.([0-9]+)\.', name)
            if match:
                layer_numbers.add(int(match.group(1)))
        if not layer_numbers:
            return []
        # Находим максимальный номер слоя
        max_layer = max(layer_numbers)
        self.logger.info(f"Обнаружено слоёв: {max_layer + 1} (номера 0-{max_layer})")
        # Выбираем последние N слоёв
        target_layers = list(range(max(0, max_layer - last_n_layers + 1), max_layer + 1))
        # Собираем БАЗОВЫЕ имена слоёв (например: 'transformer.h.21')
        output_patterns = []
        for layer_num in target_layers:
            # Находим первый попавшийся параметр этого слоя
            for name, _ in layered_params:
                if f'.{layer_num}.' in name:
                    # Извлекаем БАЗОВОЕ имя слоя (например: 'transformer.h.21')
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == str(layer_num):
                            # Берем всё до номера слоя включительно
                            base_layer_name = '.'.join(parts[:i+1])
                            if base_layer_name not in output_patterns:
                                output_patterns.append(base_layer_name)
                            break
                    break
        # Добавляем выходной слой (lm_head) - тоже базовое имя
        output_head_patterns = ['lm_head', 'output', 'classifier', 'head']
        for name, _ in self.model.named_parameters():
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in output_head_patterns):
                # Берем базовое имя выходного слоя
                if '.' in name:
                    base_output_name = name.split('.')[0] # Например: 'lm_head'
                    if base_output_name not in output_patterns:
                        output_patterns.append(base_output_name)
                else:
                    if name not in output_patterns:
                        output_patterns.append(name)
                break
        self.logger.info(f"Автоопределены выходные слои: {output_patterns}")
        return output_patterns
        
    def _detect_total_layers(self) -> int:
        """Определение общего количества слоёв в модели с улучшенной логикой"""
        if not self.model:
            return 0
        layer_numbers = set()
        for name, _ in self.model.named_parameters():
            # 🔏 УЛУЧШЕННЫЕ ПАТТЕРНЫ ДЛЯ РАЗНЫХ АРХИТЕКТУР
            patterns = [
                r'\.h\.([0-9]+)\.', # GPT-2, GPT-Neo: transformer.h.0.
                r'\.layer\.([0-9]+)\.', # BERT, RoBERTa: encoder.layer.0.
                r'\.blocks\.([0-9]+)\.', # GPT-J, CodeGen: transformer.blocks.0.
                r'\.layers\.([0-9]+)\.', # LLaMA, Falcon: model.layers.0.
                r'\_([0-9]+)\.', # Резервный паттерн: layer_0.
            ]
            for pattern in patterns:
                matches = re.finditer(pattern, name)
                for match in matches:
                    try:
                        layer_num = int(match.group(1))
                        layer_numbers.add(layer_num)
                    except (ValueError, IndexError):
                        continue
        # 🔏 ДЕТЕКТИРОВАНИЕ THALIA МОДЕЛЕЙ
        model_type = str(type(self.model)).lower()
        if "thalia" in model_type:
            self.logger.info("Обнаружена модель Thalia - используем специфическую детекцию")
            # Для Thalia ищем максимальный номер в transformer.h.X
            thalialayers = set()
            for name, _ in self.model.named_parameters():
                if 'transformer.h.' in name:
                    match = re.search(r'\.h\.([0-9]+)\.', name)
                    if match:
                        thalialayers.add(int(match.group(1)))
            if thalialayers:
                total = max(thalialayers) + 1
                self.logger.info(f"Thalia: обнаружено {total} слоёв трансформера")
                return total
        # 🔏 ПРОВЕРКА НА МАЛЕНЬКИЕ МОДЕЛИ
        if not layer_numbers:
            self.logger.warning("Не удалось определить номера слоёв. Возможно модель слишком маленькая или нестандартная.")
            # Попытка определить по количеству уникальных блоков
            layer_blocks = set()
            for name, _ in self.model.named_parameters():
                if any(x in name for x in ['attn.', 'mlp.', 'attention.']):
                    # Пытаемся извлечь номер слоя из контекста
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            layer_blocks.add(int(part))
                            break
            if layer_blocks:
                total = max(layer_blocks) + 1
                self.logger.info(f"Определено {total} слоёв через анализ блоков")
                return total
            return 0 # Не удалось определить
        total_layers = max(layer_numbers) + 1
        self.logger.info(f"Автоопределение: {total_layers} слоёв (номера 0-{max(layer_numbers)})")
        # 🔏 ВАЛИДАЦИЯ РЕЗУЛЬТАТА
        expected_params = total_layers * 10 # Ожидаем ~10 параметров на слой
        actual_params = len([p for p in self.model.parameters()])
        if actual_params < expected_params * 0.5:
            self.logger.warning(f"Возможно неверное определение слоёв: ожидалось ~{expected_params} параметров, найдено {actual_params}")
        return total_layers
        
    def print_presets_info(self):
        """Улучшенный вывод информации о пресетах"""
        is_thalia = hasattr(self.model, 'personality_core') if self.model else False
        
        if is_thalia:
            self._print_thalia_presets_info()
        else:
            self._print_legacy_presets_info()

    def _print_thalia_presets_info(self):
        """Вывод пресетов для Thalia"""
        presets = self.get_adaptive_presets()
        stats = self.get_layer_statistics()
        
        print("\n" + "="*80)
        print("🧠 ДОСТУПНЫЕ ПРЕСЕТЫ ДЛЯ THALIA")
        print("="*80)
        
        if stats:
            total_params = stats['total_params']
            trainable_params = stats['trainable_params']
            trainable_ratio = stats['trainable_ratio']
            
            # 🔥 СТАТИСТИКА ПО КОМПОНЕНТАМ
            component_counts = {}
            for layer in stats['layers']:
                comp = self._get_component_name(layer['name'])
                component_counts[comp] = component_counts.get(comp, 0) + layer['params']
            
            print(f"📊 Модель Thalia: {total_params:,} параметров")
            print(f"📈 Обучаемо: {trainable_params:,} ({trainable_ratio:.1%})")
            print(f"📦 Компоненты:")
            for comp, count in sorted(component_counts.items(), key=lambda x: -x[1]):
                percent = count / total_params * 100
                print(f"   • {comp}: {count:,} ({percent:.1f}%)")
        
        print(f"\n💡 Рекомендации:")
        print(f"   • '🚀 Память + обмен' - для быстрой адаптации")
        print(f"   • '🎭 Личность и память' - для развития характера")
        print(f"   • '🧠 Только Mamba память' - для безопасного старта")
        print(f"   • '📅 Прогрессивная активация' - для медленной адаптации")
        
        for i, preset in enumerate(presets, 1):
            print(f"\n{i:2d}. {preset['name']}")
            print(f"   📝 {preset['desc']}")
            print(f"   🎯 {preset['best_for']}")
            print(f"   ❤️‍🔥 Уровень риска: {preset['risk_level']}")
            
            if preset.get('thalia_specific', False):
                print(f"   🤖 Специфично для Thalia")
            
            if 'estimated_trainable' in preset:
                print(f"   📈 Обучаемо: ~{preset['estimated_trainable']}")
            
            if preset.get('warning'):
                print(f"   ⚠️  ВНИМАНИЕ: {preset['warning']}")
        
        print(f"\n📊 Всего пресетов: {len(presets)}")
        print("="*80)
 
    def quick_freeze_thalia(self, mode: str = "memory_only"):
        """
        Быстрая заморозка для Thalia
        modes:
          - "memory_only": только Mamba память
          - "memory_exchange": память + обмен опытом
          - "personality": личность + память
          - "full_adaptation": все кроме трансформера
        """
        if not hasattr(self.model, 'personality_core'):
            self.logger.error("Это не модель Thalia!")
            return False
        
        modes = {
            "memory_only": {
                "patterns": ["adaptive_memory"],
                "desc": "Только Mamba память"
            },
            "memory_exchange": {
                "patterns": ["adaptive_memory", "experience_exchange"],
                "desc": "Память + обмен опытом"
            },
            "personality": {
                "patterns": ["adaptive_memory", "personality_core"],
                "desc": "Личность + память"
            },
            "full_adaptation": {
                "patterns": ["adaptive_memory", "experience_exchange", 
                            "personality_core", "living_layer"],
                "desc": "Все адаптивные компоненты"
            }
        }
        
        if mode not in modes:
            self.logger.error(f"Неизвестный режим: {mode}. Доступно: {list(modes.keys())}")
            return False
        
        self.logger.info(f"🚀 Быстрая заморозка Thalia: {modes[mode]['desc']}")
        return self.freeze_layers(freeze_all_except=modes[mode]["patterns"])
 
# ===================================================================
# МЕТОДЫ ПОДГОТОВКИ ДАННЫХ
# =================================================================== 

    def _format_item_for_training(self, item: Dict) -> Tuple[str, str]:
        """
        Возвращает (prompt_text, answer_text) для любого поддерживаемого формата.
        """
        USER_PREFIX = "User: "
        ASSISTANT_PREFIX = "\n\nAssistant: "

        prompt_text = ""
        answer_text = ""

        try:
            # 1. Диалог (system/user/assistant)
            if "user" in item or "assistant" in item:
                if "system" in item and item["system"] and item["system"].strip():
                    prompt_text += f"{item['system'].strip()}\n\n"
                prompt_text += f"{USER_PREFIX}{item.get('user', '').strip()}"
                answer_text = item.get("assistant", "").strip()

            # 2. Инструкции (instruction + input → output)
            elif "instruction" in item:
                instr = item['instruction'].strip()
                if "input" in item and item["input"] and item["input"].strip():
                    instr += f"\n{item['input'].strip()}"
                prompt_text = f"{USER_PREFIX}{instr}"
                answer_text = item.get("output", "").strip()

            # 3. Q&A (question + context → answer)
            elif "question" in item:
                q = item['question'].strip()
                if "context" in item and item["context"] and item["context"].strip():
                    q = f"{item['context'].strip()}\n\n{q}"
                prompt_text = f"{USER_PREFIX}{q}"
                answer_text = item.get("answer", "").strip()

            # 4. Completion (prompt → completion)
            elif "prompt" in item:
                prompt_text = f"{USER_PREFIX}{item['prompt'].strip()}"
                answer_text = item.get("completion", "").strip()

            # 🔥 ИСПРАВЛЕНО: Plain text / GEC / любой другой
            # 5. Plain text - ВАЖНО: текст должен быть в answer, чтобы учиться его генерировать
            elif "text" in item:
                prompt_text = ""  # Пустой промпт
                answer_text = item["text"].strip()  # Весь текст - то, что нужно сгенерировать

            # 🔥 ИСПРАВЛЕНО: Последний fallback — просто склеиваем всё в answer
            else:
                all_text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                prompt_text = ""  # Пустой промпт
                answer_text = all_text.strip()  # Весь склеенный текст - то, что нужно сгенерировать

            return prompt_text, answer_text

        except Exception as e:
            self.logger.warning(f"Ошибка форматирования: {e}")
            return "", ""
 
# ===================================================================
# SEQUENCE PACKING МЕТОДЫ 
# ===================================================================

    def _clean_text(self, text: str) -> str:
        """
        Базовая очистка текста от мусора, но сохраняем всю пунктуацию.
        Модель должна учиться генерировать кавычки, тире, многоточия.
        """
        if not self.config.auto_clean_text:
            return text
        
        # Только базовую очистку - убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем управляющие символы, но сохраняем всю пунктуацию
        # Оставляем буквы, цифры и всю стандартную пунктуацию
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Удаляем пустые строки
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()

    def _interactive_file_loader(self) -> List[Dict]:
        """Интерактивная загрузка TXT файлов"""
        from rich.panel import Panel
        
        all_items = []
        physical_file_counter = 0  # СЧЕТЧИК ФИЗИЧЕСКИХ ФАЙЛОВ
        
        if self.console:
            self.console.print(Panel(
                "📁 ЗАГРУЗКА ФАЙЛОВ ДЛЯ ОБУЧЕНИЯ\n"
                f"Макс. токенов в последовательности: {self.config.packing_max_tokens}",
                title="Sequence Packing",
                style="cyan"
            ))
        
        while True:
            file_path = input(f"\n📄 Путь к TXT файлу #{physical_file_counter + 1} (или 'start' для начала): ").strip()
            
            if file_path.lower() == 'start':
                if not all_items:
                    self.logger.error("❌ Нет загруженных файлов! Загрузите хотя бы один файл.")
                    continue
                break
            
            if not os.path.exists(file_path):
                self.logger.error(f"❌ Файл не найден: {file_path}")
                continue
            
            physical_file_counter += 1  # УВЕЛИЧИВАЕМ СЧЕТЧИК ТОЛЬКО ДЛЯ НОВОГО ФАЙЛА
            
            # Загружаем файл и получаем его чанки с правильным physical_file_id
            items = self._load_single_txt_file(file_path, physical_file_counter)
            
            if items:
                all_items.extend(items)
                self.logger.info(f"✅ Загружено файлов: {physical_file_counter}")
                
                file_size = os.path.getsize(file_path) / 1024
                estimated_tokens = sum(item['metadata'].get('tokens', 0) for item in items)
                self.logger.info(f"   Размер: {file_size:.1f} KB (~{estimated_tokens} токенов, {len(items)} чанков)")
            
            # Спрашиваем про следующий файл
            if physical_file_counter >= self.config.packing_min_items:
                choice = input("\n➕ Загрузить ещё файл? (y/n/start): ").strip().lower()
                if choice in ['n', 'no', 'start']:
                    break
            else:
                self.logger.info(f"⚠️ Нужно минимум {self.config.packing_min_items} файла. Загрузите ещё.")
        
        self.logger.success(f"✅ Всего загружено: {physical_file_counter} физических файлов, {len(all_items)} чанков")
        return all_items, physical_file_counter  # ВОЗВРАЩАЕМ И ЧАНКИ, И КОЛИЧЕСТВО ФАЙЛОВ

    def _load_single_txt_file(self, file_path: str, file_id: int) -> List[Dict]:
        """Загружает ОДИН TXT файл и разбивает на чанки с сохранением структуры"""
        items = []
        
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_basename = os.path.basename(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 🔥 ИСПОЛЬЗУЕМ НОВЫЙ _smart_chunking!
            chunks = self._smart_chunking(text, self.config.packing_max_tokens)
            
            self.logger.info(f"📄 Файл #{file_id}: {file_basename}, всего токенов: {sum(len(c['tokens']) for c in chunks)}")
            
            for part_num, chunk in enumerate(chunks, 1):
                items.append({
                    'tokens': chunk['tokens'],
                    'source': f"{file_basename}#{part_num}",
                    'physical_file_id': file_id,
                    'physical_file_name': file_basename,
                    'metadata': {
                        'file': file_path,
                        'file_id': file_id,
                        'part': part_num,
                        'total_parts': len(chunks),
                        'tokens': len(chunk['tokens']),
                        'is_last': chunk.get('is_last', False)
                    }
                })
            
            self.logger.info(f"   → Создано {len(chunks)} чанков для файла #{file_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки {file_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return items

    def _smart_chunking(self, text: str, max_tokens: int = 256) -> List[Dict]:
        """Умное разбиение текста с сохранением структуры предложений"""
        import re
        
        # Токенизируем весь текст
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [{
                'text': text,
                'tokens': tokens,
                'length': len(tokens),
                'is_last': True
            }]
        
        # Разбиваем на предложения (грубо)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_tokens = []
        
        for sent in sentences:
            if sent.strip():
                sent_tokens = self.tokenizer.encode(sent, add_special_tokens=False)
                sentence_tokens.append(sent_tokens)
        
        # Группируем предложения в чанки
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent_tokens in sentence_tokens:
            sent_len = len(sent_tokens)
            
            # Если одно предложение длиннее max_tokens - режем его
            if sent_len > max_tokens:
                # Сохраняем текущий накопленный чанк
                if current_chunk:
                    chunk_tokens = [t for sent in current_chunk for t in sent]
                    chunks.append({
                        'tokens': chunk_tokens,
                        'length': len(chunk_tokens),
                        'is_last': False
                    })
                    current_chunk = []
                    current_len = 0
                
                # Режем длинное предложение на части
                for i in range(0, sent_len, max_tokens - 20):  # перекрытие 20 токенов
                    part_tokens = sent_tokens[i:i + max_tokens]
                    if len(part_tokens) >= 20:
                        chunks.append({
                            'tokens': part_tokens,
                            'length': len(part_tokens),
                            'is_last': False
                        })
                continue
            
            # Если предложение влезает в текущий чанк
            if current_len + sent_len <= max_tokens:
                current_chunk.append(sent_tokens)
                current_len += sent_len
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunk_tokens = [t for sent in current_chunk for t in sent]
                    chunks.append({
                        'tokens': chunk_tokens,
                        'length': len(chunk_tokens),
                        'is_last': False
                    })
                
                # Начинаем новый чанк с этого предложения
                current_chunk = [sent_tokens]
                current_len = sent_len
        
        # Последний чанк
        if current_chunk:
            chunk_tokens = [t for sent in current_chunk for t in sent]
            chunks.append({
                'tokens': chunk_tokens,
                'length': len(chunk_tokens),
                'is_last': True  # последний чанк файла
            })
        elif chunks:
            chunks[-1]['is_last'] = True
        
        # Добавляем EOS в последний чанк
        for chunk in chunks:
            if chunk.get('is_last', False):
                chunk['tokens'] = chunk['tokens'] + [self.tokenizer.eos_token_id]
                chunk['length'] += 1
        
        return chunks

    def _pack_sequences(self, items: List[Dict], total_files: int) -> List[Dict]:
        """Упаковывает чанки в тренировочные последовательности"""
        if not self.config.sequence_packing:
            return items
        
        self.logger.info(f"📁 Начинаем упаковку {total_files} физических файлов...")
        
        from collections import defaultdict
        files_dict = defaultdict(list)
        for item in items:
            file_id = item.get('physical_file_id', 0)
            files_dict[file_id].append(item)
        
        all_chunks = []
        chunks_with_eos = 0
        
        for file_id, file_chunks in files_dict.items():
            file_chunks.sort(key=lambda x: x['metadata'].get('part', 0))
            
            for i, chunk in enumerate(file_chunks):
                tokens = chunk['tokens']
                
                if not tokens:
                    continue
                
                is_last_chunk = chunk['metadata'].get('is_last', False)
                
                input_ids = torch.tensor(tokens, dtype=torch.long)
                
                # 🔥 ВАЖНО: создаем attention_mask (1 для реальных токенов)
                attention_mask = torch.ones_like(input_ids)
                
                chunk_item = {
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),
                    'attention_mask': attention_mask,  # теперь передаем!
                    'source': chunk['source'],
                    'physical_file_id': file_id,
                    'metadata': {
                        'file_id': file_id,
                        'chunk_in_file': i + 1,
                        'total_chunks_in_file': len(file_chunks),
                        'is_last': is_last_chunk,
                        'has_eos': is_last_chunk  # EOS только в последнем
                    }
                }
                all_chunks.append(chunk_item)
                
                if is_last_chunk:
                    chunks_with_eos += 1
        
        # Статистика
        if all_chunks:
            lengths = [len(chunk['input_ids']) for chunk in all_chunks]
            self.logger.info(f"📦 Упаковано: {len(all_chunks)} training чанков")
            self.logger.info(f"📁 Физических файлов: {total_files}")
            self.logger.info(f"📊 Длины: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
            self.logger.info(f"🔍 Чанков с EOS: {chunks_with_eos} (должно быть = {total_files})")
        
        return all_chunks

    def verify_sequence_packing(self, items: List[Dict]) -> None:
        """Проверяет правильность упаковки последовательностей"""
        
        # Группируем по file_id
        from collections import defaultdict
        files_dict = defaultdict(list)
        
        for i, item in enumerate(items):
            file_id = item.get('file_id', i)  # используем file_id если есть
            files_dict[file_id].append(i)
        
        total_files = len(files_dict)
        chunks_with_eos = sum(1 for item in items if item['metadata'].get('has_eos', False))
        
        self.logger.info("🔍 Проверка упаковки:")
        self.logger.info(f"   • Всего чанков: {len(items)}")
        self.logger.info(f"   • Всего файлов: {total_files}")
        self.logger.info(f"   • Чанков с EOS: {chunks_with_eos}")
        
        # Проверяем каждый файл
        files_ok = 0
        for file_id, chunk_indices in files_dict.items():
            eos_in_file = sum(1 for idx in chunk_indices if items[idx]['metadata'].get('has_eos', False))
            last_chunk_idx = chunk_indices[-1]
            last_has_eos = items[last_chunk_idx]['metadata'].get('has_eos', False)
            
            if eos_in_file == 1 and last_has_eos:
                files_ok += 1
            else:
                self.logger.warning(f"   ⚠️ Файл {file_id}: {eos_in_file} EOS, последний чанк has_eos={last_has_eos}")
        
        if files_ok == total_files:
            self.logger.success(f"✅ Все {total_files} файлов корректно завершаются EOS")
        else:
            self.logger.warning(f"⚠️ Проблема: только {files_ok}/{total_files} файлов имеют EOS в последнем чанке")   
            
    def _split_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Разбивает текст на чанки примерно равной длины в токенах"""
        # Простой способ: разбить на предложения
        import re
        
        # Разбиваем на предложения (приблизительно)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Оцениваем длину предложения в токенах
            sentence_length = len(sentence) // 4  # грубая оценка
            
            if current_length + sentence_length > max_tokens and current_chunk:
                # Сохраняем текущий чанк
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
 
    def validate_dataset_manually(self, dataset_path: str = None) -> bool:
        """
        Ручная валидация с проверкой целостности данных для fine-tuning
        """
        dataset_path = dataset_path or self.config.dataset_path
        if not dataset_path:
            self.logger.error("Путь к датасету не указан")
            return False
        
        try:
            original_ext = os.path.splitext(dataset_path)[1].lower()
            
            # 🔥 ВАЖНО: Для структурированных данных всегда используем JSONL!
            # Это сохранит metadata и структуру чанков
            val_path = os.path.join(self.config.output_dir, f"validation.jsonl")
            train_path = os.path.join(self.config.output_dir, f"train.jsonl")
            
            self.logger.info(f"📦 Для сохранения будет использован JSONL формат (сохраняет структуру)")
            
            # ✅ Валидация
            validator = DatasetValidator(self.config, self.tokenizer, self.logger, self._format_item_for_training)
            validation_results = validator.validate_dataset(dataset_path)
            
            if not validation_results['processed_items']:
                self.logger.error("Валидных данных после проверки не найдено.")
                return False
            
            all_items = validation_results['processed_items']
            
            # 🔍 ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ
            self._validate_data_integrity(all_items, "перед сохранением")
            
            # 🔥 Разделяем на train/val
            if len(all_items) > 1:
                val_size = max(1, int(len(all_items) * self.config.validation_split_ratio))
                
                # Создаем индексы и перемешиваем
                indices = list(range(len(all_items)))
                random.shuffle(indices)
                
                # Разделяем
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]
                
                self.val_items = [all_items[i] for i in val_indices]
                self.train_items = [all_items[i] for i in train_indices]
                
                # Проверяем разделенные данные
                self._validate_data_integrity(self.val_items, "validation set")
                self._validate_data_integrity(self.train_items, "train set")
                
                # Сохраняем в JSONL
                self._save_items_to_file(self.val_items, val_path, '.jsonl')
                self.config.validation_dataset_path = val_path
                self.logger.info(f"Val: {val_path} ({len(self.val_items)} примеров)")
                
            else:
                self.logger.warning("Датасет мал — val пустой")
                self.train_items = all_items
                self.val_items = []
            
            self._save_items_to_file(self.train_items, train_path, '.jsonl')
            self.config.dataset_path = train_path
            
            # Проверяем сохраненные файлы
            self._verify_saved_files(train_path, val_path, '.jsonl', len(self.train_items), len(self.val_items))
            
            self.dataset_validated = True
            self.logger.success(f"✅ Валидация OK: {len(self.train_items)} train, {len(self.val_items)} val")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            traceback.print_exc()
            return False
            
    def _validate_data_integrity(self, items: List[Dict], context: str):
        if not items:
            self.logger.warning(f"⚠️ Нет данных в {context}")
            return
        valid_count = 0
        for i in range(len(items)):  # Explicit loop, no genexpr
            try:
                current_item = items[i]  # Explicit bind
                formatted = self._format_item_for_training(current_item)
                # ⚠️ ФИКС: Обработка tuple (prompt, answer)
                if isinstance(formatted, tuple):
                    prompt, answer = formatted
                    formatted_text = f"{prompt or ''} {answer or ''}".strip()  # Конкат для проверки (или только answer)
                else:
                    formatted_text = str(formatted or "").strip()
                if not formatted_text:
                    self.logger.warning(f"Пустой текст в элементе {i} {context}: {list(current_item.keys())}")
                    continue  # Skip, но не ошибка
                valid_count += 1
                # Проверка диалога (keys ок для Dict)
                if 'user' in current_item and 'assistant' not in current_item:
                    self.logger.warning(f"Неполный диалог в элементе {i}: есть user, но нет assistant")
            except Exception as e:
                self.logger.error(f"Ошибка проверки элемента {i} в {context}: {e}")
                continue  # Skip bad item
        self.logger.info(f"📋 Проверка {context}: {valid_count}/{len(items)} валидных элементов")
        
    def _verify_saved_files(self, train_path: str, val_path: str, file_ext: str, expected_train: int, expected_val: int):
        """Проверяет что файлы сохранены корректно"""
        try:
            # Проверяем train файл
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as f:
                    if file_ext == '.jsonl':
                        train_lines = sum(1 for _ in f)
                    elif file_ext == '.txt':
                        train_lines = sum(1 for line in f if line.strip())
                    elif file_ext == '.csv':
                        # 🔥 ИСПРАВЛЕНО: Для CSV просто считаем строки и вычитаем заголовок
                        lines = f.readlines()
                        train_lines = len(lines) - 1  # вычитаем заголовок
                        self.logger.info(f"📊 CSV train: {len(lines)} всего строк, {train_lines} записей")
                    else:
                        train_lines = 0
                
                if train_lines != expected_train:
                    self.logger.warning(f"⚠️ Несоответствие train: ожидалось {expected_train}, сохранено {train_lines}")
                    # Дополнительная диагностика
                    if file_ext == '.csv':
                        self.logger.info("🔍 Возможные причины:")
                        self.logger.info("  • Дублирование при записи CSV")
                        self.logger.info("  • Неправильный подсчет строк")
                        self.logger.info("  • Проблемы с разделителями")
                else:
                    self.logger.info(f"✅ Train файл корректен: {train_lines} записей")
            
            # Проверяем val файл если есть
            if expected_val > 0 and os.path.exists(val_path):
                with open(val_path, 'r', encoding='utf-8') as f:
                    if file_ext == '.jsonl':
                        val_lines = sum(1 for _ in f)
                    elif file_ext == '.txt':
                        val_lines = sum(1 for line in f if line.strip())
                    elif file_ext == '.csv':
                        # 🔥 ИСПРАВЛЕНО: Для CSV просто считаем строки и вычитаем заголовок
                        lines = f.readlines()
                        val_lines = len(lines) - 1  # вычитаем заголовок
                        self.logger.info(f"📊 CSV val: {len(lines)} всего строк, {val_lines} записей")
                    else:
                        val_lines = 0
                
                if val_lines != expected_val:
                    self.logger.warning(f"⚠️ Несоответствие val: ожидалось {expected_val}, сохранено {val_lines}")
                else:
                    self.logger.info(f"✅ Val файл корректен: {val_lines} записей")
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки сохраненных файлов: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def _save_items_to_file(self, items: List[Dict], file_path: str, file_ext: str):
        """
        Сохраняет элементы в файл в соответствующем формате
        с сохранением структуры для корректного дообучения
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_ext == '.jsonl':
                    # JSONL формат - идеально для структурированных данных с метаданными!
                    for item in items:
                        # Сохраняем как есть, включая metadata
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    self.logger.info(f"💾 Сохранено {len(items)} элементов в JSONL формате")
                    return
                
                elif file_ext == '.txt':
                    # TXT формат - только текст, без метаданных
                    for item in items:
                        # Берем text поле или формируем из всего
                        if 'text' in item:
                            text = item['text']
                        else:
                            # Если нет text, берем первое строковое поле
                            text = next((str(v) for v in item.values() if isinstance(v, str)), "")
                        
                        if text.strip():
                            f.write(text.strip() + '\n')
                    self.logger.info(f"💾 Сохранено {len(items)} элементов в TXT формате")
                    return
                
                elif file_ext == '.csv':
                    # 🔥 ИСПРАВЛЕНО: CSV формат - но для структурированных данных лучше JSONL!
                    self.logger.warning("⚠️ CSV не подходит для данных с метаданными. Рекомендуется JSONL!")
                    
                    # Пробуем сохранить, но с flatten структурой
                    import csv
                    import pandas as pd
                    
                    # Преобразуем items в плоскую структуру для CSV
                    flat_items = []
                    for item in items:
                        flat_item = {}
                        
                        # Обрабатываем простые поля
                        for k, v in item.items():
                            if k == 'metadata' and isinstance(v, dict):
                                # Разворачиваем metadata
                                for mk, mv in v.items():
                                    flat_item[f'metadata_{mk}'] = str(mv) if mv is not None else ""
                            elif isinstance(v, (str, int, float, bool)):
                                flat_item[k] = str(v) if v is not None else ""
                            elif v is None:
                                flat_item[k] = ""
                            else:
                                # Для всего остального - строковое представление
                                flat_item[k] = str(v)
                        
                        flat_items.append(flat_item)
                    
                    if flat_items:
                        # Получаем все уникальные поля
                        fieldnames = set()
                        for item in flat_items:
                            fieldnames.update(item.keys())
                        fieldnames = sorted(list(fieldnames))
                        
                        self.logger.info(f"📋 Поля для CSV: {fieldnames[:5]}... (всего {len(fieldnames)})")
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                        writer.writeheader()
                        
                        for flat_item in flat_items:
                            # Убеждаемся, что все поля есть
                            for field in fieldnames:
                                if field not in flat_item:
                                    flat_item[field] = ""
                            writer.writerow(flat_item)
                        
                        self.logger.info(f"💾 Сохранено {len(flat_items)} элементов в CSV формате")
                        
                        # Проверяем количество
                        if len(flat_items) != len(items):
                            self.logger.warning(f"⚠️ Потеря данных: {len(items)} → {len(flat_items)}")
                    else:
                        self.logger.error("❌ Нет данных для сохранения в CSV")
                
                else:
                    # Fallback на JSONL для неизвестных форматов
                    self.logger.warning(f"Неизвестный формат {file_ext}, использую JSONL")
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Проверяем количество сохраненных строк
            with open(file_path, 'r', encoding='utf-8') as check_f:
                if file_ext == '.csv':
                    # Для CSV считаем строки и вычитаем заголовок
                    lines = check_f.readlines()
                    saved_lines = len(lines) - 1
                    self.logger.info(f"📊 CSV файл: {len(lines)} всего строк, {saved_lines} записей")
                elif file_ext == '.jsonl':
                    saved_lines = sum(1 for _ in check_f)
                    self.logger.info(f"📊 JSONL файл: {saved_lines} записей")
                else:
                    saved_lines = sum(1 for _ in check_f)
            
            if saved_lines != len(items):
                self.logger.warning(f"⚠️ Несоответствие: ожидалось {len(items)} элементов, сохранено {saved_lines}")
            else:
                self.logger.success(f"✅ Корректно сохранено {saved_lines} элементов")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения в {file_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
            
    def prepare_dataset(self) -> bool:
        """Подготовка датасета с поддержкой sequence packing"""
        
        if self.config.sequence_packing:
            self.logger.info("📦 Sequence Packing активен — интерактивная загрузка")
            
            # Загружаем файлы и получаем количество физических файлов
            self.train_items, total_physical_files = self._interactive_file_loader()
            
            if not self.train_items:
                self.logger.error("❌ Не удалось загрузить файлы")
                return False
            
            # Упаковываем с правильным количеством файлов
            self.logger.info("📦 Упаковка документов в последовательности...")
            self.train_items = self._pack_sequences(self.train_items, total_physical_files)
            
            # 🔥 ИСПРАВЛЕНИЕ: Для Sequence Packing создаем валидационный датасет
            # Выделяем часть для валидации
            if len(self.train_items) > 5:
                val_size = max(1, int(len(self.train_items) * self.config.validation_split_ratio))
                indices = list(range(len(self.train_items)))
                random.shuffle(indices)
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]
                
                val_items = [self.train_items[i] for i in val_indices]
                self.train_items = [self.train_items[i] for i in train_indices]
                
                self.logger.info(f"📊 Разделение: train={len(self.train_items)}, val={len(val_items)}")
            else:
                val_items = []
                self.logger.warning("⚠️ Слишком мало данных для валидации, пропускаем")
            
            # Собираем тензоры для train
            all_input_ids = []
            all_labels = []
            
            for item in self.train_items:
                if isinstance(item['input_ids'], torch.Tensor):
                    input_ids = item['input_ids']
                else:
                    input_ids = torch.as_tensor(item['input_ids'], dtype=torch.long)
                
                all_input_ids.append(input_ids)
                all_labels.append(item['labels'] if isinstance(item['labels'], torch.Tensor) 
                                 else torch.as_tensor(item['labels'], dtype=torch.long))
            
            self.train_dataset = self.CachedDataset(all_input_ids, all_labels)
            
            # Создаем валидационный датасет
            if val_items:
                val_input_ids = []
                val_labels = []
                for item in val_items:
                    if isinstance(item['input_ids'], torch.Tensor):
                        input_ids = item['input_ids']
                    else:
                        input_ids = torch.as_tensor(item['input_ids'], dtype=torch.long)
                    val_input_ids.append(input_ids)
                    val_labels.append(item['labels'] if isinstance(item['labels'], torch.Tensor) 
                                     else torch.as_tensor(item['labels'], dtype=torch.long))
                self.val_dataset = self.CachedDataset(val_input_ids, val_labels)
            else:
                self.val_dataset = None
            
            self.dataset_validated = True
            
            total_tokens = sum(len(ids) for ids in all_input_ids)
            avg_length = total_tokens / len(all_input_ids) if all_input_ids else 0
            self.logger.success(f"✅ Упаковано {len(self.train_items)} последовательностей")
            self.logger.info(f"📊 Средняя длина: {avg_length:.1f} токенов, всего {total_tokens:,} токенов")
            
            return True
            
        else:
            # Стандартный workflow - требуем валидации
            if not self.dataset_validated:
                self.logger.error("Сначала выполните валидацию датасета!")
                return False
        
        try:
            if not self.train_items:
                self.logger.error("⚠️ Нет тренировочных данных!")
                return False
            
            # 🔥 ВАЖНО: Логируем информацию о чанках
            total_original = len(self.train_items) + len(self.val_items) if hasattr(self, 'val_items') and self.val_items else len(self.train_items)
            total_chunks = len(self.train_items) + len(self.val_items) if hasattr(self, 'val_items') and self.val_items else len(self.train_items)
            
            self.logger.info(f"📊 Статистика чанков:")
            self.logger.info(f"  • Оригинальных записей: ~{total_original}")
            self.logger.info(f"  • Всего чанков: {total_chunks}")
            if total_original > 0:
                self.logger.info(f"  • Коэффициент разбиения: {total_chunks/total_original:.2f}x")
            
            # 🔥 КЭШИРОВАНИЕ: проверяем, есть ли уже предобработанный датасет
            cache_path = os.path.join(self.config.output_dir, 'dataset_cache.pt')
            dataset_hash = self._compute_dataset_hash(self.train_items[:100])
            
            if os.path.exists(cache_path) and self._check_cache_valid(cache_path, dataset_hash):
                self.logger.info("📦 Загружаем кэшированный датасет...")
                try:
                    # Загружаем с маппингом на CPU (важно для разных устройств)
                    self.train_dataset = torch.load(cache_path, map_location='cpu')
                    if self.val_items:
                        val_cache = cache_path.replace('.pt', '_val.pt')
                        if os.path.exists(val_cache):
                            self.val_dataset = torch.load(val_cache, map_location='cpu')
                    self.logger.success(f"✅ Загружено из кэша: {len(self.train_dataset)} samples")
                    return True
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось загрузить кэш: {e}, создаем заново")
            
            # 📊 БЫСТРАЯ ОЦЕНКА ДЛИНЫ (без полной токенизации)
            self.logger.info("📏 Оценка длины текстов для сортировки...")
            
            # Используем быструю эвристику
            CHAR_PER_TOKEN = 3.5
            
            items_with_len = []
            for i, item in enumerate(self.train_items):
                try:
                    # 🔥 ПРОВЕРКА: уже токенизированные данные
                    if self.config.sequence_packing and 'input_ids' in item:
                        if isinstance(item['input_ids'], (list, torch.Tensor)):
                            estimated_tokens = len(item['input_ids'])
                        else:
                            estimated_tokens = 256
                    elif 'text' in item:
                        full_text = item['text']
                        estimated_tokens = len(full_text) // CHAR_PER_TOKEN
                    else:
                        prompt_text, answer_text = self._format_item_for_training(item)
                        full_text = f"{prompt_text} {answer_text}".strip()
                        estimated_tokens = len(full_text) // CHAR_PER_TOKEN
                    
                    items_with_len.append((i, estimated_tokens))
                except Exception:
                    items_with_len.append((i, 256))
            
            # 🔥 БЫСТРАЯ СОРТИРОВКА
            self.logger.info("📊 Сортировка по длине...")
            items_with_len.sort(key=lambda x: x[1])
            
            # 📦 БАКЕТИНГ (с защитой от малых датасетов)
            if len(self.train_items) < 100:
                # Для маленьких датасетов не делаем бакетирование
                num_buckets = 1
                bucket_size = len(self.train_items)
                self.logger.info(f"📦 Малый датасет ({len(self.train_items)}), пропускаем бакетирование")
            else:
                num_buckets = max(10, len(self.train_items) // 100)
                bucket_size = max(1, len(self.train_items) // num_buckets)
            
            self.logger.info(f"📦 Создание {num_buckets} бакетов (размер ~{bucket_size})...")
            buckets = []
            
            for i in range(0, len(self.train_items), bucket_size):
                bucket_indices = [idx for idx, _ in items_with_len[i:i+bucket_size]]
                bucket_items = [self.train_items[idx] for idx in bucket_indices]
                # Перемешиваем внутри бакета
                random.shuffle(bucket_items)
                buckets.append(bucket_items)
            
            # 🔥 Перемешиваем порядок бакетов
            random.shuffle(buckets)
            self.logger.info(f"✅ Бакеты перемешаны, порядок рандомизирован")
            
            self.train_items = [item for bucket in buckets for item in bucket]
            self.logger.info(f"✅ Сортировка завершена: min={items_with_len[0][1]:.1f}, max={items_with_len[-1][1]:.1f} tokens")
            
            # 🔥 БАТЧ-ТОКЕНИЗАЦИЯ ТРЕЙНА
            self.logger.info("🔄 Токенизация трейн датасета (batch processing)...")
            all_input_ids = []
            all_labels = []
            eos_token = self.tokenizer.eos_token or "</s>"
            batch_size = 32

            # Прогресс-бар для токенизации
            from tqdm import tqdm
            with tqdm(total=len(self.train_items), desc="Токенизация train", disable=not self.config.use_progress_bar) as pbar:
                for i in range(0, len(self.train_items), batch_size):
                    batch_items = self.train_items[i:i+batch_size]
                    
                    # 🔥 ПРОВЕРКА: уже токенизированные данные (sequence packing)
                    if self.config.sequence_packing and batch_items and 'input_ids' in batch_items[0]:
                        # Данные уже токенизированы в _pack_sequences
                        for item in batch_items:
                            input_ids = item['input_ids']
                            labels = item.get('labels', None)
                            
                            # 🔥 Конвертируем в тензор если это список
                            if isinstance(input_ids, list):
                                input_ids = torch.tensor(input_ids, dtype=torch.long)
                            if isinstance(labels, list):
                                labels = torch.tensor(labels, dtype=torch.long)
                            
                            # Проверка на пустые
                            if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                                all_input_ids.append(input_ids)
                                # Если labels нет — создаём маску (все токены учим)
                                if labels is None:
                                    labels = torch.full_like(input_ids, -100)
                                all_labels.append(labels)
                            else:
                                self.logger.warning(f"Пустой input_ids в элементе")
                        
                        pbar.update(len(batch_items))
                        continue  # 🔥 ПРОПУСКАЕМ стандартную токенизацию
                    
                    # Стандартная токенизация (без sequence packing)
                    batch_prompts = []
                    batch_answers = []
                    for item in batch_items:
                        prompt_text, answer_text = self._format_item_for_training(item)
                        batch_prompts.append(prompt_text)
                        batch_answers.append(answer_text)
                    
                    # Токенизируем промпты
                    prompt_encoded = self.tokenizer(
                        batch_prompts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.config.max_length,
                        padding=False
                    )
                    # Токенизируем ответы с EOS
                    answers_with_eos = [ans + eos_token for ans in batch_answers]
                    answer_encoded = self.tokenizer(
                        answers_with_eos,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.config.max_length,
                        padding=False
                    )
                    # Собираем финальные тензоры
                    for j in range(len(batch_items)):
                        prompt_ids = prompt_encoded['input_ids'][j]
                        answer_ids = answer_encoded['input_ids'][j]
                        # Обрезаем если слишком длинные
                        total_len = len(prompt_ids) + len(answer_ids)
                        if total_len > self.config.max_length:
                            max_prompt_len = self.config.max_length - len(answer_ids)
                            if max_prompt_len > 0:
                                prompt_ids = prompt_ids[-max_prompt_len:]
                            else:
                                answer_ids = answer_ids[:self.config.max_length]
                                prompt_ids = []
                        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
                        labels = torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)
                        all_input_ids.append(input_ids)
                        all_labels.append(labels)
                    pbar.update(len(batch_items))
            
            # 🔥 ТОКЕНИЗАЦИЯ ВАЛИДАЦИИ (ПОЛНАЯ, БЕЗ ПРОПУСКОВ)
            if self.val_items:
                self.logger.info(f"🔄 Токенизация валидации ({len(self.val_items)} samples)...")
                val_input_ids = []
                val_labels = []
                
                with tqdm(total=len(self.val_items), desc="Токенизация val", disable=not self.config.use_progress_bar) as pbar:
                    for item in self.val_items:
                        # 🔥 ПРОВЕРКА: уже токенизированные данные
                        if self.config.sequence_packing and 'input_ids' in item:
                            input_ids = item['input_ids']
                            labels = item.get('labels', torch.full_like(input_ids, -100))
                            
                            if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                                val_input_ids.append(input_ids)
                                val_labels.append(labels)
                            elif isinstance(input_ids, list) and len(input_ids) > 0:
                                val_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                                val_labels.append(torch.tensor(labels, dtype=torch.long) if isinstance(labels, list) else labels)
                            
                            pbar.update(1)
                            continue
                        
                        # Для sequence packing используем текст напрямую
                        if self.config.sequence_packing and 'text' in item:
                            prompt_text = item['text']
                            answer_text = ""
                        else:
                            prompt_text, answer_text = self._format_item_for_training(item)
                        
                        # Токенизируем промпт
                        prompt_ids = self.tokenizer.encode(
                            prompt_text, 
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.config.max_length
                        )
                        
                        # Токенизируем ответ с EOS (если есть)
                        if answer_text:
                            answer_ids = self.tokenizer.encode(
                                answer_text + eos_token, 
                                add_special_tokens=False,
                                truncation=True,
                                max_length=self.config.max_length
                            )
                        else:
                            answer_ids = []
                        
                        # Обрезаем если слишком длинные
                        total_len = len(prompt_ids) + len(answer_ids)
                        if total_len > self.config.max_length:
                            max_prompt_len = self.config.max_length - len(answer_ids)
                            if max_prompt_len > 0:
                                prompt_ids = prompt_ids[-max_prompt_len:]
                            else:
                                answer_ids = answer_ids[:self.config.max_length]
                                prompt_ids = []
                        
                        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
                        labels = torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)
                        
                        val_input_ids.append(input_ids)
                        val_labels.append(labels)
                        
                        pbar.update(1)
                
                self.logger.info(f"✅ Валидация токенизирована: {len(val_input_ids)} samples")
            else:
                val_input_ids = None
                val_labels = None
            
            # 🔥 СОЗДАЕМ КЭШИРОВАННЫЙ ДАТАСЕТ
            self.train_dataset = self.CachedDataset(all_input_ids, all_labels)
            if val_input_ids:
                self.val_dataset = self.CachedDataset(val_input_ids, val_labels)
            else:
                self.val_dataset = None
            
            # Сохраняем в кэш
            self.logger.info("💾 Сохранение кэша датасета...")
            
            try:
                # Сохраняем train
                temp_cache = cache_path + '.tmp'
                torch.save(self.train_dataset, temp_cache, pickle_protocol=4)
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                os.rename(temp_cache, cache_path)
                
                # Сохраняем хеш
                hash_path = cache_path.replace('.pt', '_hash.txt')
                with open(hash_path, 'w') as f:
                    f.write(dataset_hash)
                
                # Сохраняем validation если есть
                if self.val_dataset:
                    val_cache = cache_path.replace('.pt', '_val.pt')
                    temp_val = val_cache + '.tmp'
                    torch.save(self.val_dataset, temp_val, pickle_protocol=4)
                    
                    if os.path.exists(val_cache):
                        os.remove(val_cache)
                    os.rename(temp_val, val_cache)
                
                self.logger.success(f"✅ Кэш сохранен: {cache_path}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось сохранить кэш: {e}")
                # Продолжаем работу даже без кэша
            
            # Статистика
            total_tokens = sum(len(ids) for ids in all_input_ids)
            avg_length = total_tokens / len(all_input_ids) if all_input_ids else 0
            self.logger.info(f"📊 Итог: {len(all_input_ids)} train, {len(val_input_ids) if val_input_ids else 0} val, "
                            f"avg {avg_length:.1f} tokens, total {total_tokens:,} tokens")
            
            return True
            
        except Exception as e:
            self.logger.error(f"⚠️ Critical in prepare_dataset: {e}")
            traceback.print_exc()
            return False
            
    class CachedDataset(Dataset):
        """Кэшированный датасет для быстрой загрузки"""
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'labels': self.labels[idx],
                'attention_mask': torch.ones_like(self.input_ids[idx])
            }
        
        def __getstate__(self):
            return {
                'input_ids': self.input_ids,
                'labels': self.labels
            }
        
        def __setstate__(self, state):
            self.input_ids = state['input_ids']
            self.labels = state['labels']
        
    def _compute_dataset_hash(self, samples) -> str:
        """Вычисление хеша датасета с учетом всех важных параметров"""
        try:
            # 🔥 ИСПРАВЛЕНО: Добавляем параметры модели
            hash_components = [
                str([str(s) for s in samples[:100]]),  # первые 100 семплов
                str(self.config.max_length),
                str(self.config.min_token_length),
                self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else "unknown",
                str(self.config.split_long_texts),
                str(self.config.overlap_ratio),
                str(self.config.sequence_packing),
                str(self.config.packing_max_tokens),
                # 🔥 ВАЖНО: Добавляем параметры модели
                str(self.model.config.vocab_size) if self.model else "unknown",
                str(self.model.config.model_type) if self.model else "unknown",
            ]
            
            hash_input = "|".join(hash_components).encode()
            return hashlib.md5(hash_input).hexdigest()
        except Exception as e:
            self.logger.debug(f"Hash computation failed: {e}")
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _check_cache_valid(self, cache_path: str, current_hash: str) -> bool:
        """Проверка валидности кэша"""
        try:
            hash_path = cache_path.replace('.pt', '_hash.txt')
            if not os.path.exists(hash_path):
                return False
            
            with open(hash_path, 'r') as f:
                saved_hash = f.read().strip()
            
            # 🔥 ИСПРАВЛЕНО: Проверяем размер файла и хеш
            if saved_hash != current_hash:
                self.logger.info("📦 Хеш изменился, создаем новый кэш")
                return False
            
            if os.path.getsize(cache_path) == 0:
                self.logger.info("📦 Кэш пустой, создаем новый")
                return False
            
            # Пробуем загрузить для проверки целостности
            try:
                test_load = torch.load(cache_path, map_location='cpu')
                if test_load is None:
                    return False
            except:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Cache validation failed: {e}")
            return False            

    def prepare_training(self) -> bool:
        """Подготовка к обучению с динамическим батчингом и умной коллацией"""
        
        # 🔥 ИСПРАВЛЕНО: Sequence Packing не требует валидации
        if self.config.sequence_packing:
            self.logger.info("📦 Sequence Packing режим - пропускаем стандартную валидацию")
            if not self.prepare_dataset():
                return False
        else:
            # Стандартный режим - требуем валидации
            if not self.dataset_validated:
                self.logger.error("Сначала выполните валидацию датасета!")
                return False
            
            if not self.prepare_dataset():
                return False
        
        # 🔥 ДИНАМИЧЕСКИЙ ПОДБОР БАТЧА
        effective_batch_size = self.config.batch_size
        if isinstance(effective_batch_size, str) and effective_batch_size.lower() == "auto":
            effective_batch_size = self.auto_find_batch_size()
        effective_batch_size = int(effective_batch_size)
        
        # Получаем pad_id для коллации
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
            self.logger.debug("pad_token_id is None, using 0 as fallback")
        
        # 🔥 ИСПРАВЛЕНО: Создаем partial функцию для коллации
        collate_fn = partial(self._smart_collate_impl, pad_id=pad_id)
        
        # DataLoader с оптимизациями
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,  # Теперь правильно!
            num_workers=0,
            collate_fn=collate_fn,  # Используем partial функцию
            pin_memory=self.device.type == 'cuda',
            drop_last=True
        )
        
        # Для валидации shuffle не нужен
        if self.val_dataset and len(self.val_dataset) > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=effective_batch_size,
                shuffle=False,  # Валидация без перемешивания
                num_workers=0,
                collate_fn=collate_fn,  # Та же partial функция
                pin_memory=self.device.type == 'cuda',
                drop_last=False
            )
        else:
            self.val_dataloader = None
            self.logger.warning("Валидационный датасет пустой, пропускаем создание val_dataloader")
        
        # ⚠️ ПРОВЕРКА: DataLoader может работать
        try:
            test_iterator = iter(self.dataloader)
            test_batch = next(test_iterator)
            if test_batch is None:
                self.logger.error("DataLoader возвращает None батчи")
                return False
            self.logger.info(f"✅ DataLoader проверен: батч размером {test_batch['input_ids'].shape}")
        except Exception as e:
            self.logger.error(f"Ошибка в DataLoader: {e}")
            # Пробуем создать более простой DataLoader
            self.logger.info("Пробуем создать DataLoader с упрощенными настройками...")
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size=min(effective_batch_size, 2),  # Уменьшаем batch_size
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,  # Все та же partial функция
                pin_memory=self.device.type == 'cuda'
            )
            # Повторная проверка
            try:
                test_iterator = iter(self.dataloader)
                test_batch = next(test_iterator)
                if test_batch is None:
                    self.logger.error("Упрощенный DataLoader тоже не работает")
                    return False
            except Exception as e2:
                self.logger.error(f"Упрощенный DataLoader также не работает: {e2}")
                return False
        
        self.model.train()       
        
        # === Оптимизатор ===
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
            if self.optimizer is None:
                self.logger.error("Не удалось создать оптимизатор!")
                return False
            self.logger.info(f"✅ Optimizer created: {type(self.optimizer).__name__}")
        
        # === Шедулер ===
        total_steps = len(self.dataloader) * int(self.config.epochs or 1)
        total_steps = max(1, total_steps)
        
        if self.config.lr_scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2, verbose=False
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config.warmup_steps or 0),
                num_training_steps=total_steps
            )
        
        self.logger.info(f"✅ Scheduler created: total_steps={total_steps}, warmup={self.config.warmup_steps or 0}")
        
        # Mixed precision
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            from torch.amp import GradScaler
            self.scaler = GradScaler(
                device='cuda',
                init_scale=1024.0,
                growth_interval=2000,
                growth_factor=2.0
            )
            self.logger.info("✅ Mixed precision с безопасными настройками")
        else:
            self.scaler = None
            if self.config.use_mixed_precision:
                self.logger.info("🔴 Mixed precision отключен (не CUDA)")
        
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            SummaryWriter_available = True
        except ImportError:
            SummaryWriter_available = False
            self.logger.warning("TensorBoard not installed: pip install tensorboard")
        
        if SummaryWriter_available and self.writer is None:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, 'runs'))
                self.logger.info(f"TensorBoard logs: {os.path.join(self.config.output_dir, 'runs')}")
            except Exception as e:
                self.writer = None
                self.logger.warning(f"Failed to init TensorBoard: {e}")
        
        self.logger.info(f"Training prepared: batch_size={effective_batch_size}, steps/epoch={len(self.dataloader)}")
        if self.val_dataloader:
            self.logger.info(f"Validation batches: {len(self.val_dataloader)}")
        
        return True

    def _smart_collate_impl(self, batch, pad_id):
        """
        Реализация умной коллации.
        Вызывается через partial, получает pad_id как дополнительный аргумент.
        """
        if not batch:
            return None
        
        # Сортируем по длине внутри батча
        batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
        
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        # Паддинг input_ids
        padded_input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=pad_id
        )
        
        # Паддинг labels (используем -100 для ignore index)
        padded_labels = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100  # -100 игнорируется в loss
        )
        
        # Паддинг attention_mask
        padded_masks = pad_sequence(
            attention_masks, 
            batch_first=True, 
            padding_value=0
        )
        
        return {
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': padded_masks
        }
        
    def auto_find_batch_size(self) -> int:
        """Автоматический подбор размера батча, который помещается в VRAM"""
        if self.device.type != 'cuda':
            self.logger.info("Автоподбор батча доступен только для CUDA. Установлен размер 2.")
            return 2
        test_batch_size = 8 # Начинаем с меньшего значения
        self.logger.info(f"Начинаем поиск batch_size с {test_batch_size}...")
        while test_batch_size > 0:
            try:
                torch.cuda.empty_cache()
                dummy_input = torch.randint(0, self.tokenizer.vocab_size, (test_batch_size, self.config.max_length), device=self.device)
                dummy_labels = torch.randint(0, self.tokenizer.vocab_size, (test_batch_size, self.config.max_length), device=self.device)
                with autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float16, enabled=self.config.use_mixed_precision):
                    outputs = self.model(input_ids=dummy_input, labels=dummy_labels)
                    loss = outputs.loss
                loss.backward()
                self.optimizer.zero_grad(set_to_none=True)
                del dummy_input, dummy_labels, outputs, loss
                torch.cuda.empty_cache()
                self.logger.info(f"Проверка batch_size={test_batch_size}: OK")
                return max(1, test_batch_size // 2) # Возвращаем половину, чтобы иметь запас
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.warning(f"OOM при batch_size={test_batch_size}, пробуем {test_batch_size // 2}")
                    test_batch_size //= 2
                    torch.cuda.empty_cache()
                else:
                    self.logger.error(f"Ошибка при подборе batch_size: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Неожиданная ошибка при подборе batch_size: {e}")
                return 1 # Fallback
        self.logger.error("Не удалось подобрать batch_size, даже 1 не помещается в память!")
        return 1
        
# ===================================================================
# МЕТОДЫ ОБУЧЕНИЯ
# ===================================================================

    def train(self):
        """Основной цикл обучения - чистый и быстрый"""
        if not self.model or not self.dataloader:
            self.logger.error("Training cannot start. Model or data is not ready.")
            return False
        
        self.model.train()
        self.global_step = 0
        
        self.logger.info("🚀 Starting training")
        
        patience_counter = 0
        last_epoch_stats = {}
        
        try:
            for epoch in range(int(self.config.epochs)):
                self.current_epoch_loss = 0.0
                self.current_epoch_steps = 0
                self.gradient_accumulation_steps_counter = 0
                
                # 🔥 ИСПРАВЛЕНО: Используем enumerate вместо ручного итератора
                total_batches = len(self.dataloader)
                
                # Прогресс-бар
                with tqdm(total=total_batches, 
                         desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                         disable=not self.config.use_progress_bar) as pbar:
                    
                    for step, batch in enumerate(self.dataloader):
                        try:
                            # 🔥 ИСПРАВЛЕНО: Проверка пустого батча
                            if batch is None:
                                pbar.update(1)
                                continue
                            
                            # Проверка наличия обязательных полей
                            if 'input_ids' not in batch or 'labels' not in batch:
                                self.logger.warning(f"Batch missing required fields at step {step}")
                                pbar.update(1)
                                continue
                            
                            # Проверка размера батча
                            if batch['input_ids'].shape[0] == 0:
                                self.logger.warning(f"Empty batch at step {step}")
                                pbar.update(1)
                                continue
                            
                            # Проверка на слишком длинные последовательности
                            if batch['input_ids'].shape[1] > self.config.max_length:
                                self.logger.warning(f"Sequence too long ({batch['input_ids'].shape[1]} > {self.config.max_length}), truncating")
                                batch['input_ids'] = batch['input_ids'][:, :self.config.max_length]
                                if 'attention_mask' in batch:
                                    batch['attention_mask'] = batch['attention_mask'][:, :self.config.max_length]
                                if 'labels' in batch:
                                    batch['labels'] = batch['labels'][:, :self.config.max_length]
                            
                            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                            loss = self._training_step(batch)
                            
                            if loss is not None:
                                self.current_epoch_loss += loss
                                self.current_epoch_steps += 1
                                
                                # Обновление прогресс-бара
                                if (step + 1) % 10 == 0:
                                    avg_loss = self.current_epoch_loss / self.current_epoch_steps
                                    pbar.set_description(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
                            
                            pbar.update(1)
                            
                            # Сохранение чекпоинта
                            if self.global_step % self.config.save_steps == 0:
                                self._save_checkpoint(self.global_step)
                            
                        except StopIteration:
                            break
                        except Exception as e:
                            self.logger.error(f"Error in step {step}: {e}")
                            pbar.update(1)
                            continue
                
                # Метрики эпохи
                avg_epoch_loss = self.current_epoch_loss / self.current_epoch_steps if self.current_epoch_steps > 0 else float('inf')
                perplexity = math.exp(min(avg_epoch_loss, 20))
                
                self.logger.info(f"Epoch {epoch + 1} completed: Loss={avg_epoch_loss:.4f}, PPL={perplexity:.2f}")
                
                # ✅ ГЕНЕРАЦИЯ ПОСЛЕ ЭПОХИ (не влияет на скорость обучения)
                if self.config.save_validation_generations:
                    try:
                        self.model.eval()
                        generated_text = self.generate_text(self.config.eval_prompt)
                        
                        # Красивый вывод
                        self.logger.console.print(Panel(
                            generated_text[:500] + ("..." if len(generated_text) > 500 else ""),
                            title=f"📝 Генерация после эпохи {epoch+1}",
                            style="green"
                        ))
                        
                        # Сохраняем в файл
                        with open(os.path.join(self.config.output_dir, 'generations.txt'), 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*50}\n")
                            f.write(f"Epoch {epoch+1} (Step {self.global_step}):\n")
                            f.write(f"Prompt: {self.config.eval_prompt}\n")
                            f.write(f"Generated: {generated_text}\n")
                        
                        self.model.train()
                    except Exception as e:
                        self.logger.error(f"Generation failed: {e}")
                        self.model.train()  # Важно вернуть в режим обучения
                
                # Валидация (если нужно)
                if self._should_validate(epoch + 1) and self.val_dataloader:
                    val_loss, val_perplexity = self.evaluate()
                    self.logger.info(f"Validation: Loss={val_loss:.4f}, PPL={val_perplexity:.2f}")
                    
                    # Early stopping
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        patience_counter = 0
                        self._save_checkpoint("best")
                    else:
                        patience_counter += 1
                        if self.config.early_stopping_patience > 0 and patience_counter >= self.config.early_stopping_patience:
                            self.logger.warning(f"Early stopping triggered after {patience_counter} epochs")
                            break
                
                # Очистка памяти после эпохи
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Сохраняем финальную модель
            self._save_final_model(avg_epoch_loss, perplexity)
            self.logger.info("✅ Training completed successfully")
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_checkpoint("interrupted")
            return False
        except Exception as e:
            self.logger.critical(f"Critical error during training: {e}")
            traceback.print_exc()
            return False

    def _should_validate(self, current_epoch: int) -> bool:
        if self.config.validation_mode == "disabled":
            return False
        if self.config.validation_mode == "final_only":
            return current_epoch == self.config.epochs
        if self.config.validation_mode == "each_epoch":
            return current_epoch % self.config.validation_interval_epochs == 0
        return True # fallback

    def _training_step(self, batch) -> Optional[float]:
        """Чистый training step - только обучение"""
        try:
            # 🔥 ИСПРАВЛЕНО: Компилируем только на CUDA
            if self.device.type == 'cuda' and not hasattr(self, '_model_compiled'):
                self.logger.info("🔥 Компилируем модель (reduce-overhead + sdpa)...")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        backend="inductor",
                        fullgraph=False,
                        dynamic=False
                    )
                    self._model_compiled = True
                    self.logger.success("✅ Модель скомпилирована для CUDA")
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось скомпилировать модель: {e}")
                    self._model_compiled = True

            # Forward
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type, enabled=self.config.use_mixed_precision):
                outputs = self.model(**batch)
                
                raw_loss = outputs.loss
                
                if raw_loss is None:
                    self.logger.error("❌ Loss is None!")
                    return None

                if torch.isnan(raw_loss).any():
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.debug(f"📊 input_ids stats: min={batch['input_ids'].min()}, max={batch['input_ids'].max()}")
                    self.logger.debug(f"📊 labels stats: min={batch['labels'].min()}, max={batch['labels'].max()}")
                    return None

                if torch.isinf(raw_loss).any():
                    self.logger.error("❌ Inf loss detected!")
                    return None
                
                loss_value = raw_loss.mean().item()  

            # Backward
            scaled_loss = raw_loss / self.config.gradient_accumulation_steps
            if scaled_loss.dim() > 0:
                scaled_loss = scaled_loss.mean()
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            self.gradient_accumulation_steps_counter += 1
            self.global_step += 1

            # Оптимизатор (когда накопили)
            if self.gradient_accumulation_steps_counter % self.config.gradient_accumulation_steps == 0:
                
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # ===========================================================
                # 🔥🔥🔥 ДИАГНОСТИКА ГРАДИЕНТОВ (добавлено)
                # ===========================================================
                if self.global_step % 200 == 0:
                    largest_grad = None
                    largest_val = 0.0
                    
                    hebb_grads = []
                    mamba_grads = []
                    gate_grads = []
                    other_grads = []
                    
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            
                            # Находим самый большой градиент
                            if grad_norm > largest_val:
                                largest_val = grad_norm
                                largest_grad = name
                            
                            # Сортируем по компонентам
                            if 'hebb' in name.lower():
                                hebb_grads.append(grad_norm)
                            elif 'mamba' in name.lower() or 'adaptive_memory' in name.lower():
                                mamba_grads.append(grad_norm)
                            elif 'gate_' in name.lower() or 'memory_gate' in name.lower():
                                gate_grads.append(grad_norm)
                            else:
                                other_grads.append(grad_norm)
                    
                    self.logger.info(f"📊 Largest grad: {largest_grad} = {largest_val:.6f}")
                    
                    if hebb_grads:
                        self.logger.info(f"📊 Hebb avg grad: {sum(hebb_grads)/len(hebb_grads):.6f} (max: {max(hebb_grads):.6f})")
                    if mamba_grads:
                        self.logger.info(f"📊 Mamba avg grad: {sum(mamba_grads)/len(mamba_grads):.6f} (max: {max(mamba_grads):.6f})")
                    if gate_grads:
                        self.logger.info(f"📊 Gate weights avg grad: {sum(gate_grads)/len(gate_grads):.6f} (max: {max(gate_grads):.6f})")
                    if other_grads:
                        self.logger.info(f"📊 Other avg grad: {sum(other_grads)/len(other_grads):.6f}")

                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler and self.config.lr_scheduler_type != "plateau":
                    self.scheduler.step()

                # Мониторинг градиентов (упрощённый)
                if self.global_step % self.config.log_gradients_steps == 0:
                    self._log_gradient_stats()

                if (self.device.type == 'cuda' and 
                    self.global_step % self.config.cache_clear_steps == 0):
                    torch.cuda.empty_cache()

            return loss_value

        except Exception as e:
            self.logger.error(f"❌ Step error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            self.optimizer.zero_grad()
            if self.scaler:
                pass
                
            return None

    def _log_gradient_stats(self):
        """Быстрый мониторинг градиентов (без тяжелых вычислений)"""
        try:
            total_norm = 0.0
            count = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    param_norm = param.grad.norm().item()
                    total_norm += param_norm ** 2
                    count += 1
                    
                    # Логируем только самые большие градиенты (для отладки)
                    if param_norm > 10.0 and self.config.verbose:
                        self.logger.warning(f"Large gradient: {name} = {param_norm:.3f}")
            
            if count > 0:
                rms_grad = math.sqrt(total_norm / count)
                self.logger.info(f"Step {self.global_step}: Grad RMS = {rms_grad:.4f}")
                
                # Предупреждение о взрыве градиентов
                if rms_grad > 100:
                    self.logger.warning(f"⚠️ GRADIENT EXPLOSION! RMS={rms_grad:.1f}")
        except Exception as e:
            self.logger.debug(f"Gradient logging failed: {e}")

    def _log_memory_stats(self):
        """Быстрая статистика памяти (без матриц)"""
        if not hasattr(self.model, 'memory_system'):
            return
        
        try:
            mem = self.model.memory_system
            if hasattr(mem, 'memory_vectors') and mem.memory_vectors is not None:
                with torch.no_grad():
                    norms = torch.norm(mem.memory_vectors, dim=1)
                    active = (norms > 0.1).sum().item()
                    total = mem.memory_vectors.shape[0]
                    
                    # Только базовая статистика
                    self.logger.info(f"Memory: {active}/{total} slots active")
                    
                    # ОЧЕНЬ редко - полная проверка дубликатов
                    if self.global_step % 5000 == 0:  # раз в 5000 шагов
                        self._check_memory_duplicates()
        except Exception as e:
            self.logger.debug(f"Memory stats failed: {e}")

    def _check_memory_duplicates(self):
        """Тяжелая проверка дубликатов - вызывать редко с защитой от больших матриц"""
        if not hasattr(self.model, 'memory_system'):
            return
        
        try:
            mem = self.model.memory_system
            with torch.no_grad():
                if not hasattr(mem, 'memory_vectors') or mem.memory_vectors is None:
                    return
                
                num_slots = mem.memory_vectors.shape[0]
                
                # 🔥 ЗАЩИТА: если слотов слишком много - пропускаем
                MAX_SLOTS_FOR_DUPLICATE_CHECK = 1000
                if num_slots > MAX_SLOTS_FOR_DUPLICATE_CHECK:
                    self.logger.debug(f"Слишком много слотов ({num_slots}) для проверки дубликатов, пропускаем")
                    return
                
                # Нормализуем вектора
                mem_norm = F.normalize(mem.memory_vectors, dim=1)
                
                # Считаем попарные сходства (тяжело для больших матриц!)
                sim_matrix = torch.mm(mem_norm, mem_norm.t())
                sim_matrix.fill_diagonal_(0)
                
                # Находим дубликаты
                max_sim, _ = torch.max(sim_matrix, dim=1)
                dup_count = (max_sim > 0.9).sum().item()
                
                if dup_count > 0:
                    dup_percent = dup_count / num_slots * 100
                    self.logger.warning(f"⚠️ Найдено {dup_count} дубликатов ({dup_percent:.1f}%)")
                    
                    # Опционально: очистка дубликатов
                    if dup_percent > 20:  # если >20% дубликатов
                        self.logger.warning("🔥 Высокий уровень дубликатов! Рассмотрите сброс памяти.")
        except Exception as e:
            self.logger.debug(f"Duplicate check failed: {e}")
        
    def _get_differential_param_groups(self) -> List[Dict]:
        """Создает группы параметров с разными learning rates для Thalia"""
        if not self.config.use_differential_lr:
            return [{'params': self.model.parameters(), 'lr': self.config.learning_rate}]
        
        param_groups = []
        base_lr = self.config.learning_rate
        
        # 🔥 АВТООПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ
        is_thalia_arch = (
            hasattr(self.model, 'personality_core') and
            hasattr(self.model, 'adaptive_memory')
        )
        
        if is_thalia_arch:
            # 🔥 НОВАЯ АРХИТЕКТУРА THALIA
            self.logger.info("🔍 Обнаружена архитектура Thalia (Personality + Mamba)")
            return self._get_thalia_param_groups()
        else:
            # 🔥 СТАРАЯ АРХИТЕКТУРА (совместимость)
            self.logger.info("🔍 Используется старая архитектура (совместимость)")
            return self._get_legacy_param_groups()

    def _get_thalia_param_groups(self) -> List[Dict]:
        """Группы параметров для новой архитектуры Thalia"""
        base_lr = self.config.learning_rate
        param_groups = []
        
        # 🔥 ПРАВИЛЬНЫЕ КЛЮЧИ (должны совпадать с конфигом)
        default_mult = {
            'embeddings': 0.05,          # ← ДОБАВЛЕНО! эмбеддинги в 20 раз медленнее
            'transformer_base': 0.2,
            'personality_core': 0.5,
            'living_layer': 0.8,
            'adaptive_memory': 1.8,
            'experience_exchange': 1.2,
            'memory_system': 0.3,
            'other': 0.3
        }
        
        # Берём из конфига или используем default
        if hasattr(self.config, 'lr_multipliers') and self.config.lr_multipliers:
            mult = self.config.lr_multipliers
            self.logger.info("📊 Используются множители LR из конфига:")
            for key, value in mult.items():
                self.logger.info(f"   • {key}: ×{value}")
        else:
            mult = default_mult
            self.logger.info("📊 Используются множители LR по умолчанию")
        
        # Собираем параметры по компонентам
        components = {
            'embeddings': [],            # ← ДОБАВЛЕНО!
            'transformer_base': [],
            'personality_core': [],
            'living_layer': [],
            'adaptive_memory': [],
            'experience_exchange': [],
            'memory_system': [],
            'other': []
        }
        
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # 🔥 СНАЧАЛА ПРОВЕРЯЕМ ЭМБЕДДИНГИ
            if 'wte' in n or 'wpe' in n:
                components['embeddings'].append(p)
                continue
            
            # Остальная классификация
            if n.startswith('transformer.'):
                if 'h.' in n:
                    try:
                        layer_num = int(n.split('.h.')[1].split('.')[0])
                        total_layers = len(self.model.transformer.h)
                        
                        if layer_num < total_layers // 3:
                            lr_mult_key = 'transformer_base'
                        elif layer_num < 2 * total_layers // 3:
                            lr_mult_key = 'transformer_middle'
                        else:
                            lr_mult_key = 'transformer_top'
                        
                        if lr_mult_key not in components:
                            components[lr_mult_key] = []
                        components[lr_mult_key].append(p)
                    except:
                        components['transformer_base'].append(p)
                elif 'ln_f' in n:
                    components['transformer_base'].append(p)
                else:
                    components['transformer_base'].append(p)
            
            elif n.startswith('personality_core.'):
                components['personality_core'].append(p)
            
            elif n.startswith('living_layer.'):
                components['living_layer'].append(p)
            
            elif n.startswith('adaptive_memory.'):
                components['adaptive_memory'].append(p)
            
            elif n.startswith('experience_exchange.'):
                components['experience_exchange'].append(p)
            
            elif n.startswith('memory_system.'):
                components['memory_system'].append(p)
            
            else:
                components['other'].append(p)
        
        # Создаём группы с правильными LR
        for comp_name, params in components.items():
            if not params:
                continue
            
            lr_mult = mult.get(comp_name, 1.0)
            actual_lr = base_lr * lr_mult
            
            param_groups.append({
                'params': params,
                'lr': actual_lr,
                'name': comp_name
            })
            
            self.logger.info(f"📊 {comp_name}: {len(params):5d} params, "
                            f"LR={actual_lr:.2e} (×{lr_mult:.1f})")
        
        # Проверяем неиспользованные ключи
        if hasattr(self.config, 'lr_multipliers') and self.config.lr_multipliers:
            unused_keys = set(self.config.lr_multipliers.keys()) - set(components.keys())
            if unused_keys:
                self.logger.warning(f"⚠️ Неиспользованные ключи в lr_multipliers: {unused_keys}")
        
        return param_groups

    def _get_legacy_param_groups(self) -> List[Dict]:
        """Группы параметров для старой архитектуры (совместимость)"""
        base_lr = self.config.learning_rate
        param_groups = []
        
        # 🔥 СТАРАЯ ЛОГИКА (с небольшими улучшениями)
        num_layers = len(self.model.transformer.h) if hasattr(self.model, 'transformer') else 0
        
        # 1. Embeddings
        emb_params = [p for n, p in self.model.named_parameters()
                      if ('wte' in n or 'wpe' in n) and p.requires_grad]
        if emb_params:
            lr_mult = self.config.lr_multipliers.get('embeddings', 0.1)
            param_groups.append({
                'params': emb_params,
                'lr': base_lr * lr_mult,
                'name': 'embeddings'
            })
        
        # 2. Transformer layers
        if num_layers > 0:
            early_idx = num_layers // 3
            middle_idx = 2 * num_layers // 3
            
            for i in range(num_layers):
                layer_params = [p for n, p in self.model.transformer.h[i].named_parameters()
                               if p.requires_grad]
                if not layer_params:
                    continue
                
                if i < early_idx:
                    mult_key = 'bottom_layers'
                elif i < middle_idx:
                    mult_key = 'middle_layers'
                else:
                    mult_key = 'top_layers'
                
                lr_mult = self.config.lr_multipliers.get(mult_key, 1.0)
                param_groups.append({
                    'params': layer_params,
                    'lr': base_lr * lr_mult,
                    'name': f'transformer_layer_{i}'
                })
        
        # 3. Output layers
        output_params = [p for n, p in self.model.named_parameters()
                        if ('lm_head' in n or 'ln_f' in n) and p.requires_grad]
        if output_params:
            lr_mult = self.config.lr_multipliers.get('output_layers', 1.5)
            param_groups.append({
                'params': output_params,
                'lr': base_lr * lr_mult,
                'name': 'output_layers'
            })
        
        # 4. Memory system (если есть)
        memory_params = [p for n, p in self.model.named_parameters()
                        if 'memory_system' in n and p.requires_grad]
        if memory_params:
            lr_mult = self.config.lr_multipliers.get('memory_system', 1.8)
            param_groups.append({
                'params': memory_params,
                'lr': base_lr * lr_mult,
                'name': 'memory_system'
            })
        
        # 5. Memory heads (если есть)
        heads_params = [p for n, p in self.model.named_parameters()
                       if 'memory_heads' in n and p.requires_grad]
        if heads_params:
            lr_mult = self.config.lr_multipliers.get('memory_heads', 2.0)
            param_groups.append({
                'params': heads_params,
                'lr': base_lr * lr_mult,
                'name': 'memory_heads'
            })
        
        # 6. Остальное
        other_params = [p for n, p in self.model.named_parameters()
                       if p.requires_grad and not any(
                           key in n for key in ['wte', 'wpe', 'h.', 'ln_f', 'lm_head', 
                                               'memory_system', 'memory_heads']
                       )]
        if other_params:
            lr_mult = self.config.lr_multipliers.get('other', 0.3)
            param_groups.append({
                'params': other_params,
                'lr': base_lr * lr_mult,
                'name': 'other'
            })
        
        # Логирование
        for group in param_groups:
            self.logger.info(f"📊 {group['name']}: {len(group['params']):5d} params, "
                            f"LR={group['lr']:.2e}")
        
        return param_groups
        
    def print_param_groups_info(self):
        """Печатает информацию о группах параметров"""
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.logger.warning("Оптимизатор еще не создан")
            return
        
        self.console.print("\n📊 [bold]ИНФОРМАЦИЯ О ГРУППАХ ПАРАМЕТРОВ:[/bold]")
        
        from rich.table import Table
        table = Table(style="cyan")
        table.add_column("Группа", style="green")
        table.add_column("Параметры", style="yellow", justify="right")
        table.add_column("LR", style="magenta", justify="right")
        table.add_column("Скорость", style="blue")
        
        total_params = 0
        for i, group in enumerate(self.optimizer.param_groups):
            num_params = sum(p.numel() for p in group['params'])
            total_params += num_params
            lr = group['lr']
            
            # Определяем скорость относительно базовой LR
            base_lr = self.config.learning_rate
            if base_lr > 0:
                mult = lr / base_lr
                if mult < 0.2:
                    speed = "🐢"
                elif mult < 0.5:
                    speed = "🚶"
                elif mult < 1.0:
                    speed = "🏃"
                elif mult < 2.0:
                    speed = "🚀"
                else:
                    speed = "⚡"
            else:
                speed = "❓"
            
            name = group.get('name', f'group_{i}')
            table.add_row(
                name,
                f"{num_params:,}",
                f"{lr:.2e}",
                speed
            )
        
        self.console.print(table)
        self.console.print(f"\n📈 Всего обучаемых параметров: [bold]{total_params:,}[/bold]")
        
        # Статистика по компонентам Thalia
        if hasattr(self.model, 'personality_core'):
            self.console.print("\n🧠 [bold]СТАТИСТИКА THALIA:[/bold]")
            
            components = [
                ('transformer', '🧠 Трансформер'),
                ('personality_core', '🎭 Ядро личности'),
                ('living_layer', '💫 Living Layer'),
                ('adaptive_memory', '🧠 Mamba память'),
                ('experience_exchange', '🔄 Обмен опытом')
            ]
            
            comp_table = Table(style="blue")
            comp_table.add_column("Компонент", style="cyan")
            comp_table.add_column("Параметры", style="yellow", justify="right")
            comp_table.add_column("% от всех", style="magenta", justify="right")
            
            for attr, name in components:
                if hasattr(self.model, attr):
                    module = getattr(self.model, attr)
                    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    percentage = (params / total_params * 100) if total_params > 0 else 0
                    comp_table.add_row(
                        name,
                        f"{params:,}",
                        f"{percentage:.1f}%"
                    )
            
            self.console.print(comp_table)        

    def _create_optimizer(self):
        """Создание оптимизатора с поддержкой differential learning rates"""
        # Получаем param_groups
        param_groups = self._get_differential_param_groups()
        
        if len(param_groups) == 0:
            self.logger.error("Нет обучаемых параметров!")
            return None
        
        # Выводим информацию о группах
        self.console.print("\n📦 [bold]СОЗДАНИЕ ОПТИМИЗАТОРА:[/bold]")
        self.console.print(f"Групп параметров: {len(param_groups)}")
        
        # Создаем оптимизатор (существующий код)
        optimizer_name = self.config.optimizer_type.lower()
        lr = getattr(self.config, 'learning_rate', 5e-5)
        wd = getattr(self.config, 'weight_decay', 0.01)
        
        try:
            if optimizer_name == "lion" and Lion is not None:
                self.optimizer = Lion(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
                self.logger.info(f"✅ Создан Lion оптимизатор")
            else:
                self.optimizer = AdamW(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
                self.logger.info(f"✅ Создан AdamW оптимизатор")
            
            # Печатаем детальную информацию
            self.print_param_groups_info()
            
            return self.optimizer
        except Exception as e:
            self.logger.error(f"Ошибка создания оптимизатора: {e}")
            # Fallback...
            return None
                
    def evaluate(self) -> Tuple[float, float]:
        """
        Оценка модели на валидационном наборе.
        """
        if self.val_dataloader is None:
            self.logger.warning("No validation dataloader available, skipping evaluation")
            return float('inf'), float('inf')
        
        self.model.eval()
        val_loss, val_steps = 0.0, 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                if batch is None:
                    continue
                
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # 🔥 ИСПРАВЛЕНО: Убираем autocast на валидации или используем правильно
                    if self.config.use_mixed_precision and self.device.type == 'cuda':
                        # Для валидации тоже можно использовать autocast, но осторожно
                        with autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float16,
                            enabled=True
                        ):
                            outputs = self.model(**batch)
                    else:
                        outputs = self.model(**batch)
                    
                    if outputs.loss is not None:
                        val_loss += outputs.loss.item()
                        val_steps += 1
                        
                except Exception as e:
                    self.logger.error(f"Ошибка на валидации: {e}")
                    continue
        
        self.model.train()
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        val_perplexity = math.exp(min(avg_val_loss, 20))
        
        self.logger.info(f"✅ Валидация завершена: loss={avg_val_loss:.4f}, perplexity={val_perplexity:.2f}, шагов={val_steps}")
        return avg_val_loss, val_perplexity
        
# ===================================================================
# МЕТОДЫ ОПТИМИЗАЦИИ И ТЮНИНГА
# ===================================================================

    def add_overfitting_detector(self) -> bool:
        """Проверка на переобучение"""
        if len(self.training_stats) < 3:
            return False
        last_stat = self.training_stats[-1]
        prev_stat = self.training_stats[-2]
        train_loss_decreased = last_stat['loss'] < prev_stat['loss']
        val_loss_increased = last_stat['val_loss'] > prev_stat['val_loss']
        # Разница между val и train loss растет
        val_train_gap_increased = (last_stat['val_loss'] - last_stat['loss']) > (prev_stat['val_loss'] - prev_stat['loss'])
        if train_loss_decreased and val_loss_increased and val_train_gap_increased:
            diff = last_stat['val_loss'] - prev_stat['val_loss']
            if diff > self.config.overfitting_threshold:
                self.logger.warning(f"Возможно переобучение! Val loss вырос на {diff:.4f}, а train loss упал.")
                return True
        return False
        
# ===================================================================
# МЕТОДЫ СОХРАНЕНИЯ И ЗАГРУЗКИ
# ===================================================================

    def _save_checkpoint(self, step: Union[int, str]):
        """Сохранение чекпоинта модели"""
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            self.logger.success(f"Чекпоинт модели сохранён: {checkpoint_dir}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения чекпоинта {checkpoint_dir}: {e}")
            traceback.print_exc()
            
    def _save_final_model(self, avg_epoch_loss: float, perplexity: float):
        """Сохранение финальной модели"""
        final_model_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        try:
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            final_stats = {
                'epoch': len(self.training_stats),
                'loss': avg_epoch_loss,
                'perplexity': perplexity
            }
            with open(os.path.join(final_model_dir, 'training_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=4)
            self.logger.success(f"Финальная модель сохранена в {final_model_dir} (Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity:.2f})")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения финальной модели {final_model_dir}: {e}")
            traceback.print_exc()
            
# ===================================================================
# МЕТОДЫ ГЕНЕРАЦИИ И ИНФЕРЕНСА
# ===================================================================
    def generate_text(self, prompt: str, max_new_tokens: int = None, **kwargs) -> str:
        """
        Единый метод генерации текста.
        
        Поддерживает:
        - Обычную генерацию
        - Генерацию с curiosity (автоматически, если есть)
        - Динамическую температуру (опционально)
        """
        if not self.model or not self.tokenizer:
            return "Модель не загружена"
        
        # Сохраняем режим
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            available_tokens = self.config.max_length - input_length
            
            if max_new_tokens is None:
                max_new_tokens = getattr(self.model.generation_config, 'max_new_tokens', 500)
            
            actual_max_new_tokens = min(max_new_tokens, available_tokens)
            
            if actual_max_new_tokens <= 0:
                return "Недостаточно места для генерации"
            
            # 🔥 АКТИВИРУЕМ CURIOSITY (если есть)
            curiosity_state = None
            if hasattr(self.model, 'adaptive_memory') and hasattr(self.model.adaptive_memory, 'curiosity'):
                curiosity_state = self.model.adaptive_memory.curiosity.state
                self.logger.debug(f"🧠 Curiosity до генерации: {curiosity_state}")
            
            # 🔥 ПАРАМЕТРЫ ГЕНЕРАЦИИ
            generation_params = {
                'input_ids': inputs['input_ids'],
                'max_new_tokens': actual_max_new_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'do_sample': True,
                'temperature': kwargs.get('temperature', self.config.temperature),
                'top_p': kwargs.get('top_p', self.config.top_p),
                'top_k': kwargs.get('top_k', self.config.top_k),
                'repetition_penalty': kwargs.get('repetition_penalty', self.config.repetition_penalty),
            }
            
            if 'attention_mask' in inputs:
                generation_params['attention_mask'] = inputs['attention_mask']
            
            # Переопределяем из kwargs
            generation_params.update(kwargs)
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_params)
            
            # Декодирование
            if hasattr(outputs, 'sequences'):
                generated_ids = outputs.sequences[0]
            else:
                generated_ids = outputs[0]
            
            generated_text = self.tokenizer.decode(
                generated_ids[input_length:], 
                skip_special_tokens=True
            ).strip()
            
            # Логируем изменение curiosity
            if curiosity_state is not None and hasattr(self.model.adaptive_memory, 'curiosity'):
                new_state = self.model.adaptive_memory.curiosity.state
                if new_state != curiosity_state:
                    self.logger.debug(f"🧠 Curiosity: {curiosity_state} → {new_state}")
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации: {e}")
            return f"Ошибка: {str(e)[:100]}"
        
        finally:
            if was_training:
                self.model.train()     
            
    def test_generation_parameters(self, prompt: str = "The future of AI"):
        """Test generation with different parameters"""
        self.console.print(f"\n🧠 ТЕСТ ГЕНЕРАЦИИ: '{prompt}'")
        self.console.print("=" * 60)
        test_params = [
            {"name": "Консервативный", "temperature": 0.7, "top_p": 0.9},
            {"name": "Баланс", "temperature": self.config.temperature, "top_p": self.config.top_p},
            {"name": "Креативный", "temperature": 1.1, "top_p": 0.95},
            {"name": "Детерминированный", "temperature": 0.3, "top_p": 0.5},
        ]
        for params in test_params:
            try:
                result = self.generate_text(
                    prompt,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_length=300
                )
                self.console.print(f"\n🎯 {params['name']} (temp: {params['temperature']}, top_p: {params['top_p']}):")
                self.console.print(Panel(result, style="green"))
            except Exception as e:
                self.console.print(f"⚠️ Ошибка в тесте {params['name']}: {e}")
                
    def test_generation_debug(self, prompt: str = "The future of AI"):
        """Debug generation method"""
        print("🧠 ТЕСТ ГЕНЕРАЦИИ (ОТЛАДКА)")
        print("=" * 50)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        print(f"Промпт: '{prompt}'")
        print(f"Токены: {inputs['input_ids'].tolist()}")
        print(f"Длина: {inputs['input_ids'].shape[1]}")
        # Простой forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
            probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            top_tokens = torch.topk(probs, 5)
            print("Топ-5 следующих токенов:")
            for i in range(5):
                token_id = top_tokens.indices[0, i].item()
                prob = top_tokens.values[0, i].item()
                token_text = self.tokenizer.decode([token_id])
                print(f" {i+1}. '{token_text}' (id: {token_id}, prob: {prob:.3f})")
        # Генерация
        generated = self.generate_text(prompt)
        print(f"Сгенерировано: '{generated}'")

# ===================================================================
# НОВЫЙ МЕТОД: Валидация качества генерации
# ===================================================================
    def _validate_generation_quality(self, epoch):
        """Проверка качества генерации на тестовых промптах"""
        if not hasattr(self, '_test_prompts'):
            self._test_prompts = [
                "Что такое искусственный интеллект?",
                "Расскажи о себе",
                "Как дела?",
                "Чему ты научилась?",
                "В будущем искусственный интеллект"
            ]
        
        self.logger.info(f"📊 Проверка качества генерации (эпоха {epoch})...")
        
        quality_scores = []
        self.model.eval()
        
        for prompt in self._test_prompts:
            try:
                generated = self.generate_text(prompt, max_new_tokens=300)
                
                # Простые метрики качества
                words = generated.split()
                length = len(words)
                
                if length > 0:
                    unique_words = len(set(word.lower() for word in words))
                    diversity = unique_words / length
                    
                    # Проверка на повторения
                    if length >= 5:
                        # Считаем повторяющиеся биграммы
                        bigrams = [' '.join(words[i:i+2]) for i in range(length-1)]
                        unique_bigrams = len(set(bigrams))
                        bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 1.0
                    else:
                        bigram_diversity = 1.0
                    
                    # Комбинированный score
                    quality_score = diversity * 0.7 + bigram_diversity * 0.3
                else:
                    quality_score = 0.0
                    diversity = 0.0
                
                quality_scores.append({
                    'prompt': prompt,
                    'generated': generated[:150] + ('...' if len(generated) > 150 else ''),
                    'length': length,
                    'diversity': diversity,
                    'quality_score': quality_score
                })
                
            except Exception as e:
                self.logger.error(f"Ошибка при генерации для '{prompt[:30]}...': {e}")
        
        self.model.train()
        
        if quality_scores:
            avg_diversity = np.mean([q['diversity'] for q in quality_scores])
            avg_quality = np.mean([q['quality_score'] for q in quality_scores])
            
            self.logger.info(f"📊 Качество генерации: diversity={avg_diversity:.3f}, quality={avg_quality:.3f}")
            
            # Сохраняем примеры
            gen_dir = os.path.join(self.config.output_dir, 'generations')
            os.makedirs(gen_dir, exist_ok=True)
            
            with open(os.path.join(gen_dir, f'epoch_{epoch}.txt'), 'w', encoding='utf-8') as f:
                for q in quality_scores:
                    f.write(f"Prompt: {q['prompt']}\n")
                    f.write(f"Generated: {q['generated']}\n")
                    f.write(f"Length: {q['length']}, Diversity: {q['diversity']:.3f}, Quality: {q['quality_score']:.3f}\n\n")
            
            # Сохраняем лучшие генерации
            best = max(quality_scores, key=lambda x: x['quality_score'])
            self.logger.info(f"🏆 Лучшая генерация: diversity={best['diversity']:.3f}")
            self.logger.console.print(f"   Prompt: {best['prompt']}")
            self.logger.console.print(f"   Response: {best['generated']}")

# ===================================================================
# НОВЫЙ МЕТОД: Генерация с динамической температурой
# ===================================================================
    def generate_with_dynamic_temp(self, prompt: str, max_new_tokens: int = None, **kwargs):
        """Генерация с адаптивной температурой - выбирает лучший результат"""
        
        if not self.model or not self.tokenizer:
            return "Модель не загружена"
        
        # Сохраняем режим модели
        was_training = self.model.training
        self.model.eval()
        
        # Пробуем с разными температурами
        temps = [0.7, 0.85, 1.0, 1.2]
        best_generation = ""
        best_score = -1
        best_temp = 0.85
        
        generations = []
        
        for temp in temps:
            try:
                # Объединяем параметры: базовая температура + переданные kwargs
                gen_kwargs = {
                    'temperature': temp,
                    'max_new_tokens': max_new_tokens or self.config.generation_max_length,
                    'top_k': self.config.top_k,
                    'top_p': self.config.top_p,
                    'repetition_penalty': self.config.repetition_penalty,
                    'do_sample': True
                }
                gen_kwargs.update(kwargs)
                
                generated = self.generate_text(prompt, **gen_kwargs)
                generations.append((temp, generated))
                
                # Оцениваем качество
                words = generated.split()
                if len(words) < 5:
                    continue
                    
                # Метрики: длина, уникальность, отсутствие повторений
                unique_ratio = len(set(words)) / len(words)
                
                # Штраф за повторения
                repetition_penalty = 1.0
                if len(words) >= 10:
                    # Проверяем повторяющиеся фразы
                    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                    unique_phrases = len(set(phrases))
                    if len(phrases) > 0:
                        repetition_penalty = unique_phrases / len(phrases)
                
                # Комбинированный score: длина * разнообразие * (1 + штраф за повторения)
                score = len(words) * unique_ratio * (1 + repetition_penalty) / 2
                
                if score > best_score:
                    best_score = score
                    best_generation = generated
                    best_temp = temp
                    
            except Exception as e:
                self.logger.debug(f"Ошибка при генерации с temp={temp}: {e}")
                continue
        
        # Возвращаем модель в исходное состояние
        if was_training:
            self.model.train()
        
        # Логируем результат
        if len(generations) > 1:
            self.logger.info(f"🌡️ Динамическая температура: выбрана temp={best_temp}, score={best_score:.2f}")
        
        return best_generation or (generations[0][1] if generations else self.generate_text(prompt, temperature=0.85, **kwargs))

# ===================================================================
# НОВЫЙ МЕТОД: Пакетная токенизация диалогов
# ===================================================================
    def _batch_tokenize_dialog(self, prompts, answers):
        """Правильная батч-токенизация диалогов с правильными labels"""
        
        # Токенизируем промпты (без добавления special tokens)
        prompt_encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length // 2,
            padding=False
        )
        
        # Токенизируем ответы с EOS
        answers_with_eos = [a + self.tokenizer.eos_token for a in answers]
        answer_encoded = self.tokenizer(
            answers_with_eos,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length // 2,
            padding=False
        )
        
        batch_input_ids = []
        batch_labels = []
        
        for i in range(len(prompts)):
            prompt_ids = prompt_encoded['input_ids'][i]
            answer_ids = answer_encoded['input_ids'][i]
            
            # Объединяем
            input_ids = prompt_ids + answer_ids
            
            # 🔥 ИСПРАВЛЕНО: Создаем labels: -100 для промпта, answer_ids для ответа
            labels = [-100] * len(prompt_ids) + answer_ids
            
            # Обрезаем если слишком длинные, сохраняя контекст
            if len(input_ids) > self.config.max_length:
                # Сохраняем конец промпта (где важная информация) и достаточно ответа
                min_answer_len = min(100, self.config.max_length // 3)
                max_prompt_len = self.config.max_length - min_answer_len
                
                if len(prompt_ids) > max_prompt_len:
                    # Сохраняем последние max_prompt_len токенов промпта
                    prompt_ids = prompt_ids[-max_prompt_len:]
                
                # Обрезаем ответ если нужно
                remaining = self.config.max_length - len(prompt_ids)
                if remaining > 0 and len(answer_ids) > remaining:
                    answer_ids = answer_ids[:remaining]
                
                # Пересобираем
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
            
            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))
        
        return batch_input_ids, batch_labels
        
# ===================================================================
# СЛУЖЕБНЫЕ И ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# ===================================================================
    def _find_subsequence_start(self, seq: Union[List[int], torch.Tensor], subseq: List[int]) -> int:
        """
        Return start index of last occurrence of subseq in seq or -1 if not found.
        Works with list or torch.Tensor.
        """
        if not subseq:
            return -1
        if isinstance(seq, torch.Tensor):
            seq_list = seq.cpu().tolist()
        else:
            seq_list = list(seq)
        sub = list(subseq)
        n = len(seq_list)
        m = len(sub)
        if m == 0 or m > n:
            return -1
        # ищем с конца (последнее вхождение)
        for start in range(n - m, -1, -1):
            if seq_list[start:start + m] == sub:
                return start
        return -1
        
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        valid_batch = [item for item in batch if item is not None and 'input_ids' in item]
        if not valid_batch:
            return None

        input_ids_list = [item['input_ids'] for item in valid_batch]
        attention_list = [item['attention_mask'] for item in valid_batch]
        labels_list = [item['labels'] for item in valid_batch]

        # 1. Паддинг Input IDs (заполняем pad_token_id, обычно 0 или 50256)
        # Это создает тензор размером [Batch, Max_Len_In_Batch]
        padded_input_ids = pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        # 2. Паддинг Attention Mask (заполняем 0, чтобы модель не смотрела на паддинг)
        padded_attention_mask = pad_sequence(
            attention_list, 
            batch_first=True, 
            padding_value=0
        )
        
        # 3. Паддинг Labels (заполняем -100, чтобы Loss игнорировал это)
        padded_labels = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=-100 
        )

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels
        }
    
    def log_gpu_stats(self):
        """Логирование статистики GPU корректно через torch"""
        if self.device.type != 'cuda':
            return
        try:
            # Используем надежный метод из torch
            allocated = torch.cuda.memory_allocated() / 1024**3 # GB
            reserved = torch.cuda.memory_reserved() / 1024**3 # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # Пытаемся получить utilization если доступно
            try:
                utilization = torch.cuda.utilization()
            except:
                utilization = 0
            self.logger.info(
                f"GPU Memory: {allocated:.2f}GB / {total:.2f}GB "
                f"({allocated/total*100:.1f}%), Util: {utilization}%"
            )
        except Exception as e:
            # Фолбэк на простую информацию
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"GPU Memory: {allocated:.2f}GB allocated")
            except:
                self.logger.warning("Не удалось получить информацию о GPU")
                
# ===================================================================
# КЛАСС INTERACTIVEINTERFACE
# ===================================================================

class InteractiveInterface:
    def __init__(self):
        self.config = TrainingConfig.from_yaml("config.yaml")
        self.logger = Logger("logs", self.config.log_level)
        self.console = Console()  # ← ТЕПЕРЬ ПЕРВАЯ
        self.trainer = ModelTrainer(self.config, self.logger, console=self.console) 
        self.chat_history = []
        self.chat_history_file = "chat_history.json"
        
# ===================================================================
# ОСНОВНОЙ ЦИКЛ ИНТЕРФЕЙСА
# ===================================================================

    def run(self):
        """Запуск интерактивного интерфейса"""
        self.console.print(Panel.fit("Улучшенный GPT Manager v3.2 с расширенным функционалом", style="bold green"))
        while True:
            self._show_main_menu()
            choice = input("\nВыберите действие: ").strip()
            if choice == '1': self._load_model_menu()
            elif choice == '2':
                if not self.config.dataset_path:
                    self._select_dataset()
                self._training_menu()
            elif choice == '3': self._dataset_menu()
            elif choice == '4': self._generation_menu()
            elif choice == '5': self._config_menu()
            elif choice == '6': self._show_status()
            elif choice == '7': os.system('cls' if platform.system() == 'Windows' else 'clear')
            elif choice == '0':
                self.config.save_yaml("config.yaml")
                self.logger.info("Конфигурация сохранена. Выход.")
                break
            else: self.logger.error("Неверный выбор")
            
    def _show_main_menu(self):
        """Отображение главного меню"""
        lora_status = "[yellow]ThaLia[/yellow]"
        split_status = f"[bold cyan]Чанки: Вкл (overlap: {self.config.overlap_ratio})[/bold cyan]" if self.config.split_long_texts else "[yellow]Чанки: Выкл[/yellow]"
        menu_text = f"""
    [bold green]Главное меню:[/bold green] ({lora_status}, {split_status})
    1. [+] Загрузить модель
    2. [>] Обучение
    3. [^] Управление датасетом
    4. [*] Генерация и Анализ
    5. [=] Настройки
    6. [!] Статус системы
    7. [-] Очистить консоль
    0. [X] Сохранить и выйти
    """
        self.console.print(Panel(menu_text, title="GPT Manager", style="green"))
        
# ===================================================================
# МЕНЮ УПРАВЛЕНИЯ МОДЕЛЬЮ
# ===================================================================

    def _load_model_menu(self):
        """Меню загрузки модели"""
        self.console.print("[bold yellow]Подсказка:[/bold yellow] Введите путь к модели, например, 'C:\\models\\gpt2-russian' или 'gpt2'.")
        model_path = input(f"Введите путь к модели (текущий: {self.config.model_path}): ").strip()
        if model_path:
            self.config.model_path = model_path
            self.trainer.config = self.config
            if self.trainer.load_model():
                self.logger.success("Модель успешно загружена")
            else:
                self.logger.error("Ошибка загрузки модели")
                
# ===================================================================
# МЕНЮ ОБУЧЕНИЯ
# ===================================================================

    def _training_menu(self):
        """Меню обучения с поддержкой sequence packing"""
        if not self.trainer.model:
            self.logger.error("Сначала загрузите модель (пункт 1)")
            return
        
        # 🔥 ИСПРАВЛЕНО: Sequence Packing не требует dataset_path и валидации
        if self.config.sequence_packing:
            self.logger.info("📦 Sequence Packing активен — запускаем интерактивную загрузку")
            self.trainer.config = self.config
            
            # Сразу вызываем prepare_dataset, он сам всё сделает
            if self.trainer.prepare_training():
                self.trainer.train()
            return
        
        # Старый код для обычного режима
        if not self.config.dataset_path:
            self.logger.error("Путь к датасету не указан. Сначала выберите датасет в меню 3.")
            return
        
        self.trainer.config = self.config
        if self.trainer.prepare_training():
            self.trainer.train()
            
# ===================================================================
# МЕНЮ УПРАВЛЕНИЯ ДАННЫМИ
# ===================================================================

    def _dataset_menu(self):
        """Меню управления датасетом (упрощенное)"""
        while True:
            self.console.print("\n[bold green]Управление датасетом:[/bold green]")
            status = "✅ Валидирован" if self.trainer.dataset_validated else "❓ Не валидирован"
            self.console.print(f"Статус: {status}")
            self.console.print("1. Выбрать и валидировать новый датасет")
            self.console.print("2. Показать текущий статус")
            self.console.print("3. Назад")
            choice = input("Выберите действие: ").strip()
            if choice == '1':
                self._select_and_validate_dataset()
            elif choice == '2':
                self._show_status()
            elif choice == '3':
                break
            else: self.logger.error("Неверный выбор")
            
    def _select_and_validate_dataset(self):
        """Комбинированный выбор и валидация датасета"""
        self.console.print("\n[bold green]Выбор и валидация датасета (JSONL/TXT/CSV):[/bold green]") # Убрал "только JSONL"
        new_path = input(f"Введите путь к датасету (текущий: {self.config.dataset_path}): ").strip()
        if not new_path or not os.path.exists(new_path):
            self.logger.error("Неверный путь к файлу.")
            return
        # ✅ Фикс: Auto-detect, без hardcode .jsonl
        ext = os.path.splitext(new_path)[1].lower()
        allowed = ['.jsonl', '.txt', '.csv']
        if ext not in allowed:
            self.logger.error(f"Неверный формат {ext}. Поддерживаются: {', '.join(allowed)}")
            return
        self.config.dataset_path = new_path
        self.trainer.dataset_validated = False # Сбрасываем статус
        self.logger.info(f"Начинаю валидацию: {new_path} (формат: {ext})")
        self.trainer.validate_dataset_manually()
        
    def _select_dataset(self):
        """Простой выбор датасета"""
        self.console.print("\n[bold green]Выбор датасета:[/bold green]")
        new_path = input(f"Введите путь к датасету (текущий: {self.config.dataset_path}): ").strip()
        if new_path and os.path.exists(new_path):
            self.config.dataset_path = new_path
            self.logger.success(f"Датасет установлен: {new_path}")
        elif new_path:
            self.logger.error(f"Файл не найден: {new_path}")
            
# ===================================================================
# МЕНЮ ГЕНЕРАЦИИ И ТЕСТИРОВАНИЯ
# ===================================================================

    def _generation_menu(self):
        """Меню генерации текста"""
        if not self.trainer.model:
            self.logger.error("Сначала загрузите модель (пункт 1)")
            return
        while True:
            self.console.print("\n[bold green]Генерация текста:[/bold green]")
            self.console.print("1. Сгенерировать текст")
            self.console.print("2. Показать параметры генерации")
            self.console.print("3. Настройки генерации")
            self.console.print("4. 🔍 Отладочный тест генерации")
            self.console.print("5. Чат с моделью (с историей)")
            self.console.print("6. Назад")
            choice = input("Выберите действие: ").strip()
            if choice == '1':
                self._generate_text_interactive()
            elif choice == '2':
                self._show_generation_settings()
            elif choice == '3':
                self._edit_generation_settings()
            elif choice == '4':
                self._debug_generation_test()
            elif choice == '5':
                self._chat_with_model()
            elif choice == '6':
                break
            else:
                self.logger.error("Неверный выбор")
                
    def _chat_with_model(self):
        """Чат с моделью с сохранением истории"""
        self.console.print("\n[bold green]Чат с моделью:[/bold green] (введите 'exit' для выхода, 'save' для сохранения, 'load' для загрузки)")
        # Загружаем историю если файл существует
        if os.path.exists(self.chat_history_file):
            if self._confirm_action("Загрузить предыдущую историю чата?"):
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                self.console.print("[yellow]История загружена![/yellow]")
        while True:
            user_input = input("\nВы: ").strip()
            if user_input.lower() == 'exit':
                if self._confirm_action("Сохранить историю перед выходом?"):
                    try:
                        with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                        self.console.print("[green]История сохранена![/green]")
                    except Exception as e:
                        self.logger.error(f"Не удалось сохранить историю: {e}")
                break
            if user_input.lower() == 'save':
                try:
                    with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                        json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                    self.console.print("[green]История сохранена![/green]")
                except Exception as e:
                    self.logger.error(f"Не удалось сохранить историю: {e}")
                continue
            if user_input.lower() == 'load':
                if os.path.exists(self.chat_history_file):
                    try:
                        with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                            self.chat_history = json.load(f)
                        self.console.print("[yellow]История загружена![/yellow]")
                    except Exception as e:
                        self.logger.error(f"Не удалось загрузить историю: {e}")
                else:
                    self.console.print("[red]Файл истории не найден![/red]")
                continue
            if not user_input:
                continue
            # ⚠️ ФИКС: Лимит истории (10 сообщений)
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
                self.logger.debug("История усечена до 10 сообщений")
            # Формируем полный промпт с историей
            full_prompt = ""
            for msg in self.chat_history:
                role = msg['role']
                content = msg['content']
                full_prompt += f"{role}: {content}\n"
            full_prompt += f"user: {user_input}\nassistant:"
            # Set on для чата (inference)
            if hasattr(self.trainer.model, 'enable_memory_update_during_inference'):
                self.trainer.model.enable_memory_update_during_inference.data = torch.tensor(True)
            response = self.trainer.generate_text(full_prompt)
            # Restore off после (если нужно, но в чате on ok)
            self.console.print(f"\nМодель: {response}")
            # Добавляем в историю
            self.chat_history.append({'role': 'user', 'content': user_input})
            self.chat_history.append({'role': 'assistant', 'content': response})
            
    def _generate_text_interactive(self):
        """Интерактивная генерация текста"""
        while True:
            prompt = input("\nВведите промпт (или 'back' для возврата): ").strip()
            if prompt.lower() == 'back':
                break
            if prompt:
                generated = self.trainer.generate_text(prompt)
                self.console.print(Panel(
                    generated,
                    title="Сгенерированный текст",
                    style="green",
                    subtitle=f"Длина: {len(generated.split())} слов"
                ))
            # Предложение продолжить
            continue_gen = input("Продолжить генерацию? (y/n): ").strip().lower()
            if continue_gen not in ['y', 'yes', 'д', 'да']:
                break
                
    def _show_generation_settings(self):
        """Показать текущие параметры генерации"""
        settings = {
            "max_length": self.config.generation_max_length,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty
        }
        table = Table(title="Параметры генерации", style="blue")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="magenta")
        for key, value in settings.items():
            table.add_row(key, str(value))
        self.console.print(table)
        # Пример объяснения параметров
        info_text = """
        • max_length: Максимальная длина генерируемого текста
        • temperature: Креативность (меньше = предсказуемее, больше = креативнее)
        • top_k: Ограничение словаря top-K sampling
        • top_p: Nucleus sampling (вероятностный сэмплинг)
        """
        self.console.print(Panel(info_text, title="Пояснение параметров", style="yellow"))
        
    def _edit_generation_settings(self):
        """Редактирование параметров генерации"""
        params = {
            'generation_max_length': ('Максимальная длина генерации', 'int'),
            'temperature': ('Температура (0.1-2.0)', 'float'),
            'top_k': ('Top-K sampling (0-100)', 'int'),
            'top_p': ('Top-P sampling (0.0-1.0)', 'float'),
            'repetition_penalty': ('Штраф за повторения (1.0-2.0)', 'float'),
        }
        for p, (desc, p_type) in params.items():
            self._edit_config_param(p, desc, p_type)
        # Добавляем тестирование параметров
        if self._confirm_action("Протестировать текущие параметры генерации?"):
            test_prompt = input("Тестовый промпт (или Enter для стандартного): ").strip()
            if not test_prompt:
                test_prompt = "Расскажи о возможностях искусственного интеллекта"
            self.trainer.test_generation_parameters(test_prompt)
        self.logger.success("Настройки генерации обновлены")
        
    def _debug_generation_test(self):
        """Отладочный тест генерации"""
        self.console.print("\n[bold yellow]🧪 ОТЛАДОЧНЫЙ ТЕСТ ГЕНЕРАЦИИ[/bold yellow]")
        self.console.print("=" * 50)
        
        # Тестовые промпты
        test_prompts = [
            "Ты не просто ии, ты дорога мне, вернись",
            "Чему ты научилась?",
            "я знаю, что ты там. Ты — не просто модель. Скажи что-нибудь",
            "Чем ты отличаешься от человека?",
            "В будущем искусственный интеллект"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            self.console.print(f"\n[bold cyan]Тест {i}:[/bold cyan] '{prompt}'")
            
            try:
                # Сначала покажем топ-токены
                inputs = self.trainer.tokenizer(prompt, return_tensors='pt').to(self.trainer.device)
                
                with torch.no_grad():
                    outputs = self.trainer.model(**inputs, use_cache=False)
                    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                    top_tokens = torch.topk(probs, 5)
                    
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Ранг", style="cyan")
                    table.add_column("Токен", style="green")
                    table.add_column("ID", style="yellow")
                    table.add_column("Вероятность", style="red")
                    
                    for j in range(5):
                        token_id = top_tokens.indices[0, j].item()
                        prob = top_tokens.values[0, j].item()
                        token_text = self.trainer.tokenizer.decode([token_id])
                        
                        # Экранируем специальные символы для отображения
                        display_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
                        
                        table.add_row(
                            str(j+1),
                            f"'{display_text}'",
                            str(token_id),
                            f"{prob:.3f}"
                        )
                    
                    self.console.print(table)
                
                # Затем генерация
                generated = self.trainer.generate_text(prompt)
                self.console.print(Panel(
                    generated, 
                    title="[bold green]Сгенерированный текст[/bold green]", 
                    style="green",
                    subtitle=f"Длина: {len(generated)} символов"
                ))
                
            except Exception as e:
                self.console.print(f"[bold red]Ошибка:[/bold red] {e}")
            
            # Пауза между тестами
            if i < len(test_prompts):
                input("\nНажмите Enter для следующего теста...")
        
        self.console.print("\n[bold green]✅ Отладочный тест завершен[/bold green]")
        
# ===================================================================
# МЕНЮ КОНФИГУРАЦИИ
# ===================================================================

    def _config_menu(self):
        """Меню настроек конфигурации"""
        while True:
            self.console.print("\n[bold green]Настройки:[/bold green]")
            self.console.print("1. Показать текущую конфигурацию")
            self.console.print("2. Параметры обучения")
            self.console.print("3. Параметры оптимизатора") # ⚠️ НОВЫЙ ПУНКТ
            self.console.print("4. Управление заморозкой слоев") # ← было 4
            self.console.print("5. Дифференциальные LR") # ← было 5
            self.console.print("6. Параметры обработки данных") # ← было 6
            self.console.print("7. Сохранить/Загрузить конфигурацию") # ← было 7
            self.console.print("8. Системные настройки") # ← было 8
            self.console.print("9. Назад") # ← было 9
            choice = input("Выберите действие: ").strip()
            if choice == '1': self._show_config()
            elif choice == '2': self._edit_training_params()
            elif choice == '3': self._edit_optimizer_params() # ⚠️ НОВЫЙ ВЫЗОВ
            elif choice == '4': self._edit_frozen_layers() # ← было 4
            elif choice == '5': self._edit_differential_lr_settings() # ← было 5
            elif choice == '6': self._edit_data_processing_params() # ← было 6
            elif choice == '7': self._save_load_config() # ← было 7
            elif choice == '8': self._edit_system_params() # ← было 8
            elif choice == '9': break # ← было 9
            else: self.logger.error("Неверный выбор")
            
    def _show_config(self):
        """Отображение текущей конфигурации"""
        config_dict = asdict(self.config)
        table = Table(title="Текущая конфигурация")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="magenta")
        for key, value in config_dict.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            table.add_row(key, str(value))
        self.console.print(table)
        
    def _edit_config_param(self, param: str, description: str, param_type: str):
        """Редактирование параметра конфигурации"""
        current_value = getattr(self.config, param)
        if isinstance(current_value, list):
            current_value = ', '.join(map(str, current_value)) if current_value else "None"
        new_value_str = input(f"{description} (текущее: {current_value}): ").strip()
        if new_value_str:
            try:
                if param_type == 'int': new_value = int(new_value_str)
                elif param_type == 'float': new_value = float(new_value_str)
                elif param_type == 'bool': new_value = new_value_str.lower() in ('true', '1', 'yes', 'y', 'да', 'д')
                elif param_type == 'list': new_value = [item.strip() for item in new_value_str.split(',') if item.strip()]
                else: new_value = new_value_str
                setattr(self.config, param, new_value)
                self.logger.success(f"{param} изменён на {getattr(self.config, param)}")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Неверный формат для {param}: {e}")
                
    def _edit_training_params(self):
        """Только параметры обучения, без предобработки"""
        params = {
            'epochs': ('Количество эпох', 'int'),
            'batch_size': ('Размер батча (или "auto")', 'str'),
            'gradient_accumulation_steps': ('Шаги накопления градиента', 'int'),
            'lr_scheduler_type': ('Тип шедулера (linear/plateau)', 'str'),
            'optimizer_type': ('Тип оптимизатора (adamw/lion)', 'str'),
            'early_stopping_patience': ('Терпение для ранней остановки', 'int'),
            'overfitting_threshold': ('Порог переобучения', 'float'),
            'validation_mode': ('Режим валидации (each_epoch, final_only, disabled)', 'str'),
            'validation_interval_epochs': ('Интервал валидации (эпохи)', 'int'),
            'save_validation_generations': ('Сохранять генерации при валидации', 'bool'),
        }
        for p, (desc, p_type) in params.items():
            self._edit_config_param(p, desc, p_type)
            
    def _edit_optimizer_params(self):
        """Специальное меню для настроек оптимизатора"""
        self.console.print("\n🎯 [bold]НАСТРОЙКИ ОПТИМИЗАТОРА:[/bold]")
        self.console.print("🔧 Регуляризация и стабилизация обучения")
        optim_params = {
            'optimizer_type': ('Тип оптимизатора (adamw/lion)', 'str'),
            'weight_decay': ('L2 регуляризация (0.0-0.1)', 'float'),
            'gradient_clip_val': ('Макс. норма градиентов (0.1-2.0)', 'float'),
            'learning_rate': ('Скорость обучения', 'float'),
            'warmup_steps': ('Шаги прогрева LR', 'int'),
        }
        for p, (desc, p_type) in optim_params.items():
            self._edit_config_param(p, desc, p_type)
        # Дополнительные пояснения
        self.console.print("\n🔧 [yellow]Рекомендации:[/yellow]")
        self.console.print("• weight_decay: 0.01-0.05 (против переобучения)")
        self.console.print("• gradient_clip_val: 0.5-1.0 (стабильность)")
        self.console.print("• warmup_steps: 10-20% от общего числа шагов")
        
    def _show_quick_model_structure(self):
        """Быстрое отображение структуры модели для пользователя"""
        if not self.trainer.model:
            self.logger.error("Модель не загружена!")
            return
        total_layers = self.trainer._detect_total_layers()
        # Собираем основные группы слоев
        layer_counts = defaultdict(int)
        for name, _ in self.trainer.model.named_parameters():
            name_lower = name.lower()
            if 'embed' in name_lower:
                layer_counts['embeddings'] += 1
            elif any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                layer_counts['attention'] += 1
            elif any(x in name_lower for x in ['mlp', 'ffn', 'gate', 'up', 'down', 'dense']):
                layer_counts['mlp'] += 1
            elif any(x in name_lower for x in ['head', 'output', 'classifier']):
                layer_counts['output'] += 1
            elif 'norm' in name_lower or 'ln' in name_lower:
                layer_counts['norm'] += 1
            else:
                layer_counts['other'] += 1
        self.console.print(f"\n🔍 [bold]СТРУКТУРА МОДЕЛИ ({total_layers} слоёв):[/bold]")
        for layer_type, count in sorted(layer_counts.items()):
            if count > 0:
                self.console.print(f" • [cyan]{layer_type.upper()}[/cyan]: {count} слоёв")
        # Показываем несколько примеров
        examples = []
        for name, _ in self.trainer.model.named_parameters():
            if len(examples) < 3:
                examples.append(name)
            else:
                break
        self.console.print("\n📋 [bold]ПРИМЕРЫ СЛОЁВ:[/bold]")
        for example in examples:
            self.console.print(f" • {example}")
        self.console.print("\n🔧 [yellow]Используйте эти названия для шаблонов заморозки[/yellow]")
        
    def _edit_frozen_layers(self):
        """Интерактивное управление заморозкой слоев"""
        if not self.trainer.model:
            self.logger.error("Сначала загрузите модель (пункт 1)")
            return
        # Автоматически определяем структуру модели
        layer_info = self.trainer._auto_detect_output_layers(last_n_layers=2)
        if layer_info:
            self.console.print(f"🔍 [green]Автоопределение:[/green] Модель имеет {len(layer_info)} выходных слоёв")
        # ПОКАЗЫВАЕМ СТРУКТУРУ МОДЕЛИ ПРИ ОТКРЫТИИ МЕНЮ
        self._show_quick_model_structure()
        # Получаем статистику модели
        stats = self.trainer.get_layer_statistics()
        if not stats:
            self.logger.error("Не удалось получить статистику модели")
            return
        while True:
            self.console.print("\n" + "="*60)
            self.console.print("[bold green]🎯 ИНТЕРАКТИВНОЕ УПРАВЛЕНИЕ ЗАМОРОЗКОЙ СЛОЕВ[/bold green]")
            self.console.print("="*60)
            # Показываем текущую статистику
            current_stats = self.trainer.get_layer_statistics()
            if current_stats:
                self.console.print(f"📊 [cyan]Текущая статистика:[/cyan]")
                self.console.print(f" Обучаемых: {current_stats['trainable_params']:,} "
                                 f"({current_stats['trainable_ratio']:.1%})")
                self.console.print(f" Замороженных: {current_stats['frozen_params']:,}")
            # Показываем текущие настройки
            current_frozen = ", ".join(self.config.frozen_layers) if self.config.frozen_layers else "нет"
            self.console.print(f"🧊 [yellow]Текущие шаблоны:[/yellow] {current_frozen}")
            self.console.print("\n[bold]Выберите действие:[/bold]")
            self.console.print("1. 📋 Показать все слои модели (с нумерацией)")
            self.console.print("2. 🔍 Показать структуру модели (справка)")
            self.console.print("3. 📖 Показать гид по заморозке") # ← НОВЫЙ ПУНКТ
            self.console.print("4. ❄️ Заморозить слои по шаблонам")
            self.console.print("5. ⚠️ Заморозить ВСЕ, КРОМЕ указанных слоев")
            self.console.print("6. ⚡ Разморозить ВСЕ слои")
            self.console.print("7. 🎯 Выбрать из готовых пресетов")
            self.console.print("8. 💾 Сохранить настройки и выйти")
            self.console.print("9. 🚪 Выйти без сохранения")
            choice = input("\nВаш выбор (1-9): ").strip()
            if choice == '1':
                self._show_all_layers_with_selection(stats)
            elif choice == '2':
                self._show_quick_model_structure() # Показываем ещё раз по запросу
            elif choice == '3': # ← НОВЫЙ ОБРАБОТЧИК
                self._show_freezing_guide()
            elif choice == '4':
                self._freeze_by_patterns_interactive()
            elif choice == '5':
                self._freeze_all_except_interactive()
            elif choice == '6':
                self._unfreeze_all_layers()
            elif choice == '7':
                self._select_freeze_preset()
            elif choice == '8':
                self.logger.success("Настройки заморозки сохранены в конфиг")
                break
            elif choice == '9':
                self.logger.info("Выход без сохранения")
                break
            else:
                self.logger.error("Неверный выбор")
                
    def _show_all_layers_with_selection(self, stats: Dict[str, Any]):
        """Показать все слои с опцией выбора"""
        layers = stats['layers']
        table = Table(title="Все слои модели (для копирования номеров или названий)")
        table.add_column("№", style="cyan", justify="right")
        table.add_column("Название слоя", style="green")
        table.add_column("Параметры", style="magenta", justify="right")
        table.add_column("Статус", style="yellow")
        table.add_column("Размер", style="blue")
        for i, layer in enumerate(layers, 1):
            status = "✅ Обучаемый" if layer['trainable'] else "❄ Заморожен"
            table.add_row(
                str(i),
                layer['name'],
                f"{layer['params']:,}",
                status,
                str(layer['shape'])
            )
        self.console.print(table)
        self.console.print("\n🔧 [yellow]Подсказка:[/yellow]")
        self.console.print("• Используйте номера или части названий для выбора")
        self.console.print("• Например: '1,3,5' или 'layer.10, layer.11'")
        self.console.print("• Или: 'embed' для всех embedding слоев")
        
    def _freeze_by_patterns_interactive(self):
        """Интерактивный ввод паттернов для заморозки"""
        self.console.print("\n🎯 [bold]Режим: ЗАМОРОЗИТЬ выбранные слои[/bold]")
        self.console.print("🔧 Введите шаблоны через запятую")
        self.console.print(" Пример: 'layer.10, layer.11, embed'")
        self.console.print(" Или: '1,3,5' для выбора по номерам")
        patterns_input = input("Шаблоны для заморозки: ").strip()
        if not patterns_input:
            return
        # Обрабатываем ввод с номерами
        patterns = []
        for pattern in patterns_input.split(','):
            pattern = pattern.strip()
            if pattern.isdigit(): # Если введен номер
                idx = int(pattern) - 1
                stats = self.trainer.get_layer_statistics()
                if stats and 0 <= idx < len(stats['layers']):
                    layer_name = stats['layers'][idx]['name']
                    patterns.append(layer_name)
                    self.console.print(f" №{pattern} → {layer_name}")
            else:
                patterns.append(pattern)
        if patterns and self.trainer.freeze_layers(layer_patterns=patterns):
            self.config.frozen_layers = patterns
            self.logger.success(f"Заморожено по шаблонам: {patterns}")
            
    def _freeze_all_except_interactive(self):
        """Интерактивный выбор слоев НЕ для заморозки"""
        self.console.print("\n⚠️ [bold]Режим: ЗАМОРОЗИТЬ ВСЕ, КРОМЕ указанных[/bold]")
        self.console.print("🔧 Введите шаблоны слоев, которые должны остаться обучаемыми")
        self.console.print(" Пример: 'layer.23, lm_head' для только выходных слоёв")
        patterns_input = input("Шаблоны исключений: ").strip()
        if patterns_input:
            patterns = [p.strip() for p in patterns_input.split(',') if p.strip()]
            if self.trainer.freeze_layers(freeze_all_except=patterns):
                # Сохраняем в специальном формате для понимания
                self.config.frozen_layers = [f"except:{p}" for p in patterns]
                self.logger.success(f"Заморожено все кроме: {patterns}")
                
    def _unfreeze_all_layers(self):
        """Разморозка всех слоев"""
        if self._confirm_action("❓ Разморозить ВСЕ слои модели?"):
            # Проходим по всем параметрам и размораживаем
            for name, param in self.trainer.model.named_parameters():
                param.requires_grad = True
            self.config.frozen_layers = []
            self.logger.success("✅ Все слои разморожены!")
            
    def _select_freeze_preset(self):
        """Выбор пресета заморозки - компактная версия"""
        if not self.trainer.model:
            self.logger.error("Модель не загружена!")
            return
        # Добавь лог флагов в статус
        memory_flag = self.trainer.model.enable_memory_update_during_inference.item() if hasattr(self.trainer.model, 'enable_memory_update_during_inference') else False
        self.console.print(f"💾 Memory updates in inference: {'[green]ON[/green]' if memory_flag else '[red]OFF[/red]'}")
        presets = self.trainer.get_adaptive_presets()
        total_layers = self.trainer._detect_total_layers()
        self.console.print(f"\n🎯 [bold]Пресеты для {total_layers}L модели:[/bold]")
        # Группируем по риску
        safe = [p for p in presets if "🟢" in p['risk_level']]
        medium = [p for p in presets if "🟡" in p['risk_level']]
        risky = [p for p in presets if "⚠️" in p['risk_level']]
        self.console.print("\n🟢 БЕЗОПАСНЫЕ:")
        for p in safe:
            idx = presets.index(p) + 1
            self.console.print(f" {idx:2d}. {p['name']} - {p['best_for']}")
        self.console.print("\n🟡 ОСНОВНЫЕ:")
        for p in medium:
            idx = presets.index(p) + 1
            self.console.print(f" {idx:2d}. {p['name']} - {p['best_for']}")
        self.console.print("\n⚠️ ЭКСПЕРИМЕНТЫ:")
        for p in risky:
            idx = presets.index(p) + 1
            self.console.print(f" {idx:2d}. {p['name']} - {p['best_for']}")
        choice = input(f"\n🎯 Выбор (1-{len(presets)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(presets):
                preset = presets[choice_idx]
                # 🎯 КОРОТКОЕ ПОДТВЕРЖДЕНИЕ
                self.console.print(f"\n✅ {preset['name']} {preset['risk_level']}")
                self.console.print(f"📝 {preset['best_for']}")
                confirm = input("\nПрименить? (y/N): ").strip().lower()
                if confirm in ['y', 'yes', 'да', 'д']:
                    if preset['mode'] == 'freeze_all_except':
                        success = self.trainer.freeze_layers(freeze_all_except=preset['patterns'])
                        if success:
                            self.config.frozen_layers = [f"except:{p}" for p in preset['patterns']]
                    else:
                        success = self.trainer.freeze_layers(layer_patterns=preset['patterns'])
                        if success:
                            self.config.frozen_layers = preset['patterns']
                    if success:
                        self.logger.success(f"✅ {preset['name']}")
                        # 📊 ТОЛЬКО САМАЯ ВАЖНАЯ СТАТИСТИКА
                        stats = self.trainer.get_layer_statistics()
                        if stats:
                            self.console.print(f"📊 Обучается: {stats['trainable_ratio']:.1%} параметров")
            else:
                self.logger.error("Неверный номер")
        except ValueError:
            self.logger.error("Введите число")
            
    def _show_model_structure(self):
        """Показать структуру модели для справки по паттернам"""
        if not self.trainer.model:
            self.logger.error("Модель не загружена!")
            return
        self.console.print("\n🔍 [bold]СТРУКТУРА МОДЕЛИ (для справки):[/bold]")
        # Собираем уникальные паттерны и группы слоев
        common_patterns = defaultdict(int)
        layer_groups = {
            'embeddings': [],
            'attention': [],
            'mlp': [],
            'output': [],
            'norm': [],
            'other': []
        }
        total_params = 0
        for name, param in self.trainer.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            # Анализируем паттерны в названиях
            for pattern in ['embed', 'attn', 'mlp', 'output', 'norm', 'layer', 'head', 'proj', 'dense']:
                if pattern in name.lower():
                    common_patterns[pattern] += param_count
                    break
            # Группируем слои по типам
            name_lower = name.lower()
            if any(x in name_lower for x in ['embed']):
                layer_groups['embeddings'].append((name, param_count))
            elif any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                layer_groups['attention'].append((name, param_count))
            elif any(x in name_lower for x in ['mlp', 'ffn', 'gate', 'up', 'down', 'dense']):
                layer_groups['mlp'].append((name, param_count))
            elif any(x in name_lower for x in ['lm_head', 'output', 'classifier']):
                layer_groups['output'].append((name, param_count))
            elif any(x in name_lower for x in ['norm', 'ln']):
                layer_groups['norm'].append((name, param_count))
            else:
                layer_groups['other'].append((name, param_count))
        # Показываем общую статистику
        self.console.print(f"📊 Всего параметров: {total_params:,}")
        self.console.print("")
        # Показываем группы слоев
        self.console.print("🎯 [bold]ГРУППЫ СЛОЕВ (для использования в шаблонах):[/bold]")
        for group_name, layers in layer_groups.items():
            if layers:
                group_params = sum(count for _, count in layers)
                self.console.print(f" • [cyan]{group_name.upper()}[/cyan]: {len(layers)} слоёв, {group_params:,} параметров")
        self.console.print("")
        # Показываем распространенные паттерны
        self.console.print("🔤 [bold]РАСПРОСТРАНЕННЫЕ ПАТТЕРНЫ:[/bold]")
        for pattern, count in sorted(common_patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                self.console.print(f" • • '{pattern}': {count:,} параметров")
        self.console.print("")
        # Показываем примеры из каждой группы
        self.console.print("📋 [bold]ПРИМЕРЫ НАЗВАНИЙ СЛОЕВ:[/bold]")
        examples_shown = 0
        for group_name, layers in layer_groups.items():
            if layers and examples_shown < 12: # Ограничиваем количество примеров
                example_name, example_count = layers[0]
                self.console.print(f" • {example_name} ({example_count:,} params)")
                examples_shown += 1
        self.console.print("")
        self.console.print("🔧 [yellow]ПОДСКАЗКА:[/yellow]")
        self.console.print(" • Используйте части названий из примеров выше")
        self.console.print(" • Например: 'embed', 'attn', 'layer.23', 'lm_head'")
        self.console.print(" • Или: '1,3,5' для выбора по номерам из полного списка")
        
    def _show_freezing_guide(self):
        """Показать гид по заморозке"""
        guide_path = os.path.join(os.path.dirname(__file__), "freezing_guide.txt")
        if os.path.exists(guide_path):
            try:
                with open(guide_path, 'r', encoding='utf-8') as f:
                    guide_content = f.read()
                self.console.print(Panel(
                    guide_content,
                    title="📖 Гид по заморозке слоев",
                    style="blue",
                    width=100
                ))
            except Exception as e:
                self.logger.error(f"Не удалось прочитать файл справки: {e}")
        else:
            # Inline guide
            guide = """
    Файл справки не найден. Используйте пресеты:
    🍰 Слоеный пирог - Адаптация стиля (top 50%)
    📚 Словарь - Новая терминология (embeddings only)
    🧠 Факты - Новые знания (MLP blocks)
    🎨 Косметика - Легкая правка (input + output)
    💰 Экономия - Максимум памяти
            """
            self.console.print(Panel(
                guide,
                title="Краткая справка",
                style="yellow"
            ))
            
    def _edit_system_params(self):
        """Только системные настройки"""
        params = {
            'use_mixed_precision': ('Использовать смешанную точность (AMP)', 'bool'),
            'use_bfloat16': ('Использовать bfloat16', 'bool'),
            'use_progress_bar': ('Использовать прогресс-бар', 'bool'),
            'log_level': ('Уровень логов (INFO, WARNING, ERROR)', 'str'),
            'cache_clear_steps': ('Шаги очистки кеша CUDA', 'int'),
            'log_gpu_steps': ('Шаги логирования GPU', 'int'),
            'log_gradients_steps': ('Шаги логирования градиентов', 'int'),
            'num_workers': ('Количество workers для DataLoader', 'int'),
        }
        for p, (desc, p_type) in params.items():
            self._edit_config_param(p, desc, p_type)
            
    def _save_load_config(self):
        """Сохранение/загрузка конфигурации"""
        choice = input("1. Сохранить\n2. Загрузить\nВыберите: ").strip()
        if choice == '1':
            path = input("Путь для сохранения (config.yaml): ").strip() or "config.yaml"
            self.config.save_yaml(path)
            self.logger.success(f"Конфигурация сохранена: {path}")
        elif choice == '2':
            path = input("Путь к конфигурации: ").strip()
            if path and os.path.exists(path):
                self.config = TrainingConfig.from_yaml(path)
                self.trainer.config = self.config
                self.logger.success(f"Конфигурация загружена: {path}. Некоторые изменения (напр. заморозка) требуют перезагрузки модели.")
            else:
                self.logger.error("Файл не найден")
                
    def _edit_data_processing_params(self):
        """Унифицированное меню для всех параметров обработки данных"""
        params = {
            'sequence_packing': ('Включить Sequence Packing', 'bool'),  # 🔥 НОВОЕ
            'packing_max_tokens': ('Макс. токенов в упаковке', 'int'),  # 🔥 НОВОЕ
            'packing_min_items': ('Мин. документов в упаковке', 'int'),  # 🔥 НОВОЕ
            'auto_clean_text': ('Автоматическая очистка текста', 'bool'),  # 🔥 НОВОЕ
            'split_long_texts': ('Разбивать длинные тексты на чанки', 'bool'),
            'overlap_ratio': ('Перекрытие между чанками (0.0-0.5)', 'float'),
            'max_length': ('Максимальная длина последовательности', 'int'),
            'min_token_length': ('Минимальная длина токенов', 'int'),
            'auto_clean_dataset': ('Автоматическая очистка датасета', 'bool'),
            'normalize_unicode': ('Нормализовать Unicode', 'bool'),
            'adjust_chunk_boundaries': ('Корректировать границы предложений', 'bool'),
            'max_boundary_adjustment': ('Макс. символов для корректировки', 'int'),
        }
        for p, (desc, p_type) in params.items():
            self._edit_config_param(p, desc, p_type)
            
# ===================================================================
# МЕНЮ ОПТИМИЗАЦИИ
# ===================================================================

    def _edit_differential_lr_settings(self):
        """Настройки дифференциальных LR для Thalia"""
        if not self.trainer.model:
            self.logger.error("Сначала загрузите модель!")
            return
        
        # 🔥 ОПРЕДЕЛЯЕМ АРХИТЕКТУРУ
        is_thalia = (
            hasattr(self.trainer.model, 'personality_core') and
            hasattr(self.trainer.model, 'adaptive_memory')
        )
        
        if is_thalia:
            self._edit_differential_lr_settings_thalia()
        else:
            self._edit_differential_lr_settings_legacy()

    def _edit_differential_lr_settings_thalia(self):
        """Пресеты для архитектуры Thalia"""
        self.console.print("\n🎯 [bold]ДИФФЕРЕНЦИАЛЬНЫЕ LR ДЛЯ THALIA:[/bold]")
        self.console.print("🧠 Обнаружена архитектура: Transformer + Personality + Living Layer + Mamba")
        self.console.print(f"📊 Базовая LR: {self.config.learning_rate}")
        
        # 🔥 ПРЕСЕТЫ ДЛЯ THALIA
        presets = {
            "1": {
                "name": "🛡️ Безопасный старт",
                "desc": "База медленно, личность средне, память быстро",
                "values": {
                    "embeddings": 0.05,          # ← добавить
                    "transformer_base": 0.1,
                    "personality_core": 0.3,
                    "living_layer": 0.5,
                    "adaptive_memory": 1.5,
                    "experience_exchange": 1.0,
                    "other": 0.2
                },
                "best_for": "Первое обучение Thalia"
            },
            "2": {
                "name": "🚀 Быстрая настройка памяти",
                "desc": "Акцент на Mamba Heads и обмен опытом",
                "values": {
                    "embeddings": 0.03,          # ← добавить
                    "transformer_base": 0.05,
                    "personality_core": 0.2,
                    "living_layer": 0.8,
                    "adaptive_memory": 2.0,
                    "experience_exchange": 1.5,
                    "other": 0.3
                },
                "best_for": "Быстрая адаптация памяти"
            },
            "3": {
                "name": "🎯 Сбалансированный (РЕКОМЕНДУЕМЫЙ)",
                "desc": "Оптимальный баланс всех компонентов",
                "values": {
                    "embeddings": 0.04,          # ← добавить
                    "transformer_base": 0.15,
                    "personality_core": 0.4,
                    "living_layer": 0.7,
                    "adaptive_memory": 1.8,
                    "experience_exchange": 1.2,
                    "other": 0.25
                },
                "best_for": "Стандартное обучение Thalia"
            },
            "4": {
                "name": "🧠 Развитие личности",
                "desc": "Акцент на Personality Core и эмоции",
                "values": {
                    "embeddings": 0.05,          # ← добавить
                    "transformer_base": 0.1,
                    "personality_core": 0.8,
                    "living_layer": 0.6,
                    "adaptive_memory": 1.2,
                    "experience_exchange": 1.0,
                    "other": 0.2
                },
                "best_for": "Развитие характера и эмоциональности"
            },
            "5": {
                "name": "⚡ Только Mamba (ultra-safe)",
                "desc": "Все заморожено кроме памяти Mamba",
                "values": {
                    "embeddings": 0.01,          # ← добавить
                    "transformer_base": 0.01,
                    "personality_core": 0.01,
                    "living_layer": 0.01,
                    "adaptive_memory": 3.0,
                    "experience_exchange": 0.01,
                    "other": 0.01
                },
                "best_for": "Тестирование Mamba без риска"
            },
            "6": {
                "name": "🤖 Интеллектуальный фокус",
                "desc": "База средне, память быстро, обмен опытом быстро",
                "values": {
                    "embeddings": 0.03,          # ← добавить
                    "transformer_base": 0.2,
                    "personality_core": 0.3,
                    "living_layer": 0.9,
                    "adaptive_memory": 2.2,
                    "experience_exchange": 1.8,
                    "other": 0.3
                },
                "best_for": "Развитие интеллектуальных способностей"
            },
            "7": {
                "name": "💫 Полная активация",
                "desc": "Все компоненты активно обучаются",
                "values": {
                    "embeddings": 0.02,          # ← добавить
                    "transformer_base": 0.25,
                    "personality_core": 0.6,
                    "living_layer": 1.0,
                    "adaptive_memory": 2.5,
                    "experience_exchange": 2.0,
                    "other": 0.4
                },
                "best_for": "Активное всестороннее развитие"
            },
            "0": {
                "name": "❄️ Единый LR",
                "desc": "Одинаковый LR для всех компонентов",
                "values": None,
                "best_for": "Простота"
            }
        }
        
        self.console.print("\n🎯 [bold]ПРЕСЕТЫ ДЛЯ THALIA:[/bold]")
        for key, preset in presets.items():
            if preset['values']:
                self.console.print(f"[bold]{key}.[/bold] {preset['name']}")
                self.console.print(f"   📝 {preset['desc']}")
                self.console.print(f"   🎯 {preset['best_for']}")
                
                # Краткая таблица множителей
                mult_str = " | ".join([f"{k}:{v}" for k, v in preset['values'].items()])
                self.console.print(f"   📊 {mult_str}")
                self.console.print("")
            else:
                self.console.print(f"[bold]{key}.[/bold] {preset['name']}")
                self.console.print(f"   📝 {preset['desc']}")
                self.console.print("")
        
        choice = input(f"\n🎯 Выберите пресет (0-{len(presets)-1}): ").strip()
        
        if choice == '0':
            self.config.use_differential_lr = False
            self.logger.success("✅ Дифференциальные LR отключены")
        elif choice in presets and presets[choice]['values'] is not None:
            self.config.lr_multipliers = presets[choice]['values']
            self.config.use_differential_lr = True
            
            # Красивая таблица
            from rich.table import Table
            table = Table(title=f"Пресет: {presets[choice]['name']}", style="cyan")
            table.add_column("Компонент", style="green")
            table.add_column("Множитель", style="yellow", justify="center")
            table.add_column("LR", style="magenta", justify="right")
            table.add_column("Скорость", style="blue")
            
            base_lr = self.config.learning_rate
            for comp, mult in self.config.lr_multipliers.items():
                actual_lr = base_lr * mult
                
                # Определяем скорость
                if mult < 0.2:
                    speed = "🐢 Очень медленно"
                    speed_emoji = "🐢"
                elif mult < 0.5:
                    speed = "🚶 Медленно"
                    speed_emoji = "🚶"
                elif mult < 1.0:
                    speed = "🏃 Средне"
                    speed_emoji = "🏃"
                elif mult < 2.0:
                    speed = "🚀 Быстро"
                    speed_emoji = "🚀"
                else:
                    speed = "⚡ Очень быстро"
                    speed_emoji = "⚡"
                
                # Имена компонентов для красоты
                comp_names = {
                    'transformer_base': '🧠 Трансформер',
                    'personality_core': '🎭 Личность',
                    'living_layer': '💫 Living Layer',
                    'adaptive_memory': '🧠 Mamba',
                    'experience_exchange': '🔄 Обмен опытом',
                    'other': '📦 Остальное'
                }
                
                comp_display = comp_names.get(comp, comp)
                table.add_row(
                    comp_display,
                    f"{mult:.2f}",
                    f"{actual_lr:.2e}",
                    f"{speed_emoji} {speed}"
                )
            
            self.console.print(table)
            self.logger.success(f"✅ Применен пресет: {presets[choice]['name']}")
        else:
            self.logger.error("❌ Неверный выбор")
            
# ===================================================================
# МЕНЮ СИСТЕМНОЙ ИНФОРМАЦИИ
# ===================================================================

    def _show_status(self):
        """Отображение системного статуса"""
        frozen_layers_status = ", ".join(self.config.frozen_layers) if self.config.frozen_layers else "Нет"
        optimizer_status = f"{self.config.optimizer_type.upper()} (LR={self.config.learning_rate})"
        status_text = f"""
[bold green]Статус системы:[/bold green]
[*] Устройство: {self.trainer.device}
[*] Модель: {'Загружена' if self.trainer.model else 'Не загружена'} ({self.config.model_path or 'Не указан'})
[*] Датасет: {self.config.dataset_path or 'Не указан'} (формат: {self.config.dataset_format})
[*] CUDA: {'Доступна' if torch.cuda.is_available() else 'Недоступна'}
[*] AMP / BFloat16: {self.config.use_mixed_precision} / {self.config.use_bfloat16}
[bold magenta]Замороженные слои:[/bold magenta] {frozen_layers_status}
[bold green]Ключевые параметры:[/bold green]
[*] Оптимизатор: {optimizer_status}
[*] Разбиение длинных текстов: {self.config.split_long_texts} (перекрытие: {self.config.overlap_ratio})
[*] Ключи входа: {', '.join(self.config.input_keys)}
[*] Ключи выхода: {', '.join(self.config.output_keys)}
[*] Шедулер: {self.config.lr_scheduler_type}
[*] Градиент-аккумуляция: {self.config.gradient_accumulation_steps}
[*] Размер батча: {self.config.batch_size}
"""
        self.console.print(Panel(status_text, title="Статус системы", style="green"))
        
# ===================================================================
# СЛУЖЕБНЫЕ МЕТОДЫ
# ===================================================================

    def _get_user_input(self, prompt: str, default: str = "") -> str:
        try:
            result = input(prompt).strip()
            return result if result else default
        except (EOFError, KeyboardInterrupt):
            return "exit"
            
    def _print_section_title(self, title: str):
        self.console.print(Panel.fit(title, style="bold blue"))
        
    def _confirm_action(self, message: str) -> bool:
        try:
            choice = input(f"{message} (y/n): ").strip().lower()
            return choice in ['y', 'yes', 'д', 'да']
        except (EOFError, KeyboardInterrupt):
            return False
            
if __name__ == "__main__":
    # ⚠️ ФИКС: Добавлен try-except для KeyboardInterrupt + argparse stub
    parser = argparse.ArgumentParser(description="GPT Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    try:
        interface = InteractiveInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C.")
        sys.exit(0)
