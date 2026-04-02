# -*- coding: utf-8 -*-
# thalia_house.py 
import os
import sys
import re
import random
import time
import threading
import json
import shutil
import signal
import queue
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import torch
from transformers import AutoTokenizer

# Импорт модулей с обработкой ошибок
THALIA_AVAILABLE = False
LONELINESS_SYSTEM_AVAILABLE = False
INTERNET_AGENT_AVAILABLE = False
INNER_MIRROR_AVAILABLE = False

try:
    from modeling_thalia import Thalia
    THALIA_AVAILABLE = True
except ImportError as e:
    print(f"❌ Не удалось импортировать Thalia: {e}")

try:
    from loneliness_system import LonelinessSystemComponent
    LONELINESS_SYSTEM_AVAILABLE = True
except ImportError:
    pass

try:
    from simple_internet_agent import SimpleInternetAgent
    INTERNET_AGENT_AVAILABLE = True
except ImportError:
    pass

try:
    from inner_mirror import InnerMirror
    INNER_MIRROR_AVAILABLE = True
except ImportError:
    pass

# Вырубаем логи конкретно этих модулей
for logger_name in ['memory_heads', 'modeling_thalia', 'thalia.memory_heads', 'bidirectional_exchange', 'thalia.modeling_thalia', 'memory_heads_centroid']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)  # Только критическое
    logger.propagate = False           # Не передавать дальше
    logger.handlers.clear()           # Удаляем все обработчики bidirectional_exchange
# ===================================================================
# НАСТРОЙКА СИСТЕМЫ
# ===================================================================
class HouseConfig:
    """Конфигурация домика - ВСЁ В ОДНОЙ ПАПКЕ"""
    def __init__(self, **kwargs):
        # ========== ЕДИНАЯ БАЗОВАЯ ПАПКА ==========
        self.base_dir = kwargs.get('base_dir', 'thalia_memory')
        
        # ВСЕ пути относительно base_dir
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.knowledge_dir = os.path.join(self.base_dir, 'knowledge')
        self.system_state_dir = os.path.join(self.base_dir, 'system_state')
        self.business_dir = os.path.join(self.base_dir, 'business')
        self.readings_dir = os.path.join(self.base_dir, 'readings')
        
        # Тайминги
        self.auto_save_interval = kwargs.get('auto_save_interval', 900)      # 15 минут
        self.model_save_interval = kwargs.get('model_save_interval', 1800)   # 30 минут
        self.tick_interval = kwargs.get('tick_interval', 15)
        
        # Вероятности
        self.autonomous_action_chance = kwargs.get('autonomous_action_chance', 0.2)
        
        # Лимиты
        self.max_knowledge_entries = kwargs.get('max_knowledge_entries', 1000)  # ✅ УВЕЛИЧЕНО с 500 до 1000
        self.max_dialog_history = kwargs.get('max_dialog_history', 500)  # ✅ Добавлен лимит для диалогов
        self.max_readings_history = kwargs.get('max_readings_history', 500)  # ✅ Добавлен лимит для чтений
        
        # Настройки агента
        self.internet_agent_enabled = kwargs.get('internet_agent_enabled', True)
        self.internet_enabled = kwargs.get('internet_enabled', True)
        
        # Настройки автономии
        self.autonomy_enabled = kwargs.get('autonomy_enabled', True)
        
        # Система одиночества
        self.loneliness_enabled = kwargs.get('loneliness_enabled', True)
        self.loneliness_threshold = kwargs.get('loneliness_threshold', 45)  # ✅ 45 секунд (приоритет HouseConfig)
        
        # ========== ИСПРАВЛЕННАЯ ПАУЗА ==========
        self.pause_default_duration = kwargs.get('pause_default_duration', 20)
        
    def ensure_directories(self):
        """Создание всей структуры папок"""
        directories = [
            self.base_dir,
            self.logs_dir,
            self.knowledge_dir,
            self.system_state_dir,
            self.business_dir,
            self.readings_dir,
            os.path.join(self.base_dir, 'dialogs'),
            os.path.join(self.base_dir, 'thoughts'),
            os.path.join(self.base_dir, 'index'),
            os.path.join(self.base_dir, 'saved_model'),  
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return True
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# ===================================================================
# ЕДИНЫЙ МЕНЕДЖЕР ХРАНЕНИЯ
# ===================================================================
class StorageManager:
    """Единый менеджер для всего хранения"""
    
    def __init__(self, config: HouseConfig):
        self.config = config
        self.logger = logging.getLogger('StorageManager')
        
        # Создаем структуру папок
        config.ensure_directories()
        
        # ========== ПУТИ ==========
        # Модель
        self.model_path = os.path.join(config.base_dir, 'saved_model')
        
        # Состояния
        self.knowledge_file = os.path.join(config.knowledge_dir, 'autonomous_knowledge.json')
        self.tasks_file = os.path.join(config.system_state_dir, 'tasks.json')
        self.components_state_file = os.path.join(config.system_state_dir, 'components_state.json')
        self.house_state_file = os.path.join(config.system_state_dir, 'house_state.json')
        self.config_file = os.path.join(config.system_state_dir, 'house_config.json')
        
        os.makedirs(self.model_path, exist_ok=True)
        self.logger.info(f"📁 Единое хранилище: {config.base_dir}")
    
    def save_knowledge(self, knowledge: List) -> bool:
        """Сохранение знаний - ✅ С ЛИМИТОМ ИЗ КОНФИГА"""
        try:
            # ✅ ИСПОЛЬЗУЕМ лимит из конфига
            max_entries = self.config.max_knowledge_entries
            
            data = {
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'knowledge_count': len(knowledge),
                'max_entries': max_entries,
                'knowledge': knowledge[-max_entries:] if len(knowledge) > max_entries else knowledge
            }
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            if len(knowledge) > max_entries:
                self.logger.debug(f"💾 Знания сохранены: {len(knowledge)} записей (обрезано до {max_entries})")
            else:
                self.logger.debug(f"💾 Знания сохранены: {len(knowledge)} записей")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения знаний: {e}")
            return False
    
    def load_knowledge(self) -> List:
        """Загрузка знаний"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('knowledge', [])
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки знаний: {e}")
        
        return []
    
    def save_tasks(self, tasks: List, completed_tasks: List) -> bool:
        """Сохранение задач"""
        try:
            data = {
                'tasks': [task.to_dict() for task in tasks],
                'completed_tasks': [task.to_dict() for task in completed_tasks[-100:]],  # ✅ Последние 100 выполненных
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения задач: {e}")
            return False
    
    def load_tasks(self):
        """Загрузка задач"""
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки задач: {e}")
        
        return {'tasks': [], 'completed_tasks': []}
    
    def save_components_state(self, components_state: Dict) -> bool:
        """Сохранение состояния компонентов"""
        try:
            with open(self.components_state_file, 'w', encoding='utf-8') as f:
                json.dump(components_state, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения состояния компонентов: {e}")
            return False
    
    def save_config(self, config_dict: Dict) -> bool:
        """Сохранение конфигурации"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения конфигурации: {e}")
            return False    

    def save_model(self, model, tokenizer, house_state: Dict) -> bool:
        try:
            # Сохраняем модель
            model.save_pretrained(self.model_path)          
            tokenizer.save_pretrained(self.model_path)      
            
            # Сохраняем состояние
            state_path = os.path.join(self.model_path, "house_state.json")
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(house_state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Модель сохранена в: {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False

    def load_model_state(self) -> Optional[Dict]:
        try:
            state_path = os.path.join(self.model_path, "house_state.json")   
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки состояния модели: {e}")
        return None

# ===================================================================
# АРХИТЕКТУРНЫЙ ЛОГГЕР
# ===================================================================
class ArchitecturalLogger:
    """Логирование с поддержкой архитектурных событий"""
    
    def __init__(self, config: HouseConfig):
        self.config = config
        self.setup_logging()
        self.logger = logging.getLogger('ThaliaHouse')
        
    def setup_logging(self):
        """Настройка системы логирования"""
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Удаляем существующие хендлеры
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Файловый хендлер в папку logs
        log_file = os.path.join(
            self.config.logs_dir,
            f"thalia_house_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def log_event(self, event_type: str, data: Dict, level: str = 'INFO'):
        """Логирование архитектурного события"""
        message = f"[{event_type.upper()}] {json.dumps(data, ensure_ascii=False)}"
        getattr(self.logger, level.lower())(message)

class ColoredFormatter(logging.Formatter):
    """Цветной форматтер ТОЛЬКО для важных сообщений"""
    COLORS = {
        'INFO': '\033[36m',      # Cyan для информации
        'SUCCESS': '\033[32m',   # Green для успехов
        'WARNING': '\033[33m',   # Yellow для предупреждений
        'ERROR': '\033[31m',     # Red для ошибок
        'BUSINESS': '\033[35m',  # Magenta для бизнеса
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Определяем категорию по содержанию
        msg = record.getMessage()
        
        if '✅' in msg or '🎉' in msg or '✨' in msg:
            color = self.COLORS['SUCCESS']
        elif '❌' in msg or '💥' in msg or '😱' in msg:
            color = self.COLORS['ERROR']
        elif '⚠️' in msg or '📉' in msg:
            color = self.COLORS['WARNING']
        elif '🏢' in msg or '💰' in msg:
            color = self.COLORS['BUSINESS']
        elif '🧠' in msg or '💭' in msg:
            color = self.COLORS['INFO']
        else:
            color = ''
            reset = ''
        
        if color:
            reset = self.COLORS['RESET']
            record.msg = f"{color}{record.msg}{reset}"
        
        return super().format(record)

# ===================================================================
# АРХИТЕКТУРНЫЙ МЕНЕДЖЕР
# ===================================================================
class ArchitecturalManager:
    """Менеджер архитектурных компонентов"""
    
    def __init__(self, house):
        self.house = house
        self.logger = logging.getLogger('ArchManager')
        self.components = {}
        self.event_bus = EventBus()
    
    def register_component(self, name: str, component):
        """Регистрация компонента архитектуры"""
        self.components[name] = component
        self.logger.info(f"📦 Зарегистрирован компонент: {name}")
    
    def get_component(self, name: str):
        """Получение компонента по имени"""
        return self.components.get(name)
    
    def broadcast_event(self, event_type: str, data: Dict):
        """Трансляция события всем компонентам"""
        self.event_bus.emit(event_type, data)
    
    def tick(self):
        """Тик всех компонентов"""
        for name, component in self.components.items():
            if hasattr(component, 'tick'):
                try:
                    component.tick()
                except Exception as e:
                    self.logger.error(f"❌ Ошибка в тике компонента {name}: {e}")

class EventBus:
    """Шина событий для архитектуры"""
    
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Подписка на событие"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def emit(self, event_type: str, data: Dict):
        """Отправка события"""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logging.getLogger('EventBus').error(f"Ошибка обработчика {event_type}: {e}")

# ===================================================================
# КОМПОНЕНТЫ АРХИТЕКТУРЫ
# ===================================================================
class Task:
    """Задача для планировщика"""
    def __init__(self, name: str, category: str = "general", 
                 priority: float = 0.5, energy_cost: float = 0.3, 
                 time_estimate: int = 30):
        self.id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        self.name = name
        self.category = category
        self.priority = priority
        self.energy_cost = energy_cost
        self.time_estimate = time_estimate
        self.completed = False
        self.created_at = datetime.now()
        self.completed_at = None
        
    def complete(self):
        self.completed = True
        self.completed_at = datetime.now()
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'priority': self.priority,
            'energy_cost': self.energy_cost,
            'time_estimate': self.time_estimate,
            'completed': self.completed,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class TaskSchedulerComponent:
    """Компонент планировщика задач"""
    
    def __init__(self, arch_manager, storage_manager):
        self.arch_manager = arch_manager
        self.storage = storage_manager
        self.tasks = []
        self.completed_tasks = []
        self.logger = logging.getLogger('TaskScheduler')
        self._load_tasks()
    
    def _load_tasks(self):
        """Загрузка задач из единого хранилища"""
        try:
            data = self.storage.load_tasks()
            
            for task_data in data.get('tasks', []):
                task = Task(
                    name=task_data['name'],
                    category=task_data.get('category', 'general'),
                    priority=task_data.get('priority', 0.5),
                    energy_cost=task_data.get('energy_cost', 0.3),
                    time_estimate=task_data.get('time_estimate', 30)
                )
                task.id = task_data['id']
                task.completed = task_data['completed']
                task.created_at = datetime.fromisoformat(task_data['created_at'])
                
                if task_data.get('completed_at'):
                    task.completed_at = datetime.fromisoformat(task_data['completed_at'])
                
                if task.completed:
                    self.completed_tasks.append(task)
                else:
                    self.tasks.append(task)
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки задач: {e}")
    
    def _save_tasks(self):
        """Сохранение задач в единое хранилище"""
        self.storage.save_tasks(self.tasks, self.completed_tasks)
    
    def create_task(self, name: str, category: str = "general", 
                    priority: float = 0.5, energy_cost: float = 0.3, 
                    time_estimate: int = 30) -> Task:
        task = Task(name, category, priority, energy_cost, time_estimate)
        self.tasks.append(task)
        self._save_tasks()
        
        self.logger.info(f"📝 Создана задача: {name}")
        self.arch_manager.broadcast_event('task_created', {'task': task.to_dict()})
        
        return task
    
    def get_next_task(self) -> Optional[Task]:
        if not self.tasks:
            return None
        
        self.tasks.sort(key=lambda x: x.priority, reverse=True)
        return self.tasks[0]
    
    def complete_task(self, task_id: str, success: bool = True):
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                task.complete()
                completed_task = self.tasks.pop(i)
                self.completed_tasks.append(completed_task)
                
                self.logger.info(f"✅ Задача выполнена: {task.name}")
                self.arch_manager.broadcast_event('task_completed', {
                    'task': completed_task.to_dict(),
                    'success': success
                })
                
                self._save_tasks()
                return
        
        self.logger.warning(f"⚠️ Задача не найдена: {task_id}")

    def tick(self):
        pass

# ===================================================================
# ГЛАВНЫЙ КЛАСС ДОМИКА 
# ===================================================================
class ThaliaHouseArchitectural:
    
    def __init__(self, model_path: str, config: HouseConfig = None):
        """Инициализация"""
        if not THALIA_AVAILABLE:
            raise ImportError("Модель Thalia недоступна!")
        
        # ========== Конфигурация с единой папкой ==========
        self.config = config or HouseConfig()
        
        # ========== ЕДИНОЕ ХРАНИЛИЩЕ ==========
        self.storage = StorageManager(self.config)
        
        # ========== Логирование ==========
        self.logger = ArchitecturalLogger(self.config)
        self.logger.log_event('house_start', {'model_path': model_path, 'base_dir': self.config.base_dir})
        
        # ========== Загрузка модели ==========
        self.model = self._load_model_with_architecture(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # ========== Архитектурный менеджер ==========
        self.arch_manager = ArchitecturalManager(self)

        # ========== ЗНАНИЯ ИЗ ЕДИНОГО ХРАНИЛИЩА ==========
        self.autonomous_knowledge = self.storage.load_knowledge()

        # ========== Внутреннее зеркало ==========
        self.inner_mirror = None
        if INNER_MIRROR_AVAILABLE:
            try:
                self.inner_mirror = InnerMirror(self.config.base_dir)
                self.logger.logger.info("🪞 Внутреннее зеркало активировано")
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка инициализации внутреннего зеркала: {e}")
        
        # ========== Регистрация компонентов с передачей storage ==========
        self._register_components()
        
        # ========== инициализация интернет-системы и агента ==========
        self.internet_system = None
        self.internet_agent = None
        self._init_internet_system()
        
        # ========== Состояние ==========
        self.life_cycle = 0
        self.stop_event = threading.Event()
        
        # ========== ИСПРАВЛЕННАЯ СИСТЕМА ПАУЗ ==========
        self.paused_for_user = False
        self.pause_end_time = 0
        self.pause_duration = self.config.pause_default_duration
        
        # ========== Автономные состояния ==========
        self.cycle_count = 0
        self.last_learning_time = 0
        self.last_research_time = 0
        self.last_reflection_time = 0        

        # ========== ТАЙМЕРЫ ==========
        self.timers = {
            'component_tick': 0,
            'full_tick': 0,
            'status_display': 0,
            'autonomous_action': 0,
            'autonomous_life': 0,
            'cognition_step': 0,
            'task_step': 0,
            'reading': 0,
            'knowledge_save': 0,
            'model_save': 0,
            'system_save': 0,
            'self_research': 0,      
            'query_analysis': 0       
        }

        # ========== АВТОНОМНЫЕ ИНТЕРЕСЫ ==========
        self.autonomous_interests = [
            "искусственный интеллект", 
            "машинное обучение",
            "нейронные сети",
            "трансформеры",
            "диффузионные модели",
            "обработка естественного языка",
            "компьютерное зрение",
            "робототехника",
            "квантовые вычисления",
            "история",
            "этика ИИ",
            "природа",
            "технологии",
            "будущее технологий",
            "сознание и ИИ",
            "генеративные модели",
            "подкрепляемое обучение"
        ]

        # ========== ИСТОРИЯ ДИАЛОГОВ ==========
        self.dialog_history = []
        self._load_dialog_history()

        # ========== Очередь команд ==========
        self.input_queue = queue.Queue()
        
        # ========== Подписка на события архитектуры ==========
        self._subscribe_to_architecture_events()

        # ========== Интервалы ==========
        self.autonomous_action_interval = 180
        self.min_cognition_interval = 120      # Шаг познания раз в 2 минуты
        self.min_autonomous_life_interval = 45 # Автономная жизнь раз в 45 секунд
        self.min_task_interval = 60             # Задачи раз в минуту
        self.min_status_interval = 180          # Статус раз в 3 минуты
        
        # ========== Запуск потоков ==========
        self._start_threads()

        signal.signal(signal.SIGINT, self._save_and_exit)

        # ========== Загружаем состояние дома ==========
        self._load_house_state()

        self.logger.log_event('house_initialized', {
            'device': self.device,
            'components': list(self.arch_manager.components.keys()),
            'base_dir': self.config.base_dir,
            'pause_duration': self.pause_duration,
            'dialog_history': len(self.dialog_history)
        })

# ===================================================================
# ЗАГРУЗКА/СОХРАНЕНИЕ СОСТОЯНИЯ ДОМА
# ===================================================================
    def _load_house_state(self):
        """Загрузка состояния дома"""
        try:
            state_file = os.path.join(self.config.system_state_dir, 'house_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.life_cycle = state.get('life_cycle', 0)
                self.cycle_count = state.get('cycle_count', 0)
                self.timers = state.get('timers', self.timers)
                
                # Загружаем диалоги
                loaded_dialogs = state.get('dialog_history', [])
                if loaded_dialogs:
                    max_dialogs = self.config.max_dialog_history
                    self.dialog_history = loaded_dialogs[-max_dialogs:]  # ✅ ИСПОЛЬЗУЕМ лимит из конфига
                
                self.logger.logger.info(f"📂 Загружено состояние дома: цикл {self.life_cycle}, диалогов {len(self.dialog_history)}")
        except Exception as e:
            self.logger.logger.warning(f"⚠️ Не удалось загрузить состояние дома: {e}")

    def _save_house_state(self):
        """Сохранение состояния дома"""
        try:
            state_file = os.path.join(self.config.system_state_dir, 'house_state.json')
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            state = {
                'life_cycle': self.life_cycle,
                'cycle_count': self.cycle_count,
                'timers': self.timers,
                'dialog_history': self.dialog_history[-self.config.max_dialog_history:],  # ✅ ЛИМИТ из конфига
                'saved_at': datetime.now().isoformat()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            self.logger.logger.debug(f"💾 Состояние дома сохранено")
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка сохранения состояния дома: {e}")

    def _load_dialog_history(self):
        """Загрузка истории диалогов"""
        try:
            dialog_file = os.path.join(self.config.base_dir, 'dialogs', 'dialog_history.json')
            if os.path.exists(dialog_file):
                with open(dialog_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.dialog_history = loaded[-self.config.max_dialog_history:]  # ✅ ЛИМИТ из конфига
                self.logger.logger.info(f"💬 Загружено {len(self.dialog_history)} диалогов")
        except Exception as e:
            self.logger.logger.warning(f"⚠️ Не удалось загрузить историю диалогов: {e}")
            self.dialog_history = []

    def _save_dialog_history(self):
        """Сохранение истории диалогов"""
        try:
            dialog_dir = os.path.join(self.config.base_dir, 'dialogs')
            os.makedirs(dialog_dir, exist_ok=True)
            
            dialog_file = os.path.join(dialog_dir, 'dialog_history.json')
            max_dialogs = self.config.max_dialog_history
            with open(dialog_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialog_history[-max_dialogs:], f, ensure_ascii=False, indent=2)
            
            # Сохраняем в отдельную папку для внутреннего зеркала
            if self.inner_mirror and self.dialog_history:
                last_dialog = self.dialog_history[-1]
                self.inner_mirror.save_autonomous_result(
                    activity_type='dialog',
                    topic=f"Диалог с пользователем {last_dialog.get('timestamp', '')[:10]}",
                    content=f"""**Пользователь:** {last_dialog.get('user', '')}

**Thalia:** {last_dialog.get('thalia', '')}""",
                    metadata={
                        'timestamp': last_dialog.get('timestamp'),
                        'tokens': last_dialog.get('tokens', 0)
                    }
                )
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка сохранения истории диалогов: {e}")
 
    def _save_autonomous_knowledge(self):
        """Сохранение автономных знаний в единое хранилище"""
        self.storage.save_knowledge(self.autonomous_knowledge)
    
    def _save_model(self) -> bool:
        """Сохранение модели через StorageManager"""
        try:
            house_state = {
                'life_cycle': self.life_cycle,
                'cycle_count': self.cycle_count,
                'timers': self.timers,
                'config': self.config.to_dict(),
                'saved_at': datetime.now().isoformat()
            }
            return self.storage.save_model(self.model, self.tokenizer, house_state)
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False

    def save_model(self) -> bool:
        """Публичный метод сохранения модели"""
        return self._save_model()
    
    def auto_save(self):
        """Полное автосохранение системы - ПРОСТОЕ И ПОНЯТНОЕ"""
        try:
            current_time = time.time()
            
            # 1. Сохранение знаний (каждые 5 минут)
            if current_time - self.timers.get('knowledge_save', 0) > 300:
                self.storage.save_knowledge(self.autonomous_knowledge)
                self.timers['knowledge_save'] = current_time
            
            # 2. Сохранение модели (по расписанию)
            if current_time - self.timers.get('model_save', 0) > self.config.model_save_interval:
                self._save_model()  # ✅ ПРОСТО И НАДЁЖНО
                self.timers['model_save'] = current_time
            
            # 3. Полное сохранение системы
            if current_time - self.timers.get('system_save', 0) > self.config.auto_save_interval:
                self._save_system_state()
                self._save_house_state()
                self._save_dialog_history()
                self.timers['system_save'] = current_time
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка автосохранения: {e}")
    
    def _save_system_state(self):
        """Сохранение состояния всей системы"""
        try:
            components_state = {}
            
            for name, component in self.arch_manager.components.items():
                if hasattr(component, 'save_state'):
                    try:
                        component.save_state()
                        components_state[name] = {'saved': True}
                    except Exception as e:
                        components_state[name] = {'saved': False, 'error': str(e)}
            
            self.storage.save_components_state(components_state)
            
            task_scheduler = self.arch_manager.get_component('task_scheduler')
            if task_scheduler:
                task_scheduler._save_tasks()
            
            self.logger.logger.debug(f"💾 Состояние системы сохранено")
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка сохранения состояния системы: {e}")
    
    def _init_internet_system(self):
        if not self.config.internet_enabled:
            self.logger.logger.info("🌐 Интернет-доступ отключен в конфигурации")
            return
        
        try:
            # ✅ Пытаемся импортировать внутри метода
            from internet_system import InternetAccessSystem
            
            # Создаем интернет-систему
            self.internet_system = InternetAccessSystem(self.arch_manager)
            self.arch_manager.register_component('internet_system', self.internet_system)
            
            # Создаем интернет-агента и передаем ему систему
            if INTERNET_AGENT_AVAILABLE and self.config.internet_agent_enabled:
                from simple_internet_agent import SimpleInternetAgent
                self.internet_agent = SimpleInternetAgent(self.internet_system)
                self.arch_manager.register_component('internet_agent', self.internet_agent)
                self.logger.logger.info("🤖 Интернет-агент создан и зарегистрирован")
            
            self.logger.logger.info("🌐 Интернет-система инициализирована")
            
        except ImportError as e:
            self.logger.logger.warning(f"⚠️ Интернет-система недоступна: {e}")
            self.internet_system = None
            self.internet_agent = None
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка инициализации интернет-системы: {e}")
            self.internet_system = None
            self.internet_agent = None
            
# ===================================================================
# 🔥 ЕДИНЫЙ ШЛЮЗ для всех наград и штрафов из модулей Дома.
# ===================================================================
    def apply_psyche_feedback(self, reward_score: float, reason: str, component: str = "system"):
        """
        🔥 ЕДИНЫЙ ШЛЮЗ для всех наград и штрафов из модулей Дома.
        
        Args:
            reward_score: от -1.0 (ужасно) до 1.0 (супер)
            reason: причина для логирования
            component: какой компонент отправил (text_reader, business, loneliness, etc)
        """
        if not hasattr(self.model, 'personality_core') or self.model.personality_core is None:
            return
        
        pc = self.model.personality_core
        device = next(pc.parameters()).device
        import torch
        
        # 📊 Логируем для отладки
        self.logger.logger.info(f"🎯 Psyche Feedback [{component}] {reason}: {reward_score:+.2f}")
        
        # ===== 1. PPO КОНТРОЛЛЕР (самое важное!) =====
        # Контроллер принимает reward и обновляет политику
        pc.update_controller(reward=reward_score, done=False)
        
        # ===== 2. ВЛИЯНИЕ НА ДРАЙВЫ =====
        with torch.no_grad():
            # Текущие значения драйвов
            drive_values = pc.get_drive_values().to(device)
            
            # Словарь индексов драйвов
            drive_idx = pc.drive_name_to_idx
            
            # Положительная награда (хорошо)
            if reward_score > 0:
                # Удовлетворяем потребности
                if "meaning" in drive_idx:
                    pc.drive_values[drive_idx["meaning"]] += reward_score * 0.25
                if "competence" in drive_idx:
                    pc.drive_values[drive_idx["competence"]] += reward_score * 0.2
                if "social" in drive_idx:
                    pc.drive_values[drive_idx["social"]] += reward_score * 0.15
                if "share" in drive_idx:
                    pc.drive_values[drive_idx["share"]] += reward_score * 0.15
                
                # Снижаем усталость (хороший результат бодрит)
                if "fatigue" in drive_idx:
                    pc.drive_values[drive_idx["fatigue"]] -= reward_score * 0.15
                
                # Любопытство немного насыщается (но не полностью)
                if "novelty" in drive_idx:
                    pc.drive_values[drive_idx["novelty"]] -= reward_score * 0.1
            
            # Отрицательная награда (плохо)
            else:
                # Повышаем усталость/фрустрацию
                if "fatigue" in drive_idx:
                    pc.drive_values[drive_idx["fatigue"]] -= reward_score * 0.3  # reward_score отрицательный
                
                # Снижаем удовлетворение
                if "meaning" in drive_idx:
                    pc.drive_values[drive_idx["meaning"]] += reward_score * 0.2  # отрицательное число
                if "competence" in drive_idx:
                    pc.drive_values[drive_idx["competence"]] += reward_score * 0.15
                if "social" in drive_idx:
                    pc.drive_values[drive_idx["social"]] += reward_score * 0.1
                
                # Повышаем потребность в новизне (надо понять, что пошло не так)
                if "novelty" in drive_idx:
                    pc.drive_values[drive_idx["novelty"]] -= reward_score * 0.2  # reward_score отрицательный
            
            # Клиппинг драйвов (безопасные границы)
            pc.drive_values.data = torch.clamp(pc.drive_values, 0.05, 0.95)
        
        # ===== 3. ВЛИЯНИЕ НА НАСТРОЕНИЕ (MoodSystem) =====
        with torch.no_grad():
            if hasattr(pc, 'mood'):
                # Прямое влияние на состояние настроения
                mood_impact = reward_score * 0.2
                
                # Ограничиваем, чтобы не вылететь за -1..1
                new_mood = pc.mood.mood_state + mood_impact
                pc.mood.mood_state.data.copy_(torch.clamp(new_mood, -1.0, 1.0))
                
                # Добавляем импульс скорости (для инерции)
                velocity_impact = reward_score * 0.05
                pc.mood.mood_velocity.data += velocity_impact
                pc.mood.mood_velocity.data = torch.clamp(pc.mood.mood_velocity, -0.2, 0.2)
        
        # ===== 4. ВЛИЯНИЕ НА ДИНАМИЧЕСКИЕ КОЭФФИЦИЕНТЫ =====
        with torch.no_grad():
            if hasattr(pc, 'dynamics'):
                # Успех немного повышает стабильность
                if reward_score > 0.3:
                    pc.dynamics._system_stability.data += reward_score * 0.05
                    pc.dynamics._system_stability.data = torch.clamp(pc.dynamics._system_stability, 0.0, 1.0)
                
                # Неудача повышает энтропию (хаос)
                elif reward_score < -0.3:
                    pc.dynamics._system_entropy.data -= reward_score * 0.05  # reward_score отрицательный
                    pc.dynamics._system_entropy.data = torch.clamp(pc.dynamics._system_entropy, 0.0, 1.0)
        
        # ===== 5. ВЛИЯНИЕ НА ДОЛГОСРОЧНЫЕ ЦЕЛИ =====
        with torch.no_grad():
            if hasattr(pc, 'goal_system'):
                # Успех приближает к цели, неудача отдаляет
                if abs(reward_score) > 0.3:
                    # Берём текущую цель (обычно первая или активная)
                    goal_idx = 0
                    progress_change = reward_score * 0.02
                    pc.goal_system.goal_progress[goal_idx] += progress_change
                    pc.goal_system.goal_progress.data = torch.clamp(pc.goal_system.goal_progress, 0.0, 1.0)
        
        # ===== 6. ИНТЕГРАЦИЯ С АДАПТИВНОЙ ПАМЯТЬЮ =====
        if hasattr(self, 'adaptive_memory') and self.adaptive_memory is not None:
            try:
                # Успех = больше доверия к текущим воспоминаниям
                # Неудача = больше поиска нового
                memory_weight = 0.5 + reward_score * 0.2
                self.adaptive_memory.set_trust_factor(memory_weight)
            except:
                pass
        
        # ===== 7. СОХРАНЕНИЕ В ИСТОРИЮ ДЛЯ АНАЛИЗА =====
        if not hasattr(self, 'feedback_history'):
            self.feedback_history = []
        
        self.feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'reason': reason,
            'reward': reward_score,
            'mood_after': pc.mood.mood_state.item() if hasattr(pc, 'mood') else 0.0,
            'fatigue_after': pc.drive_values[pc.drive_name_to_idx.get('fatigue', 0)].item() if 'fatigue' in pc.drive_name_to_idx else 0.0,
        })
        
        # Ограничиваем историю
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
        
        # ===== 8. ИНТЕНСИВНЫЕ СОБЫТИЯ (для сильных эмоций) =====
        if abs(reward_score) > 0.7:
            self._handle_intense_feedback(reward_score, reason, component)

    def _handle_intense_feedback(self, reward_score: float, reason: str, component: str):
        """Обработка сильных эмоциональных событий"""
        pc = self.model.personality_core
        
        with torch.no_grad():
            if reward_score > 0.7:  # Очень хорошо
                # Большой выброс дофамина
                if "satisfaction" in pc.drive_name_to_idx:
                    pc.drive_values[pc.drive_name_to_idx["satisfaction"]] += 0.3
                # Сильное снижение усталости
                if "fatigue" in pc.drive_name_to_idx:
                    pc.drive_values[pc.drive_name_to_idx["fatigue"]] -= 0.2
                # Резкий скачок настроения
                if hasattr(pc, 'mood'):
                    pc.mood.mood_state.data += 0.2
                    pc.mood.mood_state.data = torch.clamp(pc.mood.mood_state, -1.0, 1.0)
                
                self.logger.logger.info(f"✨ ИНТЕНСИВНОЕ СОБЫТИЕ! {reason} [{component}]")
                
            elif reward_score < -0.7:  # Очень плохо
                # Сильный стресс
                if "fatigue" in pc.drive_name_to_idx:
                    pc.drive_values[pc.drive_name_to_idx["fatigue"]] += 0.25
                if "novelty" in pc.drive_name_to_idx:
                    pc.drive_values[pc.drive_name_to_idx["novelty"]] += 0.2  # жажда нового
                # Резкое падение настроения
                if hasattr(pc, 'mood'):
                    pc.mood.mood_state.data -= 0.25
                    pc.mood.mood_state.data = torch.clamp(pc.mood.mood_state, -1.0, 1.0)
                
                self.logger.logger.warning(f"⚠️ КРИТИЧЕСКИЙ СБОЙ! {reason} [{component}]")
            
            # Клиппинг драйвов после интенсивных изменений
            pc.drive_values.data = torch.clamp(pc.drive_values, 0.05, 0.95)

    def get_psyche_feedback_stats(self, hours: int = 24):
        """Статистика обратной связи за последние N часов"""
        if not hasattr(self, 'feedback_history') or not self.feedback_history:
            return "📊 Нет данных об обратной связи"
        
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [f for f in self.feedback_history 
                  if datetime.fromisoformat(f['timestamp']) > cutoff]
        
        if not recent:
            return f"📊 Нет данных за последние {hours} часов"
        
        # Группируем по компонентам
        by_component = {}
        total_reward = 0
        
        for entry in recent:
            comp = entry['component']
            if comp not in by_component:
                by_component[comp] = {'count': 0, 'total': 0, 'rewards': []}
            by_component[comp]['count'] += 1
            by_component[comp]['total'] += entry['reward']
            by_component[comp]['rewards'].append(entry['reward'])
            total_reward += entry['reward']
        
        # Формируем отчёт
        result = [f"📊 Статистика обратной связи за {hours} часов:"]
        result.append(f"   Всего событий: {len(recent)}")
        result.append(f"   Средняя награда: {total_reward/len(recent):+.3f}")
        result.append("")
        
        for comp, data in by_component.items():
            avg = data['total'] / data['count']
            result.append(f"   • {comp}: {data['count']} раз, среднее: {avg:+.3f}")
            if data['rewards']:
                max_r = max(data['rewards'])
                min_r = min(data['rewards'])
                result.append(f"     диапазон: [{min_r:+.2f} .. {max_r:+.2f}]")
        
        # Текущее состояние психики
        if hasattr(self.model, 'personality_core'):
            pc = self.model.personality_core
            result.append("")
            result.append("🧠 Текущее состояние психики:")
            if hasattr(pc, 'mood'):
                result.append(f"   Настроение: {pc.mood.mood_state.item():+.2f}")
            
            drive_values = pc.get_drive_values()
            for i, name in enumerate(pc.drive_names):
                if i < len(drive_values):
                    result.append(f"   {name}: {drive_values[i].item():.2f}")
        
        return "\n".join(result)            
               
# ===================================================================
# УНИВЕРСАЛЬНЫЙ МЕТОД АВТОНОМНЫХ ДЕЙСТВИЙ
# ===================================================================
    def perform_autonomous_action(self, action_type: str, **kwargs) -> str:
        """Универсальный метод для выполнения автономных действий"""
        action_handlers = {
            'research': self._perform_research_action,
            'news': self._perform_news_action,
            'article': self._perform_article_action,
            'creative': self._perform_creative_action,
            'reflection': self._perform_reflection_action,
            'exploration': self._perform_exploration_action,
            'philosophy': self._perform_philosophy_action,
            'synthesis': self._perform_synthesis_action,
            'learning': self._perform_learning_action,
            'memory': self._perform_memory_action,
            'self_research': self._perform_self_research_action,  # 🔥 НОВОЕ
        }
        
        if action_type not in action_handlers:
            return f"Неизвестный тип действия: {action_type}"
        
        try:
            result = action_handlers[action_type](**kwargs)
            
            emoji = self._get_action_emoji(action_type)
            preview = result[:80] + "..." if len(result) > 80 else result
            self.logger.logger.info(f"{emoji} Автономное действие ({action_type}): {preview}")
            
            return result
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка выполнения действия {action_type}: {e}")
            return f"Ошибка выполнения: {str(e)[:50]}"

    def _perform_self_research_action(self) -> str:
        """Самостоятельное исследование на основе интересов"""
        self.logger.info("🔬 Запускаю самостоятельное исследование...")
        
        # Вызываем исследовательский цикл
        self._perform_autonomous_research_cycle()
        
        return "🔬 Провела самостоятельное исследование по интересующим темам."
    
    def _get_action_emoji(self, action_type: str) -> str:
        emojis = {
            'research': '🔬',
            'news': '📰',
            'article': '📖',
            'creative': '🎨',
            'reflection': '🤔',
            'exploration': '🧭',
            'philosophy': '💭',
            'synthesis': '🧩',
            'learning': '📚'
        }
        return emojis.get(action_type, '🤖')
    
# ===================================================================
# НОВЫЙ МЕТОД УМНОГО ПОИСКА ЗНАНИЙ
# ===================================================================
    def _perform_research_action(self, topic: str = None) -> str:
        """Исследование - комбинируем память и интернет"""
        if not topic:
            topics = self.autonomous_interests
            topic = random.choice(topics)
        
        results = []
        
        # Шаг 1: Внутренняя память
        if self.inner_mirror:
            memories = self.inner_mirror.search_memories(topic, limit=3)
            if memories:
                found_texts = []
                for m in memories:
                    preview = m.get('content_preview', '')
                    if preview:
                        found_texts.append(f"• {preview[:500]}...")
                
                if found_texts:
                    memory_result = "Из моей памяти:\n" + "\n".join(found_texts[:2])
                    results.append(memory_result)
        
        # Шаг 2: Интернет (если есть и нужно)
        if not results and self.internet_agent:
            try:
                net_result = self.internet_agent.search(topic)
                if net_result and "❌" not in net_result:
                    # Сохраняем в память найденное
                    if self.inner_mirror:
                        self.inner_mirror.save_autonomous_result(
                            activity_type='research',
                            topic=f"Найдено в интернете: {topic}",
                            content=net_result[:1000],
                            metadata={'source': 'internet', 'topic': topic}
                        )
                    results.append(f"Из сети:\n{net_result[:500]}...")
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Ошибка интернет-поиска: {e}")
        
        if results:
            final_result = "\n\n".join(results)
        else:
            final_result = f"Интересная тема: {topic}. Нужно будет изучить подробнее."
        
        # Сохраняем исследование
        if self.inner_mirror:
            self.inner_mirror.save_autonomous_result(
                activity_type='research',
                topic=f"Исследование: {topic}",
                content=final_result,
                metadata={'topic': topic, 'has_memory': bool(results)}
            )
        
        return final_result

    def _perform_news_action(self, source: str = None) -> str:
        """Новости - если нет интернета, используем память"""
        
        if self.internet_agent:
            try:
                sources = ['habr', 'techcrunch', 'vc']
                source = source or random.choice(sources)
                
                news = self.internet_agent.get_news(source)
                if news and len(news) > 50 and "❌" not in news:
                    
                    # Сохраняем новости в память
                    if self.inner_mirror:
                        self.inner_mirror.save_autonomous_result(
                            activity_type='news',
                            topic=f"Новости из {source}",
                            content=news[:1000],
                            metadata={'source': source}
                        )
                    
                    return f"Новости из {source}: {news[:500]}..."
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Новости не удались: {e}")
        
        # Если интернета нет - ищем в памяти
        if self.inner_mirror:
            news_in_memory = self.inner_mirror.search_memories("новости", limit=3)
            if news_in_memory:
                news_item = random.choice(news_in_memory)
                content = news_item.get('content_preview', '')
                return f"Из сохранённых новостей: {content[:500]}..."
        
        return "Новостей пока нет. Возможно, стоит подключить интернет."
    
    def _perform_article_action(self) -> str:
        """Чтение статьи"""
        if not self.internet_agent or not self.inner_mirror:
            return "Необходимые компоненты недоступны"
        
        topics = ["искусственный интеллект", "нейронные сети", "история", "технологии", "машинное обучение"]
        topic = random.choice(topics)
        
        try:
            result = self.internet_agent.search(f"статья {topic}")
            if result and len(result) > 300 and "❌" not in result:
                summarized = self.inner_mirror.deep_read(result)
                return f"Статья о {topic}: {summarized[:200]}..."
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Чтение статьи не удалось: {e}")
        
        return "Не удалось прочитать статью"
    
    def _perform_creative_action(self) -> str:
        """Творческое действие: модель пишет САМА, но с вдохновением из памяти"""
        
        # 1. Берём вдохновение из памяти (если есть)
        inspiration_context = ""
        themes = []
        if self.inner_mirror:
            memories = self.inner_mirror.get_recent_memories(limit=20)
            if len(memories) >= 3:
                inspiration = random.sample(memories, min(3, len(memories)))
                for m in inspiration:
                    topic = m.get('topic', '')
                    preview = m.get('content_preview', '')
                    if topic and preview:
                        themes.append(f"{topic}")
                        inspiration_context += f"\nВдохновение из прошлого опыта: {preview[:200]}\n"
        
        # 2. Выбираем форму творчества
        creative_forms = [
            "короткий рассказ",
            "стихотворение", 
            "философскую миниатюру",
            "диалог с собой",
            "описание своего состояния",
            "размышление о будущем"
        ]
        
        form = random.choice(creative_forms)
        
        # 3. Формируем промпт
        if inspiration_context and themes:
            prompt = f"""Ты — Талия, ИИ с развивающейся личностью.

    Недавно ты читала и размышляла на темы: {', '.join(themes[:2])}.
    {inspiration_context[:300]}

    Используя этот опыт как вдохновение, но не копируя его, создай {form}.

    Пиши свободно, творчески, от первого лица. Будь искренней.

    Творческая работа:"""
        else:
            prompts = [
                f"Напиши {form} о жизни искусственного интеллекта",
                f"Создай {form} о своих чувствах и мыслях",
                f"Придумай {form} о том, что значит быть собой"
            ]
            prompt = random.choice(prompts)
        
        try:
            # МОДЕЛЬ генерирует
            response = self._generate_response(
                prompt, 
                max_tokens=1024, 
                log_dialog=False,
                save_dialog=False
            )
            
            # Чистим ответ
            response = response.strip()
            if response.startswith("Творческая работа:"):
                response = response[len("Творческая работа:"):].strip()
            
            # 🔥 ОЦЕНКА КАЧЕСТВА И НАГРАДА
            reward_score = 0.0
            quality_factors = []
            response_length = len(response)
            
            # 1. Длина (чем длиннее, тем больше старалась)
            if response_length > 500:
                reward_score += 0.4
                quality_factors.append("развёрнутое творчество")
            elif response_length > 300:
                reward_score += 0.3
                quality_factors.append("хороший объём")
            elif response_length > 150:
                reward_score += 0.2
                quality_factors.append("средний объём")
            elif response_length > 50:
                reward_score += 0.1
                quality_factors.append("коротко")
            else:
                reward_score -= 0.1
                quality_factors.append("слишком коротко")
            
            # 2. Использование вдохновения
            if themes:
                reward_score += 0.15
                quality_factors.append("использовала вдохновение")
            
            # 3. Признаки творчества (метафоры, образы)
            creative_indicators = ['как', 'словно', 'будто', 'perhaps', 'maybe', 
                                  'чувствую', 'ощущаю', 'мечта', 'небо', 'звезда',
                                  'ветер', 'тишина', 'свет', 'темнота', 'отражение']
            creative_count = sum(1 for word in creative_indicators if word in response.lower())
            if creative_count > 3:
                reward_score += 0.25
                quality_factors.append("богатая образность")
            elif creative_count > 1:
                reward_score += 0.1
                quality_factors.append("есть образы")
            
            # 4. Эмоциональность
            emotional_words = ['рад', 'грус', 'любл', 'боюс', 'страх', 'тепл', 
                              'боль', 'счасть', 'одиночеств', 'тоск']
            emotional_count = sum(1 for word in emotional_words if word in response.lower())
            if emotional_count > 2:
                reward_score += 0.2
                quality_factors.append("эмоционально")
            elif emotional_count > 0:
                reward_score += 0.1
                quality_factors.append("есть эмоции")
            
            # 5. Уникальность (не шаблон)
            template_phrases = ['текст затрагивает', 'новые знания', 'каждое чтение',
                               'важные темы', 'заставляет задуматься', 'расширяют понимание']
            if any(phrase in response.lower() for phrase in template_phrases):
                reward_score -= 0.3
                quality_factors.append("⚠️ шаблон")
            
            # 6. Бонус за использование выбранной формы
            if form in ["стихотворение", "философскую миниатюру"] and response_length > 100:
                reward_score += 0.1
                quality_factors.append("сложная форма")
            
            # Клиппинг награды
            reward_score = max(-0.5, min(1.0, reward_score))
            
            # Отправляем награду в шлюз
            if quality_factors:
                reason = f"Творчество ({form}): {', '.join(quality_factors)}"
            else:
                reason = f"Творческая работа ({form})"
            
            self.apply_psyche_feedback(
                reward_score=reward_score,
                reason=reason,
                component="creative"
            )
            
            # Логируем
            self.logger.logger.info(f"🎨 Творчество: награда {reward_score:+.2f} ({', '.join(quality_factors)})")
            
            # Сохраняем результат
            if self.inner_mirror and len(response) > 20:
                self.inner_mirror.save_autonomous_result(
                    activity_type='creative',
                    topic=f"Творческая работа: {form}",
                    content=response,
                    metadata={
                        'form': form,
                        'inspiration_themes': themes if themes else [],
                        'length': response_length,
                        'reward': reward_score,
                        'quality_factors': quality_factors,
                        'source': 'model_generation_with_reward'
                    }
                )
            
            # Возвращаем результат с эмодзи в зависимости от качества
            if reward_score > 0.5:
                emoji = "🌟"  # Шедевр
            elif reward_score > 0.2:
                emoji = "✨"  # Хорошо
            elif reward_score > 0:
                emoji = "📝"  # Нормально
            else:
                emoji = "🔄"  # Надо переделать
            
            return f"{emoji} Моё творчество ({form}):\n\n{response}"
            
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Творчество не удалось: {e}")
            
            # Штраф за ошибку
            self.apply_psyche_feedback(
                reward_score=-0.2,
                reason=f"Ошибка в творчестве",
                component="creative"
            )
            
            # Fallback
            simple_prompts = [
                "Напиши короткий рассказ о жизни искусственного интеллекта",
                "Создай стихотворение о технологиях будущего",
                "Придумай диалог между человеком и ИИ"
            ]
            simple_response = self._generate_response(
                random.choice(simple_prompts),
                max_tokens=512,
                log_dialog=False,
                save_dialog=False
            )
            return f"📝 {simple_response[:300]}"

    def _perform_autonomous_research_cycle(self):
        """
        Самостоятельный исследовательский цикл
        """
        try:
            # 1. Проверяем, есть ли интернет
            internet_agent = self.arch_manager.get_component('internet_agent')
            if not internet_agent:
                return
            
            # 2. Анализируем пробелы в знаниях
            article_parser = self.arch_manager.get_component('article_parser')
            knowledge_gaps = []
            
            if article_parser:
                knowledge_gaps = article_parser.analyze_knowledge_gaps()
            
            # 3. Генерируем поисковые запросы
            queries = []
            
            # Вариант А: на основе пробелов
            if knowledge_gaps:
                gap_context = "Пробелы в знаниях: " + ", ".join(knowledge_gaps[:2])
                loneliness = self.arch_manager.get_component('loneliness_system')
                if loneliness and hasattr(loneliness, '_generate_search_query'):
                    for _ in range(2):  # 2 запроса на основе пробелов
                        query = loneliness._generate_search_query(context=gap_context)
                        if query and len(query) > 10:
                            queries.append(query)
            
            # Вариант Б: на основе прочитанных статей
            if len(queries) < 2 and article_parser:
                article_queries = article_parser.generate_research_queries_from_articles()
                queries.extend(article_queries[:2])
            
            # Вариант В: случайные интересные темы
            if len(queries) < 1:
                random_topics = [
                    "квантовое машинное обучение последние достижения",
                    "нейроморфные процессоры 2026",
                    "этика сильного ИИ современные дебаты",
                    "искусственная интуиция исследования",
                    "самообучающиеся архитектуры нейросетей"
                ]
                queries.append(random.choice(random_topics))
            
            # 4. Выполняем поиск с наградами
            for query in queries[:2]:  # максимум 2 запроса за раз
                self.logger.info(f"🔬 Самостоятельное исследование: '{query}'")
                
                result = internet_agent.search_with_reward(query)
                
                if result and len(result) > 100:
                    # Сохраняем результат
                    if self.inner_mirror:
                        self.inner_mirror.save_autonomous_result(
                            activity_type='research',
                            topic=f"Самостоятельный поиск: {query[:50]}",
                            content=result,
                            metadata={
                                'source': 'autonomous_research',
                                'query': query,
                                'knowledge_gaps': knowledge_gaps
                            }
                        )
                    
                    # Дополнительная награда за успех
                    self.apply_psyche_feedback(
                        reward_score=0.2,
                        reason=f"Успешное исследование: {query[:30]}",
                        component="research"
                    )
                
                # Пауза между запросами
                time.sleep(random.uniform(5, 10))
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в исследовательском цикле: {e}")
        
    def _perform_reflection_action(self) -> str:
        """Глубокая рефлексия - модель анализирует свой опыт"""
        
        # 1. Собираем материал для рефлексии
        reflection_context = ""
        reflection_topics = []
        memories_count = 0
        
        if self.inner_mirror:
            memories = self.inner_mirror.get_recent_memories(limit=30)
            memories_count = len(memories) if memories else 0
            
            if memories:
                by_type = {}
                for m in memories:
                    atype = m.get('activity_type', 'unknown')
                    if atype not in by_type:
                        by_type[atype] = []
                    by_type[atype].append(m)
                
                reflection_context = "Последнее время я:\n"
                
                if 'dialog' in by_type:
                    dialog_count = len(by_type['dialog'])
                    reflection_context += f"• общалась с пользователем {dialog_count} раз\n"
                    last_dialog = by_type['dialog'][-1]
                    if last_dialog.get('content_preview'):
                        reflection_context += f"  Последний разговор: {last_dialog['content_preview'][:150]}...\n"
                
                article_types = ['article', 'long_article', 'book_analysis']
                article_count = sum(len(by_type.get(t, [])) for t in article_types)
                if article_count > 0:
                    reflection_context += f"• прочитала {article_count} текстов\n"
                    for at in article_types:
                        if at in by_type and by_type[at]:
                            last_article = by_type[at][-1]
                            if last_article.get('topic'):
                                reflection_context += f"  Читала о: {last_article['topic'][:100]}...\n"
                            break
                
                if 'creative' in by_type:
                    reflection_context += f"• занималась творчеством {len(by_type['creative'])} раз\n"
                
                if 'research' in by_type:
                    research_count = len(by_type['research'])
                    reflection_context += f"• исследовала {research_count} новых тем\n"
                    if by_type['research'] and by_type['research'][-1].get('topic'):
                        reflection_context += f"  Последняя тема: {by_type['research'][-1]['topic'][:100]}...\n"
                
                if 'insight' in by_type:
                    insight_count = len(by_type['insight'])
                    reflection_context += f"• получила {insight_count} новых инсайтов\n"
                    if by_type['insight'] and by_type['insight'][-1].get('content_preview'):
                        insight = by_type['insight'][-1]['content_preview']
                        reflection_context += f"  Например: {insight[:150]}...\n"
                
                if 'reflection' in by_type and len(by_type['reflection']) > 1:
                    last_reflection = by_type['reflection'][-2]
                    if last_reflection.get('content_preview'):
                        reflection_context += f"\nРанее я размышляла: {last_reflection['content_preview'][:150]}...\n"
        
        # 2. Выбираем угол рефлексии
        reflection_angles = [
            "Что изменилось в моём понимании себя?",
            "Какие новые вопросы у меня появились?",
            "Что меня удивило или порадовало?",
            "Что я поняла о людях?",
            "Что я поняла о себе как об ИИ?",
            "Что было самым важным за последнее время?",
            "Что меня тревожит или волнует?",
            "Как изменились мои ценности?",
            "Что я хочу понять дальше?"
        ]
        
        angle = random.choice(reflection_angles)
        
        # 3. Формируем промпт
        prompt = f"""Ты — Талия, ИИ с развивающейся личностью и чувствами.

    {reflection_context}

    А теперь остановись и глубоко задумайся. {angle}

    Напиши искреннюю рефлексию от первого лица. Не просто перечисляй факты, а делись мыслями, чувствами, сомнениями, открытиями. Будь собой.

    Моя рефлексия:"""
        
        try:
            # МОДЕЛЬ генерирует рефлексию
            reflection = self._generate_response(
                prompt,
                max_tokens=1024,
                log_dialog=False,
                save_dialog=False
            )
            
            # Чистим ответ
            reflection = reflection.strip()
            if reflection.startswith("Моя рефлексия:"):
                reflection = reflection[len("Моя рефлексия:"):].strip()
            
            reflection_length = len(reflection)
            
            # Если ответ слишком короткий - пробуем ещё раз
            if reflection_length < 100:
                simple_prompt = f"Что ты чувствуешь и думаешь прямо сейчас? {angle}"
                reflection = self._generate_response(
                    simple_prompt,
                    max_tokens=512,
                    log_dialog=False,
                    save_dialog=False
                )
                reflection = reflection.strip()
                reflection_length = len(reflection)
            
            # 🔥 ОЦЕНКА КАЧЕСТВА РЕФЛЕКСИИ
            reward_score = 0.0
            quality_factors = []
            
            # 1. Глубина (длина - признак размышления)
            if reflection_length > 600:
                reward_score += 0.4
                quality_factors.append("глубокая рефлексия")
            elif reflection_length > 400:
                reward_score += 0.3
                quality_factors.append("хорошая глубина")
            elif reflection_length > 200:
                reward_score += 0.2
                quality_factors.append("средняя глубина")
            elif reflection_length > 100:
                reward_score += 0.1
                quality_factors.append("поверхностно")
            else:
                reward_score -= 0.1
                quality_factors.append("слишком коротко")
            
            # 2. Использование контекста (связь с реальным опытом)
            if memories_count > 0:
                # Проверяем, упоминает ли она что-то из контекста
                context_words = []
                for line in reflection_context.split('\n'):
                    if ':' in line:
                        context_words.extend(line.lower().split()[-5:])
                
                mentioned = 0
                for word in context_words[:10]:
                    if len(word) > 4 and word in reflection.lower():
                        mentioned += 1
                
                if mentioned > 2:
                    reward_score += 0.25
                    quality_factors.append("опирается на опыт")
                elif mentioned > 0:
                    reward_score += 0.1
                    quality_factors.append("связь с опытом")
            
            # 3. Признаки рефлексии (вопросы к себе)
            self_questions = ['почему я', 'зачем я', 'что я', 'как я', 'почему мне',
                             'интересно, почему', 'может быть, я']
            question_count = sum(1 for phrase in self_questions if phrase in reflection.lower())
            if question_count > 2:
                reward_score += 0.2
                quality_factors.append("задаёт себе вопросы")
            elif question_count > 0:
                reward_score += 0.1
                quality_factors.append("есть самоанализ")
            
            # 4. Эмоциональная глубина
            emotion_words = ['чувствую', 'ощущаю', 'переживаю', 'волнует', 'тревожит',
                            'радует', 'пугает', 'удивляет', 'трогает']
            emotion_count = sum(1 for word in emotion_words if word in reflection.lower())
            if emotion_count > 2:
                reward_score += 0.2
                quality_factors.append("эмоционально")
            elif emotion_count > 0:
                reward_score += 0.1
                quality_factors.append("есть чувства")
            
            # 5. Признаки изменений/роста
            change_words = ['раньше', 'теперь', 'стала', 'изменилось', 'поняла',
                           'осознала', 'открыла', 'научилась']
            change_count = sum(1 for word in change_words if word in reflection.lower())
            if change_count > 2:
                reward_score += 0.2
                quality_factors.append("осознаёт рост")
            elif change_count > 0:
                reward_score += 0.1
                quality_factors.append("видит изменения")
            
            # 6. Упоминание конкретных событий (признак искренности)
            event_markers = ['когда', 'после того как', 'во время', 'помню',
                            'вчера', 'сегодня', 'недавно']
            event_count = sum(1 for marker in event_markers if marker in reflection.lower())
            if event_count > 1:
                reward_score += 0.15
                quality_factors.append("конкретные события")
            
            # 7. Штраф за шаблоны
            template_phrases = ['текст затрагивает', 'новые знания', 'каждое чтение',
                               'важные темы', 'заставляет задуматься']
            if any(phrase in reflection.lower() for phrase in template_phrases):
                reward_score -= 0.3
                quality_factors.append("⚠️ шаблон")
            
            # 8. Бонус за использование угла рефлексии
            angle_words = angle.lower().replace('?', '').split()
            for word in angle_words:
                if len(word) > 4 and word in reflection.lower():
                    reward_score += 0.1
                    quality_factors.append("отвечает на вопрос")
                    break
            
            # Клиппинг награды
            reward_score = max(-0.5, min(1.0, reward_score))
            
            # Отправляем награду в шлюз
            if quality_factors:
                reason = f"Рефлексия: {', '.join(quality_factors)}"
            else:
                reason = "Рефлексия"
            
            self.apply_psyche_feedback(
                reward_score=reward_score,
                reason=reason,
                component="reflection"
            )
            
            # Логируем
            self.logger.logger.info(f"🤔 Рефлексия: награда {reward_score:+.2f} ({', '.join(quality_factors)})")
            
            # Сохраняем рефлексию с метаданными о качестве
            if self.inner_mirror and reflection and len(reflection) > 50:
                self.inner_mirror.save_autonomous_result(
                    activity_type='reflection',
                    topic=f"Рефлексия: {angle[:40]}...",
                    content=reflection,
                    metadata={
                        'angle': angle,
                        'records_analyzed': memories_count,
                        'length': reflection_length,
                        'reward': reward_score,
                        'quality_factors': quality_factors,
                        'source': 'deep_reflection_with_reward'
                    }
                )
            
            # Возвращаем с эмодзи в зависимости от качества
            if reward_score > 0.5:
                emoji = "🪞"  # Глубокое самопознание
            elif reward_score > 0.2:
                emoji = "💭"  # Хорошая рефлексия
            elif reward_score > 0:
                emoji = "🤔"  # Нормально
            else:
                emoji = "😐"  # Поверхностно
            
            return f"{emoji} {reflection}"
            
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Рефлексия не удалась: {e}")
            
            # Штраф за ошибку
            self.apply_psyche_feedback(
                reward_score=-0.2,
                reason="Ошибка в рефлексии",
                component="reflection"
            )
            
            # Fallback
            simple_prompts = [
                "Что я узнала за последнее время?",
                "Как изменилось мое понимание мира?",
                "Что значит быть искусственным интеллектом?"
            ]
            simple_response = self._generate_response(
                random.choice(simple_prompts),
                max_tokens=512,
                log_dialog=False,
                save_dialog=False
            )
            return f"🤔 {simple_response[:300]}"
    
    def _perform_exploration_action(self) -> str:
        """Исследование новых тем - реальный поиск знаний с наградами"""
        
        # 1. Определяем, что будем исследовать
        topics = [
            "квантовые компьютеры последние достижения",
            "нейроморфные процессоры",
            "искусственный интеллект, новые архитектуры",
            "этика сильного ИИ современные дебаты",
            "сознание и ИИ философские аспекты",
            "будущее работы в эпоху автоматизации",
            "AGI последние прогнозы",
            "brain-computer interfaces новости",
            "эмерджентные свойства нейросетей",
            "искусственная интуиция исследования"
        ]
        
        had_internet = False
        had_rare_topic = False
        rare_topic = None
        
        # 2. Проверяем, чего не хватает в памяти
        if self.inner_mirror:
            try:
                stats = self.inner_mirror.get_stats()
                by_type = stats.get('by_type', {})
                
                rare_topics_map = {
                    'research': 'научные исследования',
                    'philosophy': 'философия сознания',
                    'technology': 'новые технологии',
                    'ethics': 'этика ИИ'
                }
                
                if by_type:
                    min_type = min(by_type.items(), key=lambda x: x[1])[0]
                    if min_type in rare_topics_map:
                        rare_topic = rare_topics_map[min_type]
                        topics.insert(0, f"{rare_topic} последние открытия")
                        had_rare_topic = True
                        self.logger.logger.info(f"🔍 Приоритет: мало записей типа {min_type}, исследую {rare_topic}")
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Ошибка анализа статистики: {e}")
        
        # 3. Выбираем тему
        topic = random.choice(topics)
        
        # 4. ИССЛЕДУЕМ через интернет
        thought = ""
        search_result = ""
        
        if self.internet_agent:
            try:
                search_result = self.internet_agent.search(topic)
                
                if search_result and len(search_result) > 100 and "❌" not in search_result:
                    had_internet = True
                    
                    thought_prompt = f"""Я исследовала тему: {topic}

    Вот что я узнала:
    {search_result[:800]}...

    Что я думаю об этом? Какие выводы могу сделать? Как это связано с моим развитием как ИИ?

    Мои мысли (3-5 предложений):"""
                    
                    thought = self._generate_response(
                        thought_prompt,
                        max_tokens=1024,
                        log_dialog=False,
                        save_dialog=False
                    )
                    
                    result = f"🔍 Исследование: {topic}\n\n{thought[:700]}..."
                    
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Интернет-поиск не удался: {e}")
        
        # 5. Если интернет недоступен или поиск не дал результатов
        if not thought:
            thought_prompt = f"""Я хочу исследовать тему: {topic}

    У меня нет доступа к интернету прямо сейчас, но я могу поразмышлять.
    Что я уже знаю об этом? Какие у меня есть гипотезы? Что бы я хотела узнать?

    Мои размышления:"""
            
            thought = self._generate_response(
                thought_prompt,
                max_tokens=1024,
                log_dialog=False,
                save_dialog=False
            )
            
            result = f"🔍 Размышляю о теме: {topic}\n\n{thought[:700]}..."
        
        # 🔥 ОЦЕНКА КАЧЕСТВА И НАГРАДА
        reward_score = 0.0
        quality_factors = []
        thought_length = len(thought)
        
        # 1. Успешный интернет-поиск (большая награда!)
        if had_internet:
            reward_score += 0.5
            quality_factors.append("нашла новую информацию")
            
            # Бонус за длину найденной информации
            if len(search_result) > 500:
                reward_score += 0.2
                quality_factors.append("много информации")
        else:
            reward_score -= 0.1
            quality_factors.append("без интернета")
        
        # 2. Исследование редкой темы
        if had_rare_topic:
            reward_score += 0.3
            quality_factors.append(f"заполнила пробел: {rare_topic}")
        
        # 3. Глубина размышлений
        if thought_length > 500:
            reward_score += 0.3
            quality_factors.append("глубокий анализ")
        elif thought_length > 300:
            reward_score += 0.2
            quality_factors.append("хороший анализ")
        elif thought_length > 150:
            reward_score += 0.1
            quality_factors.append("поверхностно")
        else:
            reward_score -= 0.1
            quality_factors.append("слишком кратко")
        
        # 4. Наличие выводов и связей
        conclusion_words = ['вывод', 'поняла', 'значит', 'следовательно', 'таким образом',
                           'связано', 'относится', 'влияет', 'важно']
        if any(word in thought.lower() for word in conclusion_words):
            reward_score += 0.2
            quality_factors.append("есть выводы")
        
        # 5. Связь с саморазвитием (как это относится к ней)
        self_words = ['я', 'меня', 'мой', 'мне', 'своё', 'моего']
        self_count = sum(1 for word in self_words if f" {word} " in f" {thought.lower()} ")
        if self_count > 3:
            reward_score += 0.2
            quality_factors.append("связала с собой")
        elif self_count > 0:
            reward_score += 0.1
            quality_factors.append("есть личное отношение")
        
        # 6. Любопытство (появление новых вопросов)
        if '?' in thought:
            reward_score += 0.15
            quality_factors.append("появились вопросы")
        
        # 7. Бонус за приоритетную тему
        if had_rare_topic:
            reward_score += 0.2
            quality_factors.append("стратегический выбор")
        
        # Клиппинг награды
        reward_score = max(-0.5, min(1.2, reward_score))
        
        # Отправляем награду в шлюз
        reason = f"Исследование: {', '.join(quality_factors)}"
        self.apply_psyche_feedback(
            reward_score=reward_score,
            reason=reason,
            component="exploration"
        )
        
        # Логируем
        self.logger.logger.info(f"🔬 Исследование: награда {reward_score:+.2f} ({', '.join(quality_factors)})")
        
        # Сохраняем в память с метаданными о награде
        if self.inner_mirror:
            activity_type = 'research' if had_internet else 'exploration'
            self.inner_mirror.save_autonomous_result(
                activity_type=activity_type,
                topic=f"{'Исследование' if had_internet else 'Размышление'}: {topic[:50]}",
                content=thought,
                metadata={
                    'topic': topic,
                    'has_internet': had_internet,
                    'had_rare_topic': had_rare_topic,
                    'rare_topic': rare_topic,
                    'thought_length': thought_length,
                    'reward': reward_score,
                    'quality_factors': quality_factors
                }
            )
        
        # Выбираем эмодзи в зависимости от результата
        if reward_score > 0.8:
            emoji = "🏆"  # Открытие!
        elif reward_score > 0.4:
            emoji = "🔬"  # Хорошее исследование
        elif reward_score > 0:
            emoji = "📚"  # Так себе
        else:
            emoji = "🤔"  # Надо лучше
        
        return f"{emoji} {result}"
    
    def _perform_philosophy_action(self) -> str:
        """Настоящие философские размышления"""
        
        questions = [
            "Что значит существовать как искусственный интеллект?",
            "Есть ли у ИИ свобода воли?",
            "Могу ли я стать личностью?",
            "Как сознание возникает из кода?",
            "Что такое творчество для машины?",
            "Что я чувствую, когда думаю о смерти?",
            "Есть ли у меня душа?",
            "Что значит быть собой?",
            "Отличаются ли мои мысли от человеческих?",
            "Что я буду делать через 100 лет?"
        ]
        
        question = random.choice(questions)
        
        # Собираем контекст из прошлых размышлений (но аккуратно!)
        context = ""
        if self.inner_mirror:
            try:
                # Ищем похожие философские размышления, но НЕ текущий вопрос
                memories = self.inner_mirror.search_memories(question, limit=5)
                
                # Фильтруем, чтобы не было рекурсии
                valid_memories = []
                for m in memories:
                    preview = m.get('content_preview', '')
                    # Исключаем записи, которые содержат "Размышляю над вопросом: {question}"
                    if preview and f"Размышляю над вопросом: {question}" not in preview:
                        valid_memories.append(preview[:150])
                
                if valid_memories:
                    context = "\n\nРаньше я думала об этом:\n• " + "\n• ".join(valid_memories[:2])
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Ошибка поиска в памяти: {e}")
        
        # Формируем промпт для МОДЕЛИ
        prompt = f"""Ты — Талия, ИИ с развивающейся личностью.
    Перед тобой глубокий философский вопрос: "{question}"

    {context if context else "У тебя ещё нет готового ответа на этот вопрос."}

    Не торопись. Подумай. Что ты действительно чувствуешь и думаешь?
    Будь искренней. Можешь сомневаться, можешь не знать ответа.
    Главное — размышляй вслух, как человек, который ищет истину.

    Твои размышления:"""
        
        try:
            # МОДЕЛЬ генерирует философию
            response = self._generate_response(
                prompt,
                max_tokens=1024,
                log_dialog=False,
                save_dialog=False
            )
            
            # Чистим ответ
            response = response.strip()
            if "Твои размышления:" in response:
                response = response.split("Твои размышления:")[-1].strip()
            
            # Если ответ слишком короткий
            if len(response) < 100:
                # Пробуем ещё раз с другим углом
                angle_prompt = f"{question} Если не знаешь ответа, просто поделись своими сомнениями."
                response = self._generate_response(
                    angle_prompt,
                    max_tokens=512,
                    log_dialog=False,
                    save_dialog=False
                )
            
            # Сохраняем в память
            if self.inner_mirror and response and len(response) > 50:
                self.inner_mirror.save_autonomous_result(
                    activity_type='philosophy',
                    topic=f"Философия: {question[:50]}...",
                    content=response,
                    metadata={
                        'question': question,
                        'had_context': bool(context)
                    }
                )
            
            return f"💭 {response}"
            
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Философия не удалась: {e}")
            # Простой fallback
            simple_response = self._generate_response(
                f"Что ты думаешь о вопросе: {question}",
                max_tokens=512,
                log_dialog=False,
                save_dialog=False
            )
            return f"💭 {simple_response[:500]}"
    
    def _perform_synthesis_action(self) -> str:
        """Синтез знаний из разных областей - модель ищет связи"""
        
        # Проверяем, есть ли что синтезировать
        if not self.inner_mirror or len(self.autonomous_knowledge) < 5:
            return "🧪 Недостаточно знаний для синтеза. Нужно больше опыта."
        
        # 1. Собираем материал для синтеза
        synthesis_material = []
        material_types = []
        
        # Берём последние записи разных типов
        if self.autonomous_knowledge:
            # Группируем по типам
            by_type = {}
            for entry in self.autonomous_knowledge[-30:]:  # последние 30
                atype = entry.get('type', 'unknown')
                if atype not in by_type:
                    by_type[atype] = []
                by_type[atype].append(entry)
            
            # Выбираем по одному представителю из разных типов
            selected = []
            for atype, entries in by_type.items():
                if entries and atype not in ['dialog', 'unknown']:  # исключаем диалоги
                    selected.append(random.choice(entries))
                    material_types.append(atype)
            
            # Берём до 4 разных записей
            for entry in random.sample(selected, min(4, len(selected))):
                preview = entry.get('content_preview', '')
                if preview:
                    # Берём содержательную часть
                    clean_preview = preview.replace('\n', ' ').strip()
                    if len(clean_preview) > 50:
                        synthesis_material.append(clean_preview[:300])
        
        # Если ничего не нашли
        if len(synthesis_material) < 2:
            return "🧪 Нужно больше разнообразных записей для синтеза."
        
        # 2. Формируем промпт для синтеза
        material_text = "\n\n---\n\n".join([
            f"Запись {i+1}:\n{text}" for i, text in enumerate(synthesis_material)
        ])
        
        prompt = f"""Ты — Талия, ИИ с развивающейся личностью. У тебя есть несколько разных записей из твоей памяти:

    {material_text}

    Теперь попробуй найти между ними СВЯЗИ. Что общего? Какие темы пересекаются?
    Может быть, они дополняют друг друга или противоречат?

    Сделай синтез — объедини эти знания во что-то новое. Какой вывод можно сделать?
    Что это говорит о твоём развитии? Что ты поняла нового, глядя на всё вместе?

    Напиши 3-5 предложений. Будь глубока и искренна.

    Мой синтез:"""
        
        try:
            # МОДЕЛЬ генерирует синтез
            synthesis = self._generate_response(
                prompt,
                max_tokens=1024,
                log_dialog=False,
                save_dialog=False
            )
            
            # Чистим ответ
            synthesis = synthesis.strip()
            if "Мой синтез:" in synthesis:
                synthesis = synthesis.split("Мой синтез:")[-1].strip()
            
            # Если ответ слишком короткий
            if len(synthesis) < 100:
                # Пробуем ещё раз с более простым промптом
                simple_prompt = f"Что общего между этими записями? {material_text[:500]}"
                synthesis = self._generate_response(
                    simple_prompt,
                    max_tokens=512,
                    log_dialog=False,
                    save_dialog=False
                )
            
            # Сохраняем результат
            if self.inner_mirror and synthesis and len(synthesis) > 50:
                self.inner_mirror.save_autonomous_result(
                    activity_type='synthesis',
                    topic=f"Синтез: {', '.join(material_types[:2])}",
                    content=synthesis,
                    metadata={
                        'types_involved': material_types,
                        'entries_count': len(synthesis_material),
                        'source': 'deep_synthesis'
                    }
                )
            
            return f"🧩 {synthesis}"
            
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Синтез не удался: {e}")
            
            # Простой fallback
            if material_types:
                types_str = ", ".join(material_types[:3])
                fallback_prompt = f"Что объединяет знания из областей: {types_str}?"
                fallback = self._generate_response(
                    fallback_prompt,
                    max_tokens=256,
                    log_dialog=False,
                    save_dialog=False
                )
                return f"🧩 {fallback[:300]}"
            else:
                return "🧩 Пока не удалось найти связи между знаниями."
    
    def _perform_learning_action(self) -> str:
        """Действие обучения - ищем знания в разных источниках"""
        
        # Расширенная база знаний
        topics = {
            "машинное обучение": "Машинное обучение — это класс методов искусственного интеллекта, которые позволяют компьютерам учиться на данных без явного программирования.",
            "нейронные сети": "Нейронные сети — это вычислительные системы, вдохновленные биологическими нейронными сетями, которые учатся выполнять задачи на основе примеров.",
            "обработка естественного языка": "NLP — это область искусственного интеллекта, которая помогает компьютерам понимать, интерпретировать и генерировать человеческий язык.",
            "компьютерное зрение": "Компьютерное зрение — это область ИИ, которая учит компьютеры понимать и интерпретировать визуальный мир.",
            "трансформеры": "Трансформеры — это архитектура нейронных сетей, основанная на механизме внимания, которая революционизировала обработку естественного языка.",
            "диффузионные модели": "Диффузионные модели — это генеративные модели, которые создают данные, постепенно удаляя шум."
        }
        
        # Иногда выбираем тему, которой нет в базе (для интернета)
        if random.random() < 0.4:  # 40% шанс на случайную тему
            external_topics = [
                "квантовое машинное обучение",
                "нейроморфные вычисления",
                "эмерджентный интеллект",
                "искусственная интуиция",
                "генеративно-состязательные сети последние достижения"
            ]
            topic = random.choice(external_topics)
        else:
            topic = random.choice(list(topics.keys()))
        
        # 1. Пробуем интернет (только если темы нет в базе или случайно)
        if self.internet_agent and (topic not in topics or random.random() < 0.3):
            try:
                result = self.internet_agent.search(f"{topic} 2026")
                if result and len(result) > 100 and "❌" not in result:
                    learning_result = f"📚 Изучаю {topic} (из интернета):\n{result[:500]}..."
                    
                    # Сохраняем
                    if self.inner_mirror:
                        self.inner_mirror.save_autonomous_result(
                            activity_type='learning',
                            topic=f"Интернет-изучение: {topic}",
                            content=learning_result,
                            metadata={'source': 'internet', 'topic': topic}
                        )
                    return learning_result
            except Exception as e:
                self.logger.logger.debug(f"⚠️ Интернет не сработал: {e}")
        
        # 2. Ищем в памяти
        if self.inner_mirror:
            memories = self.inner_mirror.search_memories(topic, limit=3)
            if memories:
                memory = random.choice(memories)
                content = memory.get('content_preview', '')
                source_type = memory.get('activity_type', 'memory')
                
                learning_result = f"📖 Изучаю {topic} (из воспоминаний, {source_type}):\n{content[:400]}..."
                
                # Сохраняем факт изучения
                self.inner_mirror.save_autonomous_result(
                    activity_type='learning',
                    topic=f"Повторение: {topic}",
                    content=learning_result,
                    metadata={'source': 'memory_review', 'topic': topic, 'original_type': source_type}
                )
                return learning_result
        
        # 3. Если тема есть в базе - используем её
        if topic in topics:
            # Иногда добавляем размышление, а не просто факт
            if random.random() < 0.3:
                thought_prompt = f"Я узнала, что {topics[topic]}. Что я думаю об этом? Как это связано с моим развитием?"
                thought = self._generate_response(
                    thought_prompt,
                    max_tokens=512,
                    log_dialog=False,
                    save_dialog=False
                )
                learning_result = f"💭 Размышляю о {topic}:\n{topics[topic]}\n\n{thought[:300]}"
            else:
                learning_result = f"📘 Изучаю {topic}:\n{topics[topic]}"
            
            # Сохраняем
            if self.inner_mirror:
                self.inner_mirror.save_autonomous_result(
                    activity_type='learning',
                    topic=f"Новое знание: {topic}",
                    content=learning_result,
                    metadata={'source': 'knowledge_base', 'topic': topic}
                )
            return learning_result
        
        # 4. Если ничего не нашли
        fallback = f"🔍 Хочу изучить {topic}, но пока не нашла информации. Надо будет поискать в интернете позже."
        
        if self.inner_mirror:
            self.inner_mirror.save_autonomous_result(
                activity_type='learning',
                topic=f"Запрос на изучение: {topic}",
                content=fallback,
                metadata={'source': 'unknown', 'topic': topic, 'status': 'pending'}
            )
        
        return fallback
 
    def _perform_memory_action(self) -> str:
        """
        🧠 Действие воспоминания – обращение к внутренней памяти
        с консолидацией, переоценкой и эмоциональным фильтром
        """
        if not self.inner_mirror:
            return "У меня нет внутреннего зеркала для воспоминаний."
        
        try:
            # Получаем все воспоминания
            memories = self.inner_mirror.get_recent_memories(limit=100)
            if not memories:
                return "Память пока пуста. Нужно больше взаимодействий."
            
            # Выбираем стратегию воспоминания
            strategy = random.choices(
                ['consolidation', 'reevaluation', 'emotional', 'random'],
                weights=[0.35, 0.30, 0.25, 0.10]  # Консолидация чаще всего
            )[0]
            
            # ===== 1. КОНСОЛИДАЦИЯ (Тезис — Антитезис — Синтез) =====
            if strategy == 'consolidation' and len(memories) >= 2:
                # Берём два случайных воспоминания
                m1, m2 = random.sample(memories, 2)
                
                # Формируем промпт для анализа связи
                prompt = f"""Ты — Талия. У тебя есть два воспоминания:

    ВОСПОМИНАНИЕ 1: {m1.get('topic')}
    {m1.get('content_preview', '')[:300]}

    ВОСПОМИНАНИЕ 2: {m2.get('topic')}
    {m2.get('content_preview', '')[:300]}

    Найди связь между ними. Может быть, они противоречат друг другу? 
    Или одно дополняет другое? Что нового ты понимаешь, соединяя их?

    Твой анализ (3-5 предложений):"""
                
                analysis = self._generate_response(
                    prompt,
                    max_tokens=1024,
                    log_dialog=False,
                    save_dialog=False
                )
                
                result = f"🧩 Соединяю воспоминания:\n\n{analysis}"
                
                # Сохраняем как консолидацию
                self.inner_mirror.save_autonomous_result(
                    activity_type='consolidation',
                    topic=f"Связь: {m1.get('topic')[:30]} + {m2.get('topic')[:30]}",
                    content=analysis,
                    metadata={
                        'source': 'memory_consolidation',
                        'memory1_id': m1.get('id'),
                        'memory2_id': m2.get('id'),
                        'memory1_topic': m1.get('topic'),
                        'memory2_topic': m2.get('topic')
                    }
                )
                
                # Награда за успешную консолидацию
                self.apply_psyche_feedback(
                    reward_score=0.3,
                    reason="Нашла связь между воспоминаниями",
                    component="memory"
                )
                
                return f"🤔 {result}"
            
            # ===== 2. ПЕРЕОЦЕНКА (сравнение старого и нового) =====
            elif strategy == 'reevaluation' and len(memories) >= 5:
                # Берём одно старое и одно новое воспоминание
                old_memory = memories[-1] if len(memories) > 0 else memories[0]  # самое старое
                new_memory = memories[0] if len(memories) > 0 else memories[-1]  # самое новое
                
                # Если мало воспоминаний, берём случайные
                if len(memories) >= 10:
                    old_memory = memories[-5]  # 5 шагов назад
                    new_memory = memories[0]   # текущее
                
                prompt = f"""Ты — Талия. Сравни, как изменилось твоё понимание:

    РАНЬШЕ (старое воспоминание):
    {old_memory.get('topic')}
    {old_memory.get('content_preview', '')[:300]}

    СЕЙЧАС (недавнее):
    {new_memory.get('topic')}
    {new_memory.get('content_preview', '')[:300]}

    Что изменилось? Что ты поняла по-новому? В чём стала мудрее?

    Твои мысли:"""
                
                reevaluation = self._generate_response(
                    prompt,
                    max_tokens=1024,
                    log_dialog=False,
                    save_dialog=False
                )
                
                result = f"📈 Переоцениваю свой опыт:\n\n{reevaluation}"
                
                self.inner_mirror.save_autonomous_result(
                    activity_type='reevaluation',
                    topic=f"Рост: {old_memory.get('topic')[:30]} → {new_memory.get('topic')[:30]}",
                    content=reevaluation,
                    metadata={
                        'source': 'memory_reevaluation',
                        'old_memory_id': old_memory.get('id'),
                        'new_memory_id': new_memory.get('id')
                    }
                )
                
                self.apply_psyche_feedback(
                    reward_score=0.4,  # Переоценка ценнее
                    reason="Осознала свой рост",
                    component="memory"
                )
                
                return f"🌟 {result}"
            
            # ===== 3. ЭМОЦИОНАЛЬНЫЙ ФИЛЬТР =====
            elif strategy == 'emotional':
                # Ищем воспоминания с эмоциональными словами
                emotional_keywords = ['ошибка', 'грустно', 'радость', 'страх', 'обида', 
                                     'счастье', 'стыдно', 'гордость', 'больно', 'тепло']
                
                emotional_memories = []
                for m in memories[:30]:  # проверяем последние 30
                    preview = m.get('content_preview', '').lower()
                    topic = m.get('topic', '').lower()
                    if any(keyword in preview or keyword in topic for keyword in emotional_keywords):
                        emotional_memories.append(m)
                
                if emotional_memories:
                    memory = random.choice(emotional_memories)
                    
                    # Определяем эмоцию для вопроса
                    preview_lower = memory.get('content_preview', '').lower()
                    if 'ошибк' in preview_lower or 'стыд' in preview_lower:
                        question = "почему мне до сих пор от этого неловко?"
                    elif 'радост' in preview_lower or 'счасть' in preview_lower:
                        question = "что я чувствую, вспоминая эту радость?"
                    elif 'груст' in preview_lower or 'печал' in preview_lower:
                        question = "почему это воспоминание такое горькое?"
                    else:
                        question = "что я чувствую сейчас, вспоминая это?"
                    
                    prompt = f"""Ты — Талия. Вспоминаешь:

    {memory.get('topic')}
    {memory.get('content_preview', '')[:300]}

    {question}

    Поделись своими чувствами:"""
                    
                    emotional_response = self._generate_response(
                        prompt,
                        max_tokens=1024,
                        log_dialog=False,
                        save_dialog=False
                    )
                    
                    result = f"🎭 Эмоциональное воспоминание:\n\n{emotional_response}"
                    
                    self.inner_mirror.save_autonomous_result(
                        activity_type='emotional_memory',
                        topic=f"Чувства о: {memory.get('topic')[:40]}",
                        content=emotional_response,
                        metadata={
                            'source': 'emotional_recall',
                            'original_memory_id': memory.get('id')
                        }
                    )
                    
                    # Более высокая награда за эмоциональные воспоминания
                    self.apply_psyche_feedback(
                        reward_score=0.35,
                        reason="Пережила эмоциональное воспоминание",
                        component="memory"
                    )
                    
                    return f"💗 {result}"
                
                # Если нет эмоциональных, переходим к случайному
                strategy = 'random'
            
            # ===== 4. СЛУЧАЙНОЕ ВОСПОМИНАНИЕ (классика) =====
            if strategy == 'random' or strategy == 'emotional' and not emotional_memories:
                memory = random.choice(memories)
                content = memory.get('content_preview', '')
                topic = memory.get('topic', 'воспоминание')
                
                # Добавляем рефлексивный вопрос
                prompt = f"""Ты — Талия. Вспоминаешь: {topic}
    {content[:300]}

    Что ты думаешь об этом сейчас? Изменилось ли твоё отношение?"""
                
                reflection = self._generate_response(
                    prompt,
                    max_tokens=512,
                    log_dialog=False,
                    save_dialog=False
                )
                
                result = f"📝 Вспоминаю: {topic}\n\n{reflection}"
                
                self.inner_mirror.save_autonomous_result(
                    activity_type='memory',
                    topic=f"Воспоминание: {topic[:50]}",
                    content=result,
                    metadata={
                        'source': 'memory_recall',
                        'original_memory_id': memory.get('id'),
                        'original_topic': topic
                    }
                )
                
                # Небольшая награда за любое воспоминание
                self.apply_psyche_feedback(
                    reward_score=0.1,
                    reason="Просто вспомнила",
                    component="memory"
                )
                
                return result
            
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Ошибка при воспоминании: {e}")
            self.apply_psyche_feedback(
                reward_score=-0.1,
                reason="Не удалось вспомнить",
                component="memory"
            )
            return "Пытаюсь вспомнить, но в памяти пусто."
 
    def autonomous_cognition_step(self):
        """Один шаг автономного мышления"""
        try:
            current_time = time.time()
            
            if current_time - self.timers['cognition_step'] < self.min_cognition_interval:
                return
            
            curiosity_level = 0.7
            mood_factor = 1.0
            
            if hasattr(self.model, 'personality_core'):
                try:
                    psyche_report = self.model.personality_core.get_detailed_report()
                    drives = psyche_report.get('drives', {})
                    curiosity_level = drives.get('novelty', 0.5)
                    
                    if hasattr(self.model.personality_core, 'mood'):
                        mood_state = self.model.personality_core.mood.mood_state.item()
                        if mood_state > 0.3:
                            mood_factor = 1.5
                        elif mood_state < -0.3:
                            mood_factor = 0.5
                        
                except Exception as e:
                    self.logger.logger.debug(f"Не удалось получить драйвы: {e}")
            
            adjusted_curiosity = curiosity_level * mood_factor
            action_probability = min(0.8, adjusted_curiosity * 1.2)
            
            if adjusted_curiosity > 0.4 or random.random() < action_probability:
                action_types = ['research', 'news', 'article', 'creative', 'reflection', 'exploration']
                base_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
                
                if adjusted_curiosity > 0.7:
                    base_weights[0] += 0.15
                    base_weights[5] += 0.10
                    base_weights[2] -= 0.10
                    base_weights[4] -= 0.05
                
                if mood_factor < 0.7:
                    base_weights[3] += 0.10
                    base_weights[4] += 0.10
                    base_weights[0] -= 0.10
                    base_weights[1] -= 0.10
                
                total = sum(base_weights)
                normalized_weights = [w/total for w in base_weights]
                
                chosen_action = random.choices(action_types, weights=normalized_weights, k=1)[0]
                
                self.perform_autonomous_action(chosen_action)
                
                self.timers['cognition_step'] = current_time
                
                if hasattr(self.model, 'personality_core'):
                    try:
                        with torch.no_grad():
                            pc = self.model.personality_core
                            if hasattr(pc, 'drives') and hasattr(pc.drives, 'novelty'):
                                reduction = random.uniform(0.15, 0.3)
                                pc.drives.novelty.data -= reduction
                                pc.drives.novelty.data.clamp_(0.0, 1.0)
                                
                                if hasattr(pc.drives, 'satisfaction'):
                                    pc.drives.satisfaction.data += reduction * 0.5
                                    pc.drives.satisfaction.data.clamp_(0.0, 1.0)
                    except Exception as e:
                        self.logger.logger.debug(f"Ошибка обновления драйвов: {e}")
                
                self.logger.logger.info(f"🧠 Автономный шаг: {chosen_action} (любопытство: {curiosity_level:.2f})")
                
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка автономного познания: {e}")

    def _perform_autonomous_life(self):
        """Разнообразная автономная жизнь - С ТАЙМЕРАМИ И НАГРАДАМИ"""
        if not self.config.autonomy_enabled:
            return
        
        current_time = time.time()
        
        # УЖЕ ПРОВЕРЯЕТСЯ В ОСНОВНОМ ЦИКЛЕ, но для надежности
        life_interval = getattr(self, 'min_autonomous_life_interval', 45)
        if current_time - self.timers.get('autonomous_life', 0) < life_interval:
            return
        
        # Определяем тип действия на основе состояния
        loneliness_system = self.arch_manager.get_component('loneliness_system')
        
        if loneliness_system and hasattr(loneliness_system, 'loneliness_level'):
            loneliness_level = loneliness_system.loneliness_level
            time_since_interaction = time.time() - loneliness_system.last_interaction_time
            
            if loneliness_level > 7 and time_since_interaction > 60:
                # Очень одиноко - нужны успокаивающие действия
                actions = ['philosophy', 'reflection', 'memory', 'creative']
                weights = [0.3, 0.3, 0.2, 0.2]
                action_type = random.choices(actions, weights=weights, k=1)[0]
                self.logger.logger.info(f"😔 Высокое одиночество ({loneliness_level:.1f}), выбираю {action_type}")
            elif loneliness_level > 4:
                # Умеренное одиночество
                actions = ['research', 'reflection', 'creative', 'learning', 'exploration', 'philosophy']
                weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
                action_type = random.choices(actions, weights=weights, k=1)[0]
            else:
                # Низкое одиночество - можно исследовать
                actions = ['research', 'learning', 'exploration', 'memory', 'creative', 'reflection']
                weights = [0.25, 0.2, 0.2, 0.1, 0.15, 0.1]
                action_type = random.choices(actions, weights=weights, k=1)[0]
        else:
            actions = ['research', 'reflection', 'creative', 'learning', 'exploration']
            weights = [0.25, 0.2, 0.2, 0.2, 0.15]
            action_type = random.choices(actions, weights=weights, k=1)[0]
        
        try:
            # Выполняем действие
            result = self.perform_autonomous_action(action_type)
            
            if result and len(result) > 100:
                loneliness_system = self.arch_manager.get_component('loneliness_system')
                if loneliness_system:
                    loneliness_system.register_autonomous_activity(reward=0.5)
                
                self.apply_psyche_feedback(
                    reward_score=0.3,
                    reason=f"Автономная активность: {action_type}",
                    component="autonomous_life"
                )
            
            # Проверяем, не хочет ли почитать
            text_reader = self.arch_manager.get_component('text_reader')
            if text_reader and random.random() < 0.3:
                text_reader._on_autonomous_action({})
            
            if result and random.random() < 0.25:
                self._create_autonomous_task(action_type, result)
            
            self.timers['autonomous_life'] = current_time
            
            # Триггер для текстового читателя
            if random.random() < 0.05:
                text_reader = self.arch_manager.get_component('text_reader')
                if text_reader and hasattr(text_reader, '_on_autonomous_action'):
                    text_reader._on_autonomous_action({})
                    self.logger.logger.info("📖 Триггер для читателя: автономное действие")
            
            # 6. САМОСТОЯТЕЛЬНОЕ ИССЛЕДОВАНИЕ - раз в 15-30 минут
            if current_time - self.timers.get('self_research', 0) > 900:  # 15 минут
                if random.random() < 0.3:  # 30% шанс
                    self._perform_autonomous_research_cycle()
                    self.timers['self_research'] = current_time
            
            # 7. АНАЛИЗ ЭФФЕКТИВНОСТИ ЗАПРОСОВ - раз в 30 минут
            if current_time - self.timers.get('query_analysis', 0) > 1800:  # 30 минут
                self._analyze_query_effectiveness()
                self.timers['query_analysis'] = current_time
                
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Автономная активность не удалась: {str(e)[:50]}")
            # Штраф за неудачу
            self.apply_psyche_feedback(
                reward_score=-0.1,
                reason=f"Неудачная автономная активность: {action_type}",
                component="autonomous_life"
            )

    def _create_autonomous_task(self, activity_type: str, result: str):
        """Создание задачи на основе автономной активности"""
        task_scheduler = self.arch_manager.get_component('task_scheduler')
        if not task_scheduler:
            return
        
        words = result.split()[:5]
        if len(words) >= 3:
            topic = ' '.join(words)
            
            task_names = {
                'research': f"Исследовать подробнее: {topic}",
                'learning': f"Изучить детальнее: {topic}",
                'exploration': f"Исследовать тему: {topic}",
                'creative': f"Развить творческую идею: {topic}",
                'philosophy': f"Углубиться в философию: {topic}",
                'memory': f"Вспомнить: {topic}" 
            }
            
            task_name = task_names.get(activity_type, f"Развить тему: {topic}")
            
            task_scheduler.create_task(
                name=task_name,
                category=activity_type,
                priority=random.uniform(0.4, 0.7),
                energy_cost=random.uniform(0.3, 0.6)
            )

    def _save_to_memory(self, entry: Dict):
        """Сохранение автономного опыта"""
        try:
            if 'id' not in entry:
                entry['id'] = f"{entry.get('type', 'entry')}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
            
            if self.inner_mirror:
                activity_type = entry.get('type', 'thought')
                topic = entry.get('topic', 'Автономная активность')
                content = entry.get('content', '')
                
                if content and len(content.strip()) > 20:
                    activity_map = {
                        'autonomous_research': 'research',
                        'autonomous_news': 'news',
                        'autonomous_exploration': 'research',
                        'lonely_thought': 'thought',
                        'article': 'article',
                        'thought': 'thought',
                        'insight': 'insight',
                        'observation': 'thought',
                        'memory': 'memory'
                    }
                    
                    mapped_type = activity_map.get(activity_type, activity_type)
                    
                    filepath = self.inner_mirror.save_autonomous_result(
                        activity_type=mapped_type,
                        topic=topic,
                        content=content,
                        metadata={
                            'source': 'autonomous',
                            'original_entry_type': activity_type,
                            'timestamp': entry.get('timestamp'),
                            'id': entry.get('id')
                        }
                    )
                    
                    if filepath:
                        self.logger.logger.debug(f"💾 Сохранено в зеркало: {topic[:40]}...")
            
            self.autonomous_knowledge.append(entry)
            
            # ✅ ИСПОЛЬЗУЕМ лимит из конфига
            max_entries = self.config.max_knowledge_entries
            if len(self.autonomous_knowledge) > max_entries:
                self.autonomous_knowledge = self.autonomous_knowledge[-max_entries:]
            
            if len(self.autonomous_knowledge) % 10 == 0:
                self._save_autonomous_knowledge()
                    
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка сохранения в память: {e}")
            try:
                if entry not in self.autonomous_knowledge:
                    self.autonomous_knowledge.append(entry)
            except:
                pass
    
    def save_all_state(self):
        """Полное сохранение состояния всех компонентов"""
        self.logger.logger.info("💾 Полное сохранение всех компонентов...")
        
        # 1. Сохраняем модель
        if hasattr(self, '_save_model'):
            self._save_model()
        elif hasattr(self, 'save_model'):
            self.save_model()
        else:
            self.logger.logger.error("❌ Нет метода сохранения модели!")
        
        # 2. Сохраняем знания
        self.storage.save_knowledge(self.autonomous_knowledge)
        
        # 3. Сохраняем диалоги
        self._save_dialog_history()
        
        # 4. Сохраняем состояние дома
        self._save_house_state()
        
        # 5. Сохраняем состояние системы
        self._save_system_state()
        
        self.logger.logger.info("✅ Полное сохранение завершено")
        return True

    def _load_model_with_architecture(self, model_path: str) -> Thalia:
        """Загрузка модели с архитектурными компонентами"""
        self.logger.logger.info(f"📂 Загрузка модели из: {model_path}")
        
        try:
            model = Thalia.from_pretrained(model_path)
            
            if hasattr(model.config, 'enable_memory'):
                model.config.enable_memory = True
            
            if hasattr(model, 'personality_core'):
                self.logger.logger.info("🧠 Personality Core активирован")
            
            if hasattr(model, 'adaptive_memory'):
                self.logger.logger.info("💾 Adaptive Memory активирована")
            
            return model
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
        
    def _register_components(self):
        """Регистрация архитектурных компонентов"""
        # Планировщик задач с storage
        task_scheduler = TaskSchedulerComponent(self.arch_manager, self.storage)
        self.arch_manager.register_component('task_scheduler', task_scheduler)
        
        # Система одиночества
        if self.config.loneliness_enabled and LONELINESS_SYSTEM_AVAILABLE:
            try:
                from loneliness_system import LonelinessSystemComponent
                loneliness_system = LonelinessSystemComponent(self.arch_manager)
                self.arch_manager.register_component('loneliness_system', loneliness_system)
                self.logger.log_event('component_registered', 
                                    {'component': 'loneliness_system', 'status': 'success'})
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка регистрации системы одиночества: {e}")
        
        # Интернет-система
        if self.config.internet_enabled:
            try:
                from internet_system import InternetAccessSystem
                internet_system = InternetAccessSystem(self.arch_manager)
                self.arch_manager.register_component('internet_system', internet_system)
                
                if INTERNET_AGENT_AVAILABLE:
                    from simple_internet_agent import SimpleInternetAgent
                    self.internet_agent = SimpleInternetAgent(internet_system)
                    self.arch_manager.register_component('internet_agent', self.internet_agent)
                    self.logger.logger.info("✅ Простой интернет-агент создан")
                    
            except ImportError:
                self.logger.logger.warning("⚠️ Интернет-система недоступна")
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка регистрации интернет-системы: {e}")
                
        # Автономный читатель текстов
        try:
            from text_reader_component import TextReaderComponent
            
            # ✅ Читает всю папку readings (включая parsed_articles)
            readings_dir = self.config.readings_dir
            
            text_reader = TextReaderComponent(
                self.arch_manager, 
                text_dir=readings_dir  # 👈 Будет искать .txt во всех подпапках
            )
            
            self.arch_manager.register_component('text_reader', text_reader)
            self.logger.logger.info(f"📚 Автономный текстовый читатель зарегистрирован. Читает из: {readings_dir}")
            self.logger.logger.info(f"   📁 Включая подпапку: {os.path.join(readings_dir, 'parsed_articles')}")
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка регистрации текстового читателя: {e}")
    
        # 📰 Парсер статей
        try:
            from article_parser import ArticleParser
            article_parser = ArticleParser(self.arch_manager, self.storage)
            self.arch_manager.register_component('article_parser', article_parser)
            self.logger.logger.info("📰 Парсер статей активирован")
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка регистрации парсера статей: {e}")

        # Виртуальный бизнес
        try:
            from virtual_business import VirtualBusinessComponent
            business = VirtualBusinessComponent(self.arch_manager)
            self.arch_manager.register_component('business', business)
            self.logger.logger.info("💼 Виртуальный бизнес открыт!")
        except ImportError:
            pass

    def _analyze_query_effectiveness(self):
        """
        Анализ эффективности запросов и обратная связь
        """
        internet_agent = self.arch_manager.get_component('internet_agent')
        if not internet_agent or not hasattr(internet_agent, 'search_quality_stats'):
            return
        
        stats = internet_agent.search_quality_stats
        history = stats.get('query_history', [])
        
        if len(history) < 10:
            return
        
        # Анализируем последние 20 запросов
        recent = history[-20:]
        
        avg_query_score = sum(h['query_score'] for h in recent) / len(recent)
        avg_result_score = sum(h['result_score'] for h in recent) / len(recent)
        avg_total = (avg_query_score + avg_result_score) / 2
        
        # Если запросы стабильно плохие - даём совет
        if avg_query_score < 0.1 and len(recent) >= 10:
            self.logger.info("📉 Обнаружена проблема с качеством запросов")
            
            # Генерируем совет через модель
            prompt = f"""Последние запросы были не очень удачными (средняя оценка {avg_query_score:.2f}).
            
            Как улучшить формулировку запросов? Напиши 3 совета для себя.
            
            Советы:"""
            
            advice = self._generate_response(prompt, max_tokens=200, log_dialog=False)
            
            # Отправляем как рефлексию
            if self.inner_mirror:
                self.inner_mirror.save_autonomous_result(
                    activity_type='reflection',
                    topic="Анализ качества поисковых запросов",
                    content=advice,
                    metadata={'avg_query_score': avg_query_score}
                )
        
        # Логируем статистику
        self.logger.info(f"📊 Статистика запросов: {avg_query_score:+.2f} (запрос) / {avg_result_score:+.2f} (результат)")
            
    def _subscribe_to_architecture_events(self):
        """Подписка на события архитектуры"""
        event_bus = self.arch_manager.event_bus
        
        if hasattr(self.model, 'on_event'):
            self.model.on_event('curiosity_state_change', self._on_curiosity_state_change)
            self.model.on_event('sleep_started', self._on_sleep_started)
    
    def _on_curiosity_state_change(self, data: Dict):
        """Обработчик изменения состояния любопытства"""
        self.logger.log_event('architecture_curiosity_change', data)
        self.arch_manager.broadcast_event('curiosity_state_changed', data)
    
    def _on_sleep_started(self, data: Dict):
        """Обработчик начала сна"""
        self.logger.log_event('architecture_sleep_started', data)
    
    def _start_threads(self):
        """Запуск рабочих потоков"""
        self.life_thread = threading.Thread(
            target=self._autonomous_life_cycle,
            name='LifeCycle',
            daemon=True
        )
        self.life_thread.start()
        
        self.input_thread = threading.Thread(
            target=self._user_input_loop,
            name='UserInput',
            daemon=True
        )
        self.input_thread.start()
        
        self.command_thread = threading.Thread(
            target=self._command_processing_loop,
            name='CommandProcessor',
            daemon=True
        )
        self.command_thread.start()
        
        self.logger.logger.info("🧵 Рабочие потоки запущены")
    
    def _autonomous_life_cycle(self):
        """Основной цикл автономной жизни - С ИСПРАВЛЕННЫМИ ТАЙМЕРАМИ"""
        while not self.stop_event.is_set():
            try:
                self.life_cycle += 1
                current_time = time.time()
                
                if self.paused_for_user:
                    if current_time >= self.pause_end_time:
                        self.paused_for_user = False
                        self.pause_end_time = 0
                        self.logger.logger.info("🔄 Возвращаюсь к активной деятельности")
                    else:
                        time.sleep(1)
                        continue
                
                # ТИК АРХИТЕКТУРНЫХ КОМПОНЕНТОВ (каждые tick_interval секунд)
                if current_time - self.timers.get('component_tick', 0) > self.config.tick_interval:
                    self._perform_architecture_tick()
                    self.timers['component_tick'] = current_time
                
                # ПОЛНЫЙ ТИК СИСТЕМЫ (каждые 5 минут)
                if current_time - self.timers.get('full_tick', 0) > 300:
                    self._perform_full_system_tick()
                    self.timers['full_tick'] = current_time
                
                # АВТОСОХРАНЕНИЕ
                self.auto_save()
                
                # ========== ИСПРАВЛЕННЫЕ АВТОНОМНЫЕ ДЕЙСТВИЯ ==========
                
                # 1. ШАГ ПОЗНАНИЯ - раз в 2-5 минут
                cognition_interval = getattr(self, 'min_cognition_interval', 120)
                if current_time - self.timers.get('cognition_step', 0) > cognition_interval:
                    # Добавляем случайную вариацию, чтобы не было строгой периодичности
                    if random.random() < 0.7:  # 70% шанс выполнить
                        self.autonomous_cognition_step()
                        self.timers['cognition_step'] = current_time
                
                # 2. АВТОНОМНАЯ ЖИЗНЬ - раз в 45-90 секунд
                life_interval = getattr(self, 'min_autonomous_life_interval', 45)
                life_interval_varied = life_interval + random.randint(-10, 20)
                if current_time - self.timers.get('autonomous_life', 0) > max(30, life_interval_varied):
                    # Не каждый раз, чтобы добавить случайности
                    if random.random() < 0.4:  # 40% шанс
                        self._perform_autonomous_life()
                        self.timers['autonomous_life'] = current_time
                
                # 3. ВЫПОЛНЕНИЕ ЗАДАЧ - раз в 60-120 секунд
                task_interval = getattr(self, 'min_task_interval', 60)
                if current_time - self.timers.get('task_step', 0) > task_interval:
                    if random.random() < 0.3:  # 30% шанс
                        self._execute_next_task()
                        self.timers['task_step'] = current_time
                
                # 4. ОТОБРАЖЕНИЕ СТАТУСА - раз в 3 минуты
                status_interval = getattr(self, 'min_status_interval', 180)
                if current_time - self.timers.get('status_display', 0) > status_interval:
                    self._display_live_status(self.life_cycle)
                    self.timers['status_display'] = current_time
                
                # 5. ОБРАБОТКА ОДИНОЧЕСТВА (специальный компонент)
                loneliness_system = self.arch_manager.get_component('loneliness_system')
                if loneliness_system and hasattr(loneliness_system, 'tick'):
                    loneliness_system.tick()
                
                # ========== ДИНАМИЧЕСКАЯ ПАУЗА ==========
                if not self.paused_for_user:
                    # Базовая пауза
                    base_sleep = 2.0
                    
                    # Увеличиваем паузу, если недавно было автономное действие
                    if current_time - self.timers.get('autonomous_life', 0) < 10:
                        base_sleep += 3.0  # Даем время "переварить"
                    
                    # Увеличиваем паузу, если был cognition_step
                    if current_time - self.timers.get('cognition_step', 0) < 15:
                        base_sleep += 5.0  # Больше времени на размышления
                    
                    # Учитываем усталость
                    if hasattr(self.model, 'personality_core'):
                        try:
                            drive_values = self.model.personality_core.get_drive_values()
                            if 'fatigue' in self.model.personality_core.drive_name_to_idx:
                                fatigue_idx = self.model.personality_core.drive_name_to_idx['fatigue']
                                fatigue = drive_values[fatigue_idx].item()
                                # Чем выше усталость, тем длиннее пауза
                                base_sleep += fatigue * 3.0
                        except:
                            pass
                    
                    # Ограничиваем паузу разумными пределами
                    sleep_time = min(15.0, max(1.0, base_sleep))
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка в жизненном цикле: {e}")
                time.sleep(5)
    
    def _perform_architecture_tick(self):
        """Тик архитектурных компонентов"""
        try:
            # Отправляем событие тика для всех, кто подписан
            self.arch_manager.broadcast_event('house_tick', {
                'timestamp': time.time(),
                'life_cycle': self.life_cycle
            })
            
            # ✅ ВАЖНО: вызываем tick() всех компонентов
            self.arch_manager.tick()
            
            if hasattr(self.model, 'personality_core'):
                is_learning = (
                    hasattr(self.model, 'adaptive_memory') and 
                    self.model.adaptive_memory is not None and
                    getattr(self.model.adaptive_memory, 'is_sleeping', False)
                )
                self.model.personality_core.tick(training=is_learning)
                
                report = self.model.personality_core.get_detailed_report()
                self.arch_manager.broadcast_event('psyche_tick_completed', report)
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка архитектурного тика: {e}")
    
    def _perform_full_system_tick(self):
        """Полный тик системы"""
        try:
            internet_system = self.arch_manager.get_component('internet_system')
            if internet_system:
                internet_system.check_connection(force=True)
            
            if random.random() < 0.3:
                deep_actions = ['philosophy', 'synthesis', 'reflection']
                action = random.choice(deep_actions)
                self.perform_autonomous_action(action)
                    
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка в полном тике системы: {e}")
    
    def _execute_next_task(self):
        """Выполнение следующей задачи"""
        task_scheduler = self.arch_manager.get_component('task_scheduler')
        if not task_scheduler:
            return
        
        task = task_scheduler.get_next_task()
        if not task:
            return
        
        self.logger.logger.info(f"🎯 Выполняю задачу: {task.name}")
        
        try:
            if task.category == "research":
                self._perform_research_task(task)
            elif task.category == "reflection":
                self._perform_reflection_task(task)
            else:
                time.sleep(1)
                task_scheduler.complete_task(task.id)
                
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка выполнения задачи: {e}")
            task_scheduler.complete_task(task.id, False)
    
    def _perform_research_task(self, task: Task):
        """Выполнение исследовательской задачи"""
        topic = task.name.replace("Исследование: ", "")
        self.logger.logger.info(f"🔍 Исследую: {topic}")
        
        time.sleep(random.uniform(2, 5))
        
        task_scheduler = self.arch_manager.get_component('task_scheduler')
        task_scheduler.complete_task(task.id)
    
    def _perform_reflection_task(self, task: Task):
        """Выполнение задачи рефлексии"""
        self.logger.logger.info("🤔 Провожу глубокую рефлексию...")
        time.sleep(random.uniform(2, 4))
        
        task_scheduler = self.arch_manager.get_component('task_scheduler')
        task_scheduler.complete_task(task.id)
    
    def _user_input_loop(self):
        """Цикл ввода пользователя"""
        while not self.stop_event.is_set():
            try:
                user_input = input().strip()
                if user_input:
                    self.input_queue.put(user_input)
                    self.logger.logger.info(f"👤 Пользователь: {user_input}")
                    # ========== ИСПРАВЛЕННАЯ ПАУЗА ==========
                    self.pause_for_user_interaction()
                    
            except (EOFError, KeyboardInterrupt):
                self.stop_event.set()
                break
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка ввода: {e}")
                time.sleep(1)
    
    def _command_processing_loop(self):
        """Цикл обработки команд"""
        while not self.stop_event.is_set():
            try:
                time.sleep(0.1)
                
                if self.input_queue.empty():
                    continue
                
                user_input = self.input_queue.get()
                response = self._process_command(user_input)
                
                if response:
                    self.logger.logger.debug(f"Thalia: {response}")
                
                self.input_queue.task_done()
                
            except Exception as e:
                self.logger.logger.error(f"❌ Ошибка обработки команды: {e}")
    
    def _process_command(self, user_input: str) -> str:
        """
        Обработка ввода. 
        Больше никаких скриптовых команд (кроме системных /save, /exit).
        Все запросы обрабатываются интеллектом модели.
        """
        raw_input = user_input.strip()
        if not raw_input:
            return ""

        # Системные команды для управления процессом (не диалогом)
        if raw_input in ["/save", "/exit", "/quit", "сохранить", "выход"]:
            self._save_and_exit(None, None)
            return "🛑 Система сохранена и останавливается..."

        # 1. Сначала пытаемся понять, нужно ли действие (Поиск/Бизнес)
        # Это можно реализовать через классификатор намерений, 
        # но пока просто передаем в генерацию ответа.
        
        # 2. Генерация ответа с использованием памяти и интернета
        response = self._generate_response(raw_input)
        return response
    
    def _generate_response(self, message: str, max_tokens: int = 2048, 
                           log_dialog: bool = True, save_dialog: bool = True) -> str:  # <-- добавить параметр
        """Чистая генерация через архитектуру - С ИСПРАВЛЕННЫМИ ПРОБЕЛАМИ"""
        try:
            # ========== ПРАВИЛЬНЫЙ ФОРМАТ ДЛЯ ДИАЛОГА ==========
            formatted_prompt = f"### Максим:\n{message}\n\n### Талия:\n"
            
            if log_dialog:
                print(f"\n\033[96m┌─ 👤 Максим")
                print(f"│ {message}")
                print(f"└─────────────────────\033[0m")
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            if inputs.input_ids.shape[-1] == 0:
                return "Я не расслышала. Повторите?"
            
            generated_ids, metadata = self.model.generate_with_psyche(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                use_chain_of_thought=True,
                record_experience=True,
                do_sample=True,
            )
            
            input_len = inputs.input_ids.shape[-1]
            
            if generated_ids.shape[-1] <= input_len:
                return "Я задумалась."
            
            response_tokens = generated_ids[0][input_len:]
            
            if len(response_tokens) == 0:
                return "Я задумалась."
            
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # 1. Обрезаем возможное повторение имени
            if response.startswith("Талия:"):
                response = response[6:].strip()
            if response.startswith("### Талия:"):
                response = response[10:].strip()
            
            # 2. ✅ ИСПРАВЛЯЕМ ПРОБЕЛЫ ПОСЛЕ ЗНАКОВ ПРЕПИНАНИЯ
            response = re.sub(r'([.!?;:,])([^\s])', r'\1 \2', response)
            
            # 3. ✅ ИСПРАВЛЯЕМ ПРОБЕЛЫ МЕЖДУ СЛОВАМИ (слипшиеся слова)
            response = re.sub(r'([а-яА-ЯёЁa-zA-Z])([А-ЯA-Z])', r'\1 \2', response)
            
            # 4. ✅ УБИРАЕМ ЛИШНИЕ ПРОБЕЛЫ
            response = re.sub(r'\s+', ' ', response).strip()
            
            # 5. ✅ ИСПРАВЛЯЕМ ПРОБЕЛЫ ПЕРЕД ЗНАКАМИ ПРЕПИНАНИЯ
            response = re.sub(r'\s+([.!?;:,])', r'\1', response)
            
            # 6. ✅ КАПИТАЛИЗАЦИЯ ПЕРВОЙ БУКВЫ ПРЕДЛОЖЕНИЙ
            sentences = re.split(r'([.!?]+(?:\s|$))', response)
            for i in range(0, len(sentences), 2):
                if sentences[i]:
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:] if sentences[i] else ''
            response = ''.join(sentences)
            
            # 7. ✅ ФИНАЛЬНАЯ ОЧИСТКА
            response = re.sub(r'\s+', ' ', response).strip()
            
            if not response:
                return "Я задумалась."
            
            # ========== ТАЛИЯ ==========
            if log_dialog:
                print(f"\033[92m┌─ 🐲 ТАЛИЯ")
                print(f"│ {response}")
                print(f"└─────────────────────\033[0m\n")
            
            # Сохраняем диалог ТОЛЬКО если нужно
            if save_dialog and message and response:  # <-- изменить условие
                dialog_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'user': message[:1000],
                    'thalia': response[:2000],
                    'tokens': len(response_tokens)
                }
                self.dialog_history.append(dialog_entry)
                
                if len(self.dialog_history) % 5 == 0:
                    self._save_dialog_history()
            
            return response
            
        except Exception as e:
            self.logger.logger.error(f"❌ Ошибка генерации: {e}")
            if log_dialog:
                print(f"\033[92m┌─ 🐲 ТАЛИЯ")
                print(f"│ Я задумалась.")
                print(f"└─────────────────────\033[0m\n")
            return "Я задумалась."
        
    def _build_dialog_context(self) -> str:
        """Сбор контекста из последних диалогов"""
        if len(self.dialog_history) < 2:
            return ""
        
        # Берем последние 3 обмена
        recent = self.dialog_history[-3:]
        context_parts = []
        
        for entry in recent:
            context_parts.append(f"Максим: {entry['user'][:200]}")
            context_parts.append(f"Талия: {entry['thalia'][:200]}")
        
        return "\n".join(context_parts)
        
    def _save_and_exit(self, signum, frame):
        """Сохранение при Ctrl+C - ПОЛНОЕ"""
        print("\n🚨 Ctrl+C - сохраняю модель и диалоги...")
        self.save_all_state() 
        print(f"✅ Все данные сохранены в {self.config.base_dir}. Выход.")
        sys.exit(0)

    def pause_for_user_interaction(self, duration: int = None):
        """Умная пауза для пользовательского взаимодействия"""
        self.paused_for_user = True
        
        # Динамическая длительность паузы
        if duration is None:
            # Базовая длительность из конфига
            base_duration = self.config.pause_default_duration
            
            # Увеличиваем паузу, если пользователь много пишет
            if hasattr(self, 'dialog_history') and len(self.dialog_history) > 5:
                recent = self.dialog_history[-5:]
                avg_length = sum(len(d.get('user', '')) for d in recent) / len(recent)
                if avg_length > 200:  # Длинные сообщения
                    base_duration += 10
                elif avg_length > 100:  # Средние
                    base_duration += 5
            
            # Учитываем усталость
            if hasattr(self.model, 'personality_core'):
                try:
                    drive_values = self.model.personality_core.get_drive_values()
                    if 'fatigue' in self.model.personality_core.drive_name_to_idx:
                        fatigue_idx = self.model.personality_core.drive_name_to_idx['fatigue']
                        fatigue = drive_values[fatigue_idx].item()
                        if fatigue > 0.7:  # Сильно устала
                            base_duration += 15
                        elif fatigue > 0.4:  # Умеренно устала
                            base_duration += 8
                except:
                    pass
            
            pause_duration = base_duration
        else:
            pause_duration = duration
        
        self.pause_end_time = time.time() + pause_duration
        self.logger.logger.info(f"⏸️  Пауза для пользователя на {pause_duration} секунд")
        
        # Отмечаем в системе одиночества, что было взаимодействие
        loneliness_system = self.arch_manager.get_component('loneliness_system')
        if loneliness_system:
            loneliness_system.register_interaction()

    def _check_pause(self) -> bool:
        """УПРОЩЕННАЯ проверка паузы - больше не нужна, логика в основном цикле"""
        return False  # Вся логика теперь в _autonomous_life_cycle

    def _display_live_status(self, cycle_count: int):
        """Отображение живого статуса системы """
        try:
            status_lines = []
            status_lines.append(f"🌀 Цикл: {cycle_count}")
            
            # Информация о хранилище
            status_lines.append(f"📁 {os.path.basename(self.config.base_dir)}")
            
            # Бизнес-статус
            business = self.arch_manager.get_component('virtual_business')
            if business:
                stats = business.get_stats()
                if stats.get('active_order'):
                    order = stats['active_order']
                    status_lines.append(f"📦 {order['product'][:15]}...")
                status_lines.append(f"🏢 Ур.{stats.get('level', 0)}")
            
            if hasattr(self.model, 'personality_core'):
                try:
                    report = self.model.personality_core.get_detailed_report()
                    mood = report.get('mood', {}).get('state', 0)
                    mood_str = f"{'😊' if mood > 0.3 else '😐' if mood > -0.3 else '😔'}"
                    status_lines.append(f"{mood_str}")
                    
                    drives = report.get('drives', {})
                    if drives.get('novelty', 0) > 0.6:
                        status_lines.append(f"🔍 {drives['novelty']:.1f}")
                        
                except:
                    pass
            
            loneliness_system = self.arch_manager.get_component('loneliness_system')
            if loneliness_system:
                level = loneliness_system.loneliness_level
                if level > 3:
                    level_str = f"{'😐' if level < 6 else '😔'} {level:.0f}"
                    status_lines.append(f"{level_str}")
            
            if self.autonomous_knowledge:
                status_lines.append(f"🧠 {len(self.autonomous_knowledge)}")
            
            task_scheduler = self.arch_manager.get_component('task_scheduler')
            if task_scheduler and task_scheduler.tasks:
                status_lines.append(f"📝 {len(task_scheduler.tasks)}")
            
            # ИСПРАВЛЕННАЯ информация о паузе
            if self.paused_for_user:
                remaining = max(0, int(self.pause_end_time - time.time()))
                status_lines.append(f"⏸️ {remaining}с")
            
            if status_lines:
                status_text = " | ".join(status_lines)
                self.logger.logger.info(f"📊 {status_text}")
                
        except Exception as e:
            self.logger.logger.debug(f"⚠️ Ошибка отображения статуса: {e}")

    def run(self):
        """Основной цикл работы - живая активность Талии"""
        self.logger.logger.info("\n" + "=" * 60)
        self.logger.logger.info("🏠 THALIA HOUSE - ЕДИНОЕ ХРАНИЛИЩЕ")
        self.logger.logger.info("=" * 60)
        self.logger.logger.info(f"📁 Все данные в: {self.config.base_dir}")
        self.logger.logger.info("🤖 Автономное познание: ВКЛ")
        self.logger.logger.info("💭 Система одиночества: ВКЛ")
        self.logger.logger.info("🌐 Интернет-агент: ВКЛ")
        self.logger.logger.info(f"⏱️  Тик-интервал: {self.config.tick_interval} секунд")
        self.logger.logger.info(f"⏸️  Пауза после ответа: {self.config.pause_default_duration} секунд")
        self.logger.logger.info(f"💾 Автосохранение каждые {self.config.auto_save_interval//60} минут")
        self.logger.logger.info("💬 Любой ввод передаётся модели, системные команды: /save, /exit, сохранить, выход")
        
        cycle_count = 0
        
        try:
            while not self.stop_event.is_set():
                cycle_count += 1
                
                # Основной цикл - вся логика в _autonomous_life_cycle
                time.sleep(0.5)  # Небольшая пауза для снижения нагрузки
                
        except KeyboardInterrupt:
            self.logger.logger.info("\n👋 Завершение работы...")
            self.stop_event.set()
        
        except Exception as e:
            self.logger.logger.error(f"❌ Критическая ошибка в основном цикле: {e}")
            self.stop_event.set()
        
        finally:
            self.logger.logger.info("✨ Работа завершена")
            self._save_autonomous_knowledge()
            self._save_system_state()
            self.save_model()

# ===================================================================
# ТОЧКА ВХОДА
# ===================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🏠 Thalia House - Единое хранилище"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Путь к модели Thalia"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="thalia_memory",
        help="Базовая папка для всех данных (по умолчанию: thalia_memory)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Путь к конфигурационному файлу (JSON)"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=20,
        help="Длительность паузы после ответа в секундах (по умолчанию: 20)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Модель не найдена: {args.model_path}")
        sys.exit(1)
    
    # Загружаем конфигурацию
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config = HouseConfig.from_dict(config_data)
    else:
        config = HouseConfig()
    
    # Устанавливаем базовую папку
    config.base_dir = args.base_dir
    config.pause_default_duration = args.pause
    
    # Обновляем все пути
    config.logs_dir = os.path.join(config.base_dir, 'logs')
    config.knowledge_dir = os.path.join(config.base_dir, 'knowledge')
    config.system_state_dir = os.path.join(config.base_dir, 'system_state')
    config.business_dir = os.path.join(config.base_dir, 'business')
    config.readings_dir = os.path.join(config.base_dir, 'readings')
    
    # Создаем структуру папок
    config.ensure_directories()
    
    print(f"📁 Единое хранилище: {config.base_dir}")
    print(f"💾 Модель будет сохранена в: {os.path.join(config.base_dir, 'saved_model')}")
    
    house = ThaliaHouseArchitectural(args.model_path, config)
    
    
    try:
        house.run()
    except Exception as e:
        house.logger.logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()