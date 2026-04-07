# thalia/__init__.py
import sys
import os

# Добавляем текущую директорию в путь, чтобы компоненты видели друг друга
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .config import ThaliaConfig
from .modeling_thalia import Thalia
from transformers import AutoConfig, AutoModelForCausalLM

def _register_with_transformers():
    try:
        AutoConfig.register("thalia", ThaliaConfig)
        # После того как мы добавили config_class в Thalia, эта строка сработает:
        AutoModelForCausalLM.register(ThaliaConfig, Thalia)
    except Exception as e:
        # Если уже зарегистрировано, просто пропускаем
        pass

_register_with_transformers()

__all__ = ["Thalia", "ThaliaConfig"]