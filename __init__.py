# Файл: ComfyUI/custom_nodes/wanhumo_custom_nodes/__init__.py

# Імпортуємо класи, які ми створили у наших окремих файлах
from .wanhumo_conditioning import WanHuMo_Conditioning
from .wanhumo_sampler import WanHuMo_Sampler

# --- 1. Словник класів (Обов'язково) ---
# Ключ: Технічна назва класу
# Значення: Сам клас
NODE_CLASS_MAPPINGS = {
    "WanHuMo_Conditioning": WanHuMo_Conditioning,
    "WanHuMo_Sampler": WanHuMo_Sampler,
}

# --- 2. Словник відображуваних назв (Опціонально, але рекомендовано) ---
# Ключ: Технічна назва класу
# Значення: Назва, яка буде відображатися в меню ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanHuMo_Conditioning": "WanHuMo Conditioning (w/ Scales)",
    "WanHuMo_Sampler": "WanHuMo Sampler (Multi-CFG)",
}

# --- 3. (Опціонально) WEB_DIRECTORY ---
# Якщо ви використовуєте власний JavaScript для фронтенду, вкажіть шлях
# import os
# WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

# --- 4. (Опціонально) __all__ ---
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']