# downstream_tasks/train_lora_dit.py
import torch
from stable_audio_tools.models.factory import create_model_from_config
from downstream_tasks.eff_models.lora_dit_wrapper import LoRADiTWrapper
from downstream_tasks.configs.lora_config import get_lora_config

def setup_lora_training():
    # Load original DiT model
    original_model = create_model_from_config("path/to/dit_config.json")
    
    # Create LoRA config
    lora_config = get_lora_config()
    
    # Wrap with LoRA
    lora_model = LoRADiTWrapper(original_model, lora_config)
    
    return lora_model