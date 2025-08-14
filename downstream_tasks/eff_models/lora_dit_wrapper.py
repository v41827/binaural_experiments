# downstream_tasks/eff_models/lora_dit_wrapper.py (enhance your existing file)
import torch
import torch.nn as nn
from peft import get_peft_model
from stable_audio_tools.models.dit import DiffusionTransformer

class LoRADiTWrapper(nn.Module):
    def __init__(self, original_dit_model, lora_config):
        super().__init__()
        self.original_model = original_dit_model
        self.peft_model = get_peft_model(original_dit_model, lora_config)
        
    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)
    
    def save_pretrained(self, path):
        self.peft_model.save_pretrained(path)
    
    def load_pretrained(self, path):
        self.peft_model.load_pretrained(path)