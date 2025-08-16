# downstream_tasks/eff_models/lora_dit_wrapper.py
import torch
import torch.nn as nn
from peft import get_peft_model

class LoRADiTWrapper(nn.Module):
    def __init__(self, original_dit_model, lora_config):
        super().__init__()
        self.original_model = original_dit_model
        
        # Apply LoRA to the original model
        try:
            self.peft_model = get_peft_model(original_dit_model, lora_config)
            print("✅ LoRA successfully applied!")
        except Exception as e:
            print(f"❌ LoRA initialization failed: {e}")
            print("⚠️  Falling back to original model without LoRA")
            self.peft_model = original_dit_model
        
        # CRITICAL: Maintain the expected structure for training wrapper
        self.model = self.peft_model
        
        # ✅ ADD: Copy ALL attributes from original model
        for attr_name in dir(original_dit_model):
            if not attr_name.startswith('_') and not callable(getattr(original_dit_model, attr_name)):
                try:
                    attr_value = getattr(original_dit_model, attr_name)
                    setattr(self, attr_name, attr_value)
                except Exception as e:
                    print(f"⚠️  Could not copy attribute {attr_name}: {e}")
        
        # ✅ ADD: Ensure critical attributes exist with defaults
        if not hasattr(self, 'dist_shift'):
            self.dist_shift = None
        if not hasattr(self, 'timestep_sampler'):
            self.timestep_sampler = None
        if not hasattr(self, 'validation_timesteps'):
            self.validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
        if not hasattr(self, 'cfg_dropout_prob'):
            self.cfg_dropout_prob = 0.1
        if not hasattr(self, 'use_ema'):
            self.use_ema = False
        if not hasattr(self, 'ema_copy'):
            self.ema_copy = None
        if not hasattr(self, 'log_loss_info'):
            self.log_loss_info = False
        if not hasattr(self, 'clip_grad_norm'):
            self.clip_grad_norm = 0.0
        if not hasattr(self, 'trim_config'):
            self.trim_config = None
        if not hasattr(self, 'inpainting_config'):
            self.inpainting_config = None
        
    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)
    
    def save_pretrained(self, path):
        if hasattr(self.peft_model, 'save_pretrained'):
            self.peft_model.save_pretrained(path)
            print(f"✅ LoRA weights saved to {path}")
        else:
            print("⚠️  save_pretrained not available")
    
    def load_pretrained(self, path):
        if hasattr(self.peft_model, 'load_pretrained'):
            self.peft_model.load_pretrained(path)
            print(f"✅ LoRA weights loaded from {path}")
        else:
            print("⚠️  load_pretrained not available")