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
        # The training wrapper expects self.model to exist
        self.model = self.peft_model
        
        # Copy other attributes that the training wrapper might need
        if hasattr(original_dit_model, 'diffusion_objective'):
            self.diffusion_objective = original_dit_model.diffusion_objective
        if hasattr(original_dit_model, 'conditioner'):
            self.conditioner = original_dit_model.conditioner
        if hasattr(original_dit_model, 'io_channels'):
            self.io_channels = original_dit_model.io_channels
        if hasattr(original_dit_model, 'sample_rate'):
            self.sample_rate = original_dit_model.sample_rate
        if hasattr(original_dit_model, 'min_input_length'):
            self.min_input_length = original_dit_model.min_input_length
        if hasattr(original_dit_model, 'cross_attn_cond_ids'):
            self.cross_attn_cond_ids = original_dit_model.cross_attn_cond_ids
        if hasattr(original_dit_model, 'global_cond_ids'):
            self.global_cond_ids = original_dit_model.global_cond_ids
        if hasattr(original_dit_model, 'input_concat_ids'):
            self.input_concat_ids = original_dit_model.input_concat_ids
        if hasattr(original_dit_model, 'prepend_cond_ids'):
            self.prepend_cond_ids = original_dit_model.prepend_cond_ids
        if hasattr(original_dit_model, 'pretransform'):
            self.pretransform = original_dit_model.pretransform
        
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