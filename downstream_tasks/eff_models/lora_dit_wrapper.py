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
        
        # ✅ ROBUST: Copy ALL attributes including methods
        self._copy_all_attributes(original_dit_model)
        
        # ✅ ADD: Ensure critical attributes exist with defaults
        self._ensure_critical_attributes()
        
        # ✅ ADD: Debug information
        self._debug_attributes()
        
    def _copy_all_attributes(self, original_model):
        """Copy all attributes from original model"""
        # Get all attributes from original model
        original_attrs = set(dir(original_model))
        # Get current attributes (excluding built-ins)
        current_attrs = set(dir(self))
        
        # Find attributes to copy
        attrs_to_copy = original_attrs - current_attrs
        
        for attr_name in attrs_to_copy:
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(original_model, attr_name)
                    setattr(self, attr_name, attr_value)
                    print(f"✅ Copied attribute: {attr_name}")
                except Exception as e:
                    print(f"⚠️  Could not copy attribute {attr_name}: {e}")
    
    def _ensure_critical_attributes(self):
        """Ensure all critical attributes exist with defaults"""
        critical_attrs = {
            'dist_shift': None,
            'timestep_sampler': None,
            'validation_timesteps': [0.1, 0.3, 0.5, 0.7, 0.9],
            'cfg_dropout_prob': 0.1,
            'use_ema': False,
            'ema_copy': None,
            'log_loss_info': False,
            'clip_grad_norm': 0.0,
            'trim_config': None,
            'inpainting_config': None,
            'conditioner': None,  # This is critical!
        }
        
        for attr_name, default_value in critical_attrs.items():
            if not hasattr(self, attr_name):
                setattr(self, attr_name, default_value)
                print(f"⚠️  Set default for missing attribute: {attr_name}")
    
    def _debug_attributes(self):
        """Debug information about copied attributes"""
        print(f"🔍 Total attributes: {len([attr for attr in dir(self) if not attr.startswith('_')])}")
        print(f"🔍 Has conditioner: {hasattr(self, 'conditioner')}")
        if hasattr(self, 'conditioner'):
            print(f"🔍 Conditioner type: {type(self.conditioner)}")
        print(f"🔍 Has dist_shift: {hasattr(self, 'dist_shift')}")
        print(f"🔍 Has timestep_sampler: {hasattr(self, 'timestep_sampler')}")
        
    def forward(self, *args, **kwargs):
        # ✅ ROBUST: Handle any calling pattern
        try:
            # Try to call the underlying model directly
            return self.peft_model(*args, **kwargs)
        except TypeError as e:
            # If that fails, try to fix the arguments
            if "missing 1 required positional argument: 'cond'" in str(e):
                # The model expects 'cond' as third positional argument
                # But demo generation passes conditioning as kwargs
                if len(args) >= 2:
                    x, t = args[0], args[1]
                    # Extract conditioning from kwargs
                    cond = {}
                    # Look for conditioning keys in kwargs
                    cond_keys = ['cross_attn_cond', 'global_cond', 'input_concat_cond', 'prepend_cond']
                    for key in cond_keys:
                        if key in kwargs:
                            cond[key] = kwargs.pop(key)
                    # Also check for any other conditioning-related keys
                    for key in list(kwargs.keys()):
                        if 'cond' in key or 'mask' in key:
                            cond[key] = kwargs.pop(key)
                    # Pass cond as third positional argument
                    return self.peft_model(x, t, cond, **kwargs)
                else:
                    raise e
            else:
                raise e
    
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