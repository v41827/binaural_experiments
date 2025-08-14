# downstream_tasks/configs/lora_config.py
from peft import LoraConfig

def get_lora_config():
    return LoraConfig(
        r=8,  # Lower rank for efficiency
        lora_alpha=16,
        target_modules=[
            # Attention layers in transformer blocks
            "to_q", "to_kv", "to_out",
            
            # Feed-forward layers in transformer blocks  
            "linear_in", "linear_out",
            
            # Conditioning projection layers
            "to_timestep_embed.0", "to_timestep_embed.2",  # Sequential layers
            "to_cond_embed.0", "to_cond_embed.2",          # Sequential layers
            
            # Global conditioning layers
            "to_global_embed.0", "to_global_embed.2",      # Sequential layers
            
            # Prepend conditioning layers
            "to_prepend_embed.0", "to_prepend_embed.2",    # Sequential layers
            
            # Projection layers
            "project_in", "project_out"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # Appropriate for diffusion models
    )