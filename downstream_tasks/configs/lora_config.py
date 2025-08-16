# downstream_tasks/configs/lora_config.py
from peft import LoraConfig

def get_lora_config():
    return LoraConfig(
        r=12,  # Increased from 8 for better azimuth precision
        lora_alpha=24,  # Keep 2:1 ratio
        target_modules=[
            # CRITICAL: Azimuth conditioning layers (highest priority)
            "to_cond_embed.0", "to_cond_embed.2",          # Text conditioning
            "to_global_embed.0", "to_global_embed.2",      # Global conditioning  
            "to_prepend_embed.0", "to_prepend_embed.2",    # Prepend conditioning
            
            # IMPORTANT: Cross-attention for azimuth-text interaction
            "transformer.layers.*.cross_attn.to_q",         # Cross-attention Q
            "transformer.layers.*.cross_attn.to_kv",        # Cross-attention KV
            "transformer.layers.*.cross_attn.to_out",       # Cross-attention output
            
            # MODERATE: Core attention for azimuth influence
            "to_q", "to_kv", "to_out",                      # Main attention
            
            # LOWER PRIORITY: Keep minimal for efficiency
            "linear_in", "linear_out",                      # Basic FFN
            "project_in", "project_out"                     # I/O projections
        ],
        lora_dropout=0.05,  # Reduced dropout for better azimuth learning
        bias="none",
        task_type="CAUSAL_LM"
    )