# downstream_tasks/configs/lora_config.py
from peft import LoraConfig

def get_lora_config():
    return LoraConfig(
        r=12,  # Increased from 8 for better azimuth precision
        lora_alpha=24,  # Keep 2:1 ratio
        target_modules=[
            # SAFE: Azimuth conditioning layers (highest priority)
            "to_cond_embed.0", "to_cond_embed.2",          # Text conditioning (768 -> 1536)
            "to_global_embed.0", "to_global_embed.2",      # Global conditioning (1536 -> 1536)
            "to_prepend_embed.0", "to_prepend_embed.2",    # Prepend conditioning
            
            # SAFE: Cross-attention for azimuth-text interaction
            "transformer.layers.*.cross_attn.to_q",         # Cross-attention Q (1536 -> 1536)
            "transformer.layers.*.cross_attn.to_kv",        # Cross-attention KV (1536 -> 3072)
            "transformer.layers.*.cross_attn.to_out",       # Cross-attention output (1536 -> 1536)
            
            # SAFE: Core attention for azimuth influence (only in transformer layers)
            "transformer.layers.*.attn.to_q",               # Main attention Q (1536 -> 1536)
            "transformer.layers.*.attn.to_kv",              # Main attention KV (1536 -> 3072)
            "transformer.layers.*.attn.to_out",             # Main attention output (1536 -> 1536)
            
            # SAFE: Feed-forward networks in transformer layers
            "transformer.layers.*.ff.linear_in",            # FFN input (1536 -> 6144)
            "transformer.layers.*.ff.linear_out",           # FFN output (6144 -> 1536)
        ],
        lora_dropout=0.05,  # Reduced dropout for better azimuth learning
        bias="none",
    )