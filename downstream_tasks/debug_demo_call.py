#!/usr/bin/env python3
"""
Debug script to understand the demo generation calling pattern
"""

import json
import torch
from stable_audio_tools.models.factory import create_model_from_config
from downstream_tasks.eff_models.lora_dit_wrapper import LoRADiTWrapper
from downstream_tasks.configs.lora_config import get_lora_config

def debug_demo_call():
    print("🔍 Debugging Demo Generation Call Pattern...")
    
    # Load model config
    with open('configs/model_config.json', 'r') as f:
        model_config = json.load(f)
    
    # Create base model
    print(f"\n🏗️  Creating base model...")
    base_model = create_model_from_config(model_config)
    
    # Apply LoRA
    print(f"\n🔧 Applying LoRA...")
    lora_config = get_lora_config()
    lora_model = LoRADiTWrapper(base_model, lora_config)
    
    # Create dummy inputs similar to demo generation
    print(f"\n🧪 Creating dummy inputs...")
    batch_size = 4
    io_channels = 64
    seq_len = 108  # 5 seconds at 44.1kHz with 2048 downsampling
    
    x = torch.randn(batch_size, io_channels, seq_len)
    t = torch.rand(batch_size)
    
    # Create dummy conditioning (similar to what cond_inputs would contain)
    cond_inputs = {
        'cross_attn_cond': torch.randn(batch_size, 128, 768),  # T5 embeddings
        'cross_attn_mask': torch.ones(batch_size, 128).bool(),
        'global_cond': torch.randn(batch_size, 2304),  # Global conditioning
        'input_concat_cond': None,
        'prepend_cond': None,
        'prepend_cond_mask': None,
        'cfg_scale': 3.0,
        'dist_shift': None,
        'batch_cfg': True
    }
    
    print(f"\n📊 Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  cond_inputs keys: {list(cond_inputs.keys())}")
    
    # Test 1: Direct call to lora_model (this is what demo generation does)
    print(f"\n🧪 Test 1: Direct call to lora_model...")
    try:
        result = lora_model(x, t, **cond_inputs)
        print(f"✅ Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print(f"Error type: {type(e)}")
    
    # Test 2: Call with cond as positional argument (training pattern)
    print(f"\n🧪 Test 2: Training pattern call...")
    try:
        cond = {k: v for k, v in cond_inputs.items() if v is not None}
        result = lora_model(x, t, cond)
        print(f"✅ Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print(f"Error type: {type(e)}")
    
    # Test 3: Call underlying model directly
    print(f"\n🧪 Test 3: Call underlying model directly...")
    try:
        if hasattr(lora_model.peft_model, 'model'):
            result = lora_model.peft_model.model(x, t, **cond_inputs)
            print(f"✅ Success! Result shape: {result.shape}")
        else:
            print(f"❌ No 'model' attribute found")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print(f"Error type: {type(e)}")
    
    # Test 4: Check what the base model expects
    print(f"\n🧪 Test 4: Check base model signature...")
    import inspect
    if hasattr(base_model, 'forward'):
        sig = inspect.signature(base_model.forward)
        print(f"Base model forward signature: {sig}")
    
    if hasattr(lora_model.peft_model, 'forward'):
        sig = inspect.signature(lora_model.peft_model.forward)
        print(f"PEFT model forward signature: {sig}")

if __name__ == "__main__":
    debug_demo_call()
