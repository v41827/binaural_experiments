# custom_metadata.py
import json
import os

def get_custom_metadata(info, audio):
    """
    info: dict, the .json metadata for latent data points (for now, 20250817)
    Returns consistent prompt + azimuth for azimuth-focused training.
    Handles both raw audio format and pre-encoded latent format.
    """
    # Check if this is latent data (has latent_filename) or raw data (has relpath)
    if "relpath" in info:
        # LATENT FORMAT: Use the existing azimuth and prompt from the JSON
        # The latent JSON already contains the correct azimuth and prompt
        #usage: info.get("key", default_value)
        return {
            "prompt": info.get("prompt", "environment sound"),
            "azimuth": info.get("azimuth", 0)
        }
    else:
        # Fallback for unknown format
        print(f"Warning: Unknown data format in info: {list(info.keys())}")
        return {
            "prompt": "environment sound",
            "azimuth": 0
        }
"""
    elif "relpath" in info:
        # RAW FORMAT: Extract filename and look up in metadata.json
        filename = os.path.basename(info["relpath"])
        
        # Load your metadata.json for raw format
        metadata_path = "/nobackup/babbage/users/yc01847/all_binaural/metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
            
            if filename in all_metadata:
                azimuth = all_metadata[filename]["azimuth"]
            else:
                # Fallback: try to extract azimuth from filename (e.g., "file_az35.wav" -> 35)
                try:
                    if "_az" in filename:
                        # More robust extraction: find the last occurrence of "_az" and extract number
                        parts = filename.split("_az")
                        if len(parts) > 1:
                            # Get the last part and extract the number before the extension
                            last_part = parts[-1]
                            azimuth_str = last_part.split(".")[0]  # Remove file extension
                            azimuth = int(azimuth_str)
                        else:
                            azimuth = 0
                    else:
                        azimuth = 0
                except (ValueError, IndexError) as e:
                    print(f"Error extracting azimuth from filename '{filename}': {e}")
                    azimuth = 0
                    
            return {
                "prompt": "environment sound",  # Consistent prompt
                "azimuth": azimuth              # Raw azimuth value
            }
            
        except Exception as e:
            print(f"Error loading metadata for {filename}: {e}")
            return {
                "prompt": "environment sound",
                "azimuth": azimuth
            }
    
    else:
        # Fallback for unknown format
        print(f"Warning: Unknown data format in info: {list(info.keys())}")
        return {
            "prompt": "environment sound",
            "azimuth": 0
        }

if __name__ == "__main__":
    print("🧪 Testing custom_metadata_1.py")
    print("=" * 50)
    
    # Test 1: Latent format (like 0000002240003.json)
    print("\n📁 Test 1: Latent Format")
    latent_info = {
        "latent_filename": "0000002240003.npy",
        "path": "/parallel_scratch/yc01847/binaural_experiments/downstream_tasks/data/all_binaural/wavs/4-172143-A-13_az55.wav",
        "relpath": "4-172143-A-13_az55.wav",
        "prompt": "environment sound",
        "azimuth": 55
    }
    dummy_audio = None
    result1 = get_custom_metadata(latent_info, dummy_audio)
    print(f"Input: {latent_info}")
    print(f"Output: {result1}")
    print(f"✅ Expected: {{'prompt': 'environment sound', 'azimuth': 55}}")
    print(f"✅ Actual: {result1}")
    
    # Test 2: Raw format (like your metadata.json)
    print("\n Test 2: Raw Format")
    raw_info = {
        "relpath": "1-100032-A-0_az35.wav"
    }
    result2 = get_custom_metadata(raw_info, dummy_audio)
    print(f"Input: {raw_info}")
    print(f"Output: {result2}")
    print(f"✅ Expected: {{'prompt': 'environment sound', 'azimuth': 35}} (if file exists in metadata.json)")
    print(f"✅ Actual: {result2}")
    
    # Test 3: Raw format with filename fallback
    print("\n Test 3: Raw Format with Filename Fallback")
    raw_info_fallback = {
        "relpath": "test_file_az42.wav"
    }
    result3 = get_custom_metadata(raw_info_fallback, dummy_audio)
    print(f"Input: {raw_info_fallback}")
    print(f"Output: {result3}")
    print(f"✅ Expected: {{'prompt': 'environment sound', 'azimuth': 42}} (extracted from filename)")
    print(f"✅ Actual: {result3}")
    
    # Test 4: Unknown format
    print("\n📁 Test 4: Unknown Format")
    unknown_info = {
        "unknown_key": "unknown_value"
    }
    result4 = get_custom_metadata(unknown_info, dummy_audio)
    print(f"Input: {unknown_info}")
    print(f"Output: {result4}")
    print(f"✅ Expected: {{'prompt': 'environment sound', 'azimuth': 0}} (fallback)")
    print(f"✅ Actual: {result4}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    
    # Summary
    print("\n📊 Test Summary:")
    tests = [
        ("Latent Format", result1 == {"prompt": "environment sound", "azimuth": 55}),
        ("Raw Format", result2.get("azimuth") is not None),
        ("Filename Fallback", result3 == {"prompt": "environment sound", "azimuth": 42}),
        ("Unknown Format", result4 == {"prompt": "environment sound", "azimuth": 0})
    ]
    
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
"""