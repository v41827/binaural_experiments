# custom_metadata.py
import json
import os

def get_custom_metadata(info, audio):
    """
    Returns consistent prompt + azimuth for azimuth-focused training.
    """
    filename = os.path.basename(info["relpath"])
    
    # Load your metadata.json
    metadata_path = "/parallel_scratch/yc01847/binaural_experiments/downstream_tasks/data/all_binaural/metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        if filename in all_metadata:
            azimuth = all_metadata[filename]["azimuth"]
        else:
            azimuth = 0  # fallback
            
        return {
            "prompt": "environment sound",  # Consistent prompt
            "azimuth": azimuth              # Raw azimuth value
        }
        
    except Exception as e:
        print(f"Error loading metadata for {filename}: {e}")
        return {
            "prompt": "environment sound",
            "azimuth": 0
        }