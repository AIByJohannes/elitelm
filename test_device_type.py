
import onnxruntime_genai as og
import argparse
from pathlib import Path

def test_device_type():
    print("Testing og.Model.device_type behavior...")
    
    # 1. CPU Model
    print("\n--- CPU Load ---")
    try:
        model_path = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
        if not Path(model_path).exists():
            print(f"Skipping CPU test: {model_path} not found")
        else:
            model = og.Model(model_path)
            print(f"Model created. device_type: {model.device_type}")
    except Exception as e:
        print(f"CPU Load failed: {e}")

    # 2. QNN Load (Simulated failure if possible, or just check what it says)
    print("\n--- QNN Load ---")
    # We will try to load the NPU model but with a config that might fail or just check normal NPU load
    model_path = "cpu_and_mobile/llama-3.2-3b-npu-complete" 
    # Note: The user's path might be different, I'll use a generic one or try to find one
    
    # Actually, let's just use the same path as CPU but try to configure QNN
    # If QNN fails, we want to see what device_type says
    
    try:
        # Create a dummy config to force QNN provider
        config = og.Config(model_path)
        config.clear_providers()
        config.append_provider("QNNExecutionProvider")
        # We won't set valid paths, so it SHOULD fail to load the backend
        # and hopefully fallback to CPU (if that's what's happening) or throw error
        
        print("Attempting to create model with QNN provider (invalid config)...")
        model = og.Model(config)
        print(f"Model created. device_type: {model.device_type}")
    except Exception as e:
        print(f"QNN Load failed as expected: {e}")

if __name__ == "__main__":
    test_device_type()
