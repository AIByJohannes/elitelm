#!/usr/bin/env python3
"""
Compare CPU vs NPU inference performance for Llama models using EliteLM ChatSession.
"""

import argparse
import sys
import time
from pathlib import Path

from elitelm import ChatSession, load_config, AppConfig

def run_session(config: AppConfig, prompt: str, label: str) -> tuple[str, float, float]:
    print(f"\n" + "-"*70)
    print(f"Running on {label}...")
    print("-" * 70)
    
    try:
        session = ChatSession(config)
        print(f"Model loaded on {session.device_label}")
        
        # Validate device matches expectation
        expected_device = "QNN" if config.device == "qnn" else "CPU"
        if session.device_label != expected_device:
            raise RuntimeError(
                f"Device mismatch! Expected {expected_device} but got {session.device_label}. "
                f"Model may not have loaded with device={config.device}"
            )
        
        print("Generating...")
        result = session.generate(prompt, timings=True)
        
        text = result.text
        print(f"Output: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        if result.stats:
            print(f"Time to first token: {result.stats.time_to_first_token:.4f}s")
            print(f"Gen TPS: {result.stats.generated_tokens_per_second:.2f}")
            tps = result.stats.generated_tokens_per_second
            ttf = result.stats.time_to_first_token
        else:
            tps = 0.0
            ttf = 0.0
        
        # Explicit cleanup
        del session
        return text, tps, ttf

    except FileNotFoundError as e:
        print(f"Error: Model or config file not found for {label}: {e}")
        return "", 0.0, 0.0
    except RuntimeError as e:
        if "QNN" in str(e) or "device mismatch" in str(e).lower():
            print(f"Error: QNN/NPU not available for {label}: {e}")
            print("Tip: Check QNN SDK installation and NPU hardware availability")
        else:
            print(f"Runtime error on {label}: {e}")
        return "", 0.0, 0.0
    except Exception as e:
        print(f"Unexpected error running on {label}: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, 0.0

def check_prerequisites(config: AppConfig) -> bool:
    """Validate that prerequisites for NPU testing are met."""
    try:
        import onnxruntime_genai as og
    except ImportError:
        print("⚠️  Warning: onnxruntime_genai not installed")
        return False
    
    # Check if QNN is compiled in
    try:
        if not og.is_qnn_available():
            print("⚠️  Warning: QNN is not available in onnxruntime-genai")
            print("NPU test will likely fail. Continuing anyway...")
            return False
    except AttributeError:
        print("⚠️  Warning: is_qnn_available() not found in onnxruntime-genai")
        return False
    
    # Check if model directory exists
    model_dir = Path(config.model)
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        return False
    
    # Check if model has genai_config.json for QNN
    genai_config = model_dir / "genai_config.json"
    if not genai_config.exists():
        print(f"⚠️  Warning: {genai_config} not found")
        print("Model may not be configured for QNN/NPU execution")
        return False
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU inference performance")
    parser.add_argument("-c", "--cpu-config", default="llama3-qa.yaml", help="Path to CPU config file")
    parser.add_argument("-n", "--npu-config", default="llama3-npu.yaml", help="Path to NPU config file")
    parser.add_argument("prompt", nargs="?", default="What is the capital of France?", help="Input prompt")
    args = parser.parse_args()

    print("="*70)
    print("CPU vs NPU Performance Comparison")
    print("="*70)
    print(f"Prompt: {args.prompt}")
    print()

    # CPU Run
    try:
        cpu_config = load_config(args.cpu_config)
        print(f"CPU Config: {args.cpu_config} (model: {cpu_config.model})")
    except Exception as e:
        print(f"Failed to load CPU config: {e}")
        sys.exit(1)
    
    _, cpu_tps, cpu_ttf = run_session(cpu_config, args.prompt, "CPU")

    # NPU Run
    try:
        npu_config = load_config(args.npu_config)
        print(f"\nNPU Config: {args.npu_config} (model: {npu_config.model})")
        
        # Pre-flight checks for NPU
        print("Running NPU pre-flight checks...")
        check_prerequisites(npu_config)
        print()
    except Exception as e:
        print(f"Failed to load NPU config: {e}")
        print("Skipping NPU test.")
        npu_tps, npu_ttf = 0.0, 0.0
    else:
        _, npu_tps, npu_ttf = run_session(npu_config, args.prompt, "NPU")


    # Compare
    print("\n" + "="*70)
    print("Performance Summary")
    print("="*70)
    
    print(f"{'Metric':<20} | {'CPU':<10} | {'NPU':<10} | {'Speedup':<10}")
    print("-" * 60)
    
    tps_speedup = npu_tps / cpu_tps if cpu_tps > 0 else 0
    print(f"{'Tokens/Sec':<20} | {cpu_tps:<10.2f} | {npu_tps:<10.2f} | {tps_speedup:<10.2f}x")
    
    ttf_speedup = cpu_ttf / npu_ttf if npu_ttf > 0 else 0
    print(f"{'Time to First (s)':<20} | {cpu_ttf:<10.4f} | {npu_ttf:<10.4f} | {ttf_speedup:<10.2f}x")

    if tps_speedup > 1:
        print(f"\n✅ NPU is {tps_speedup:.2f}x faster than CPU (throughput)!")
    elif cpu_tps > 0 and npu_tps > 0 and tps_speedup > 0:
        print(f"\n⚠️  NPU is slower than CPU ({1/tps_speedup:.2f}x)")

if __name__ == "__main__":
    main()
