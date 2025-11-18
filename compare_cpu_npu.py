#!/usr/bin/env python3
"""
Compare CPU vs NPU inference performance for Llama models using EliteLM ChatSession.
"""

import argparse
import sys
import time
from copy import deepcopy

from elitelm import ChatSession, load_config, AppConfig

def run_session(config: AppConfig, prompt: str, label: str) -> tuple[str, float, float]:
    print(f"\n" + "-"*70)
    print(f"Running on {label}...")
    print("-" * 70)
    
    try:
        session = ChatSession(config)
        print(f"Model loaded on {session.device_label}")
        
        # Force timings on
        session.args.runtime.timings = True
        # Also update search options if needed, but config should have them.
        
        print("Generating...")
        result = session.generate(prompt, timings=True)
        
        text = result.text
        print(f"Output: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        if result.stats:
            print(f"Time to first token: {result.stats.time_to_first_token:.4f}s")
            print(f"Gen TPS: {result.stats.generated_tokens_per_second:.2f}")
            return text, result.stats.generated_tokens_per_second, result.stats.time_to_first_token
        else:
            return text, 0.0, 0.0

    except Exception as e:
        print(f"Error running on {label}: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU inference performance")
    parser.add_argument("-c", "--config", default="llama3-qa.yaml", help="Path to config file")
    parser.add_argument("prompt", nargs="?", default="What is the capital of France?", help="Input prompt")
    args = parser.parse_args()

    try:
        base_config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    print("="*70)
    print("CPU vs NPU Performance Comparison")
    print("="*70)
    print(f"Prompt: {args.prompt}")

    # CPU Run
    cpu_config = base_config.model_copy(deep=True)
    cpu_config.device = "cpu"
    
    _, cpu_tps, cpu_ttf = run_session(cpu_config, args.prompt, "CPU")

    # NPU Run
    npu_config = base_config.model_copy(deep=True)
    npu_config.device = "qnn"
    
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
    elif cpu_tps > 0 and npu_tps > 0:
        print(f"\n⚠️  NPU is slower than CPU ({1/tps_speedup:.2f}x)")

if __name__ == "__main__":
    main()
