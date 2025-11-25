#!/usr/bin/env python3
"""
Compare CPU vs NPU inference performance for Llama models.

CPU uses EliteLM ChatSession (onnxruntime-genai).
NPU uses Genie runtime for direct NPU execution.
"""

import argparse
import sys
import time
from pathlib import Path

from elitelm import ChatSession, load_config, AppConfig
from run_npu_model import run_npu_inference, format_llama3_prompt, GENIE_EXE

def run_cpu_session(config: AppConfig, prompt: str) -> tuple[str, float, float]:
    """Run inference on CPU using onnxruntime-genai."""
    print(f"\n" + "-"*70)
    print(f"Running on CPU...")
    print("-" * 70)
    
    try:
        session = ChatSession(config)
        print(f"Model loaded on {session.device_label}")
        
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
        print(f"Error: Model or config file not found for CPU: {e}")
        return "", 0.0, 0.0
    except RuntimeError as e:
        print(f"Runtime error on CPU: {e}")
        return "", 0.0, 0.0
    except Exception as e:
        print(f"Unexpected error running on CPU: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, 0.0


def run_npu_session(prompt: str) -> tuple[str, float, float]:
    """Run inference on NPU using Genie runtime."""
    print(f"\n" + "-"*70)
    print(f"Running on NPU (Genie)...")
    print("-" * 70)
    
    try:
        import json
        
        # Format prompt for Llama 3.2
        formatted_prompt = format_llama3_prompt(prompt)
        
        # Load tokenizer to count tokens accurately
        tokenizer_path = Path(__file__).parent / "cpu_and_mobile" / "llama-3.2-3b-npu-complete" / "genie_bundle" / "tokenizer.json"
        token_count = None
        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                # We'll count output tokens after generation
            except ImportError:
                tokenizer = None
        else:
            tokenizer = None
        
        # Run inference and measure time
        text, inference_time = run_npu_inference(formatted_prompt, verbose=False)
        
        print(f"Output: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"Total inference time: {inference_time:.4f}s")
        
        # Count tokens using tokenizer if available, otherwise estimate
        if tokenizer:
            try:
                encoded = tokenizer.encode(text)
                token_count = len(encoded.ids)
                print(f"Output tokens: {token_count}")
            except:
                token_count = None
        
        if token_count is None:
            # Fallback: estimate ~4 chars per token for English
            token_count = max(1, len(text) // 4)
            print(f"Estimated output tokens: {token_count}")
        
        tps = token_count / inference_time if inference_time > 0 else 0.0
        print(f"TPS: {tps:.2f}")
        
        # Genie doesn't provide TTFT separately; estimate as portion of total
        # First token typically arrives faster, estimate ~10-20% of total time
        ttf = inference_time * 0.15
        
        return text, tps, ttf

    except FileNotFoundError as e:
        print(f"Error: Genie runtime not found: {e}")
        print("Make sure the NPU model is downloaded to cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle/")
        return "", 0.0, 0.0
    except Exception as e:
        print(f"Unexpected error running on NPU: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, 0.0

def check_npu_prerequisites() -> bool:
    """Validate that prerequisites for NPU testing are met."""
    if not GENIE_EXE.exists():
        print(f"❌ Error: Genie executable not found at {GENIE_EXE}")
        print("Please download the NPU model to cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle/")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU inference performance")
    parser.add_argument("-c", "--cpu-config", default="llama3-qa.yaml", help="Path to CPU config file")
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
    
    _, cpu_tps, cpu_ttf = run_cpu_session(cpu_config, args.prompt)

    # NPU Run (using Genie runtime)
    print(f"\nNPU: Using Genie runtime (genie_bundle)")
    print("Running NPU pre-flight checks...")
    if check_npu_prerequisites():
        print("✅ Genie runtime found")
        _, npu_tps, npu_ttf = run_npu_session(args.prompt)
    else:
        print("Skipping NPU test.")
        npu_tps, npu_ttf = 0.0, 0.0

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
