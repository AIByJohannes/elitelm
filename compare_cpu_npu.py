#!/usr/bin/env python3
"""
Compare CPU vs NPU inference performance for Llama models.

This script runs the same prompts on both CPU (via onnxruntime-genai)
and NPU (via Genie) to demonstrate the performance difference.
"""

import time
from pathlib import Path
import sys

# Import NPU wrapper
from run_npu_model import run_npu_inference, format_llama3_prompt

# Import CPU runtime
try:
    import onnxruntime_genai as og
except ImportError:
    print("Error: onnxruntime_genai not installed", file=sys.stderr)
    sys.exit(1)


def run_cpu_inference(model_path: str, prompt: str) -> tuple[str, float]:
    """
    Run inference on CPU using onnxruntime-genai.

    Args:
        model_path: Path to the ONNX model directory
        prompt: The input prompt

    Returns:
        Tuple of (generated_text, inference_time_seconds)
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    start_time = time.time()

    # Load model and tokenizer
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    # Tokenize input
    tokens = tokenizer.encode(prompt)

    # Generate
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=100)  # Limit for fair comparison

    # Set input tokens (batched)
    batched_tokens = tokens.reshape(1, -1) if tokens.ndim == 1 else tokens
    if hasattr(params, "set_model_input"):
        params.set_model_input("input_ids", batched_tokens)
    else:
        params.input_ids = tokens

    generator = og.Generator(model, params)

    # Generate tokens
    generated_text = ""
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        generated_text += tokenizer_stream.decode(int(new_token))

    end_time = time.time()
    inference_time = end_time - start_time

    return generated_text, inference_time


def main():
    """Compare CPU vs NPU performance."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare CPU vs NPU inference performance"
    )
    parser.add_argument(
        "--cpu-model",
        default="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        help="Path to CPU model directory"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is the capital of France?",
        help="Input prompt"
    )

    args = parser.parse_args()

    print("="*70)
    print("CPU vs NPU Performance Comparison")
    print("="*70)

    # Format prompt for Llama 3.2
    formatted_prompt = format_llama3_prompt(args.prompt)
    print(f"\nPrompt: {args.prompt}")

    # Run CPU inference
    print("\n" + "-"*70)
    print("Running on CPU...")
    print("-"*70)
    try:
        cpu_text, cpu_time = run_cpu_inference(args.cpu_model, formatted_prompt)
        print(f"CPU Output: {cpu_text[:200]}{'...' if len(cpu_text) > 200 else ''}")
        print(f"CPU Time: {cpu_time:.2f}s")
    except Exception as e:
        print(f"CPU Error: {e}")
        cpu_time = None

    # Run NPU inference
    print("\n" + "-"*70)
    print("Running on NPU...")
    print("-"*70)
    try:
        npu_text, npu_time = run_npu_inference(formatted_prompt, verbose=False)
        print(f"NPU Output: {npu_text[:200]}{'...' if len(npu_text) > 200 else ''}")
        print(f"NPU Time: {npu_time:.2f}s")
    except Exception as e:
        print(f"NPU Error: {e}")
        npu_time = None

    # Compare
    print("\n" + "="*70)
    print("Performance Summary")
    print("="*70)
    if cpu_time and npu_time:
        speedup = cpu_time / npu_time
        print(f"CPU Time:    {cpu_time:6.2f}s")
        print(f"NPU Time:    {npu_time:6.2f}s")
        print(f"Speedup:     {speedup:6.2f}x {'(NPU faster)' if speedup > 1 else '(CPU faster)'}")

        if speedup > 1:
            print(f"\n✅ NPU is {speedup:.2f}x faster than CPU!")
        else:
            print(f"\n⚠️  NPU is slower than CPU ({1/speedup:.2f}x)")
            print("Note: This may be due to:")
            print("  - Cold start overhead")
            print("  - Model optimization differences")
            print("  - Short prompt/generation length")
    elif cpu_time:
        print(f"CPU Time:    {cpu_time:6.2f}s")
        print("NPU Time:    Failed")
    elif npu_time:
        print("CPU Time:    Failed")
        print(f"NPU Time:    {npu_time:6.2f}s")
    else:
        print("Both CPU and NPU failed to run")

    print("="*70)


if __name__ == "__main__":
    main()
