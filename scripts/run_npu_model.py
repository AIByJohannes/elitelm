#!/usr/bin/env python3
"""
Simple wrapper to run the Llama 3.2 3B model on Snapdragon X Elite NPU via Genie.

This script provides an easy Python interface to the genie-t2t-run executable.
"""

import subprocess
import sys
import time
from pathlib import Path

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).parent.parent
GENIE_BUNDLE_DIR = PROJECT_ROOT / "cpu_and_mobile" / "llama-3.2-3b-npu-complete" / "genie_bundle"
GENIE_EXE = GENIE_BUNDLE_DIR / "genie-t2t-run.exe"
GENIE_CONFIG = GENIE_BUNDLE_DIR / "genie_config.json"


def format_llama3_prompt(user_message: str, system_message: str = None) -> str:
    """Format a prompt in Llama 3.2 format."""
    if system_message:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    else:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )


def run_npu_inference(prompt: str, verbose: bool = True) -> tuple[str, float]:
    """
    Run inference on NPU using Genie.

    Args:
        prompt: The input prompt (can be plain text or Llama 3.2 formatted)
        verbose: Whether to print verbose output

    Returns:
        Tuple of (generated_text, inference_time_seconds)
    """
    if not GENIE_EXE.exists():
        raise FileNotFoundError(
            f"Genie executable not found at {GENIE_EXE}. "
            "Please download the NPU model first."
        )

    # If prompt doesn't contain Llama 3.2 format markers, format it
    if "<|begin_of_text|>" not in prompt:
        prompt = format_llama3_prompt(prompt)

    start_time = time.time()

    try:
        result = subprocess.run(
            [str(GENIE_EXE), "-c", str(GENIE_CONFIG), "-p", prompt],
            cwd=str(GENIE_BUNDLE_DIR),
            capture_output=True,
            text=True,
            check=True
        )

        end_time = time.time()
        inference_time = end_time - start_time

        output = result.stdout

        # Extract the generated text between [BEGIN]: and [END]
        if "[BEGIN]:" in output and "[END]" in output:
            start_idx = output.index("[BEGIN]:") + len("[BEGIN]:")
            end_idx = output.index("[END]")
            generated_text = output[start_idx:end_idx].strip()
        else:
            generated_text = output

        if verbose:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")
            print(f"Generated: {generated_text}")
            print(f"{'='*60}")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"{'='*60}\n")

        return generated_text, inference_time

    except subprocess.CalledProcessError as e:
        print(f"Error running Genie: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise


def main():
    """Example usage of the NPU inference wrapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Llama 3.2 3B on Snapdragon X Elite NPU"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is the capital of France?",
        help="Input prompt (default: 'What is the capital of France?')"
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System message (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    try:
        if args.system:
            prompt = format_llama3_prompt(args.prompt, args.system)
        else:
            prompt = args.prompt

        text, time_taken = run_npu_inference(prompt, verbose=not args.quiet)

        if args.quiet:
            print(text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
