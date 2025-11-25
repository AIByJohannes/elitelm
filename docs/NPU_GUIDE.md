# NPU Guide for EliteLM

> ⚠️ **Info: NPU inference is currently NOT recommended for most use cases.**
>
> My benchmarks show that the **CPU is significantly faster** than the NPU for LLM inference with the current Genie runtime:
>
> | Metric            | CPU        | NPU        | Speedup   |
> |-------------------|------------|------------|-----------|
> | Tokens/Sec        | 36.64      | 1.08       | 0.03x     |
> | Time to First (s) | 0.0000     | 0.9710     | 0.00x     |
>
> **The CPU is ~34x faster than the NPU** in my tests. This is because Qualcomm optimized NPU for power efficiency, sacrificing generation speed. 
>
> **Recommendation**: Use CPU inference (`chat_cli.py` or `elitelm.ChatSession`) for best performance. The NPU scripts are provided for experimental purposes only.

---

This guide details how to run Large Language Models (LLMs) on the **Qualcomm Hexagon NPU** using **EliteLM**. It leverages the **Genie** (Gen AI Inference Extensions) architecture directly via the `genie-t2t-run` executable for experimental NPU inference on Snapdragon X Elite devices.

## 1. Why Use the NPU? (Experimental)

> ⚠️ **Note**: The theoretical benefits below are NOT currently realized with the Genie subprocess workflow. See the warning above.

The Hexagon NPU (Neural Processing Unit) is a specialized accelerator for INT8 matrix operations. In theory, compared to CPU inference:
*   **Higher Throughput**: Potentially faster token generation (in optimized scenarios).
*   **Lower Latency**: Faster prompt processing in "burst" mode (when properly warmed up).
*   **Power Efficiency**: Offloads compute from the CPU, extending battery life.

**Current Reality**: The overhead of subprocess invocation and lack of persistent model loading negates these benefits. Future versions may integrate a persistent NPU server or native Python bindings.

### Genie Native vs. ONNX Runtime
EliteLM now uses the **Genie Native** workflow, which differs from `onnxruntime-genai`:
*   **Genie Native**: Uses the `genie-t2t-run` executable directly. This offers the most direct path to the NPU hardware and often supports features or optimizations before they reach ONNX Runtime.
*   **Context Binaries**: Executes the entire model as a pre-compiled "context binary" on the NPU.

---

## 2. Prerequisites

### Hardware & OS
*   **Device**: Snapdragon X Elite or X Plus (e.g., Surface Laptop 7, Dell XPS 13 9345).
*   **OS**: Windows 11 on Arm.
*   **Memory**: At least 16GB RAM recommended (NPU shares system memory).

### Software Dependencies
1.  **Python 3.11**: The supported version for the Qualcomm AI stack.
2.  **Visual C++ Redistributable (ARM64)**: Required for QNN libraries.
3.  **Qualcomm AI Engine Direct SDK (QNN)**:
    *   Download from [Qualcomm Developer Network](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).
    *   Extract to `qairt/` in the project root (e.g., `qairt/2.37.0.250724`).

### Model Requirements (Critical)
You need a **Genie-compatible model bundle** containing **QNN context binaries** and the Genie executable.

**Recommended Model (Llama 3.2 3B):**
We recommend using the `Volko76/Llama-3.2-3B-Genie-Compatible-QNN-Binaries` repository or generating your own using Qualcomm AI Hub.

The expected structure in `cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle` is:
*   `genie-t2t-run.exe`: The inference executable.
*   `genie_config.json`: Configuration for the model.
*   `htp_backend_ext_config.json`: Backend configuration.
*   `tokenizer.json`: The model's tokenizer.
*   `*.bin`: The compiled model context binaries.
*   `*.dll`: Required QNN and Genie libraries (copied from SDK).

---

## 3. Configuration

The `scripts/run_npu_model.py` script is hardcoded to look for the Genie bundle in:
`cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle`

Ensure your model files are placed there.

---

## 4. Running Inference (Experimental)

> ⚠️ **Reminder**: CPU inference is recommended. Use `python chat_cli.py` for best performance.

### Single Prompt (NPU)
Use the wrapper script to run a prompt on the NPU:
```bash
python scripts/run_npu_model.py "Why is the sky blue?"
```

This script wraps `genie-t2t-run.exe`, handling prompt formatting (Llama 3.2 style) and output parsing.

### CPU vs NPU Comparison
To benchmark CPU vs NPU performance:
```bash
python scripts/compare_cpu_npu.py "Your prompt here"
```

### System Prompts
You can also provide a system prompt:
```bash
python scripts/run_npu_model.py "Who are you?" --system "You are a helpful assistant named EliteLM."
```

---

## 5. Technical Details

### The "Context Binary"
The files named `prompt_*_qnn_ctx.onnx` and `token_*_qnn_ctx.onnx` are wrappers around **QNN Context Binaries**.
## 5. Technical Details

### The "Context Binary"
The files named `*.bin` in the Genie bundle are **QNN Context Binaries**.
*   These contain the model graph *compiled specifically for the Hexagon DSP*.
*   They are non-portable (specific to the Snapdragon architecture).
*   `genie-t2t-run` loads these binaries and delegates execution directly to the QNN HTP backend.

### Memory Constraints
The Hexagon NPU has limited directly addressable memory (often ~4GB for the NPU subsystem).
*   **Model Size**: 3B parameter models (quantized to INT8) fit comfortably.
*   **Quantization**: INT8 is mandatory for best performance.

---

## 6. Troubleshooting

### Common Errors

**1. `Genie executable not found`**
*   **Cause**: The `genie-t2t-run.exe` file is missing from the bundle directory.
*   **Fix**: Ensure you have copied the executable and all required DLLs to `cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle`.

**2. `Unable to load backend` / `QNN HTP backend failed to load`**
*   **Cause**: Missing DLL dependencies in the bundle folder.
*   **Fix**:
    *   Install **Visual C++ Redistributable for ARM64**.
    *   Ensure all DLLs from the QNN SDK (`lib/hexagon-v73/unsigned/*` and `lib/aarch64-windows-msvc/*`) are copied to the bundle folder.

**3. `[WARN] "Unable to initialize logging in backend extensions."`**
*   **Cause**: Benign warning from the QNN backend.
*   **Fix**: Ignore it.

**4. Slow Performance (First Run)**
*   **Cause**: Graph initialization and caching.
*   **Fix**: The first prompt may take a few seconds to start. Subsequent prompts will be much faster.

