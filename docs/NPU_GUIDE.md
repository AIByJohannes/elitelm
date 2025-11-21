# NPU Guide for EliteLM

This guide details how to run Large Language Models (LLMs) on the **Qualcomm Hexagon NPU** using **EliteLM**. It leverages the **Genie** (Gen AI Inference Extensions) architecture via `onnxruntime-genai` to achieve high-performance, low-power inference on Snapdragon X Elite devices.

## 1. Why Use the NPU?

The Hexagon NPU (Neural Processing Unit) is a specialized accelerator for INT8 matrix operations. Compared to CPU inference:
*   **Higher Throughput**: Significantly faster token generation (often 4-5x speedup).
*   **Lower Latency**: Faster prompt processing, especially in "burst" mode.
*   **Power Efficiency**: Offloads compute from the CPU, extending battery life.

### Genie vs. Standard ONNX
EliteLM uses the **Genie** workflow (via QNN), which is distinct from running standard ONNX models:
*   **Standard ONNX**: Runs operators individually; may fall back to CPU if an op isn't supported on NPU.
*   **Genie / QNN Context**: Executes the entire model (or large subgraphs) as a pre-compiled "context binary" on the NPU. This minimizes overhead and maximizes hardware utilization.

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
4.  **Python Packages**:
    ```bash
    pip install onnxruntime-genai onnxruntime-qnn
    ```

### Model Requirements (Critical)
You cannot use standard `.onnx` or `.gguf` files directly on the NPU. You need a **Genie-compatible model** containing **QNN context binaries**.

**Recommended Model (Llama 3.2 3B):**
We recommend the `llmware/llama-3.2-3b-onnx-qnn` repository, which contains the complete set of compiled binaries.

```bash
# Download to the cpu_and_mobile directory
huggingface-cli download llmware/llama-3.2-3b-onnx-qnn --local-dir ./cpu_and_mobile/llama-3.2-3b-onnx-qnn
```

> **Warning**: The `onnx-community/Llama-3.2-3B-instruct-hexagon-npu-assets` repository contains *only* config files and tokenizers. It is **missing** the actual inference binaries. Do not use it alone.

---

## 3. Configuration

Configure `llama3-npu.yaml` to point to your downloaded model and select the `qnn` device.

```yaml
# Path to the directory containing .onnx context binaries and genai_config.json
model: ./cpu_and_mobile/llama-3.2-3b-onnx-qnn

# Device selection: 'qnn' for NPU, 'cpu' for CPU
device: qnn

generation:
  max_length: 2048
  do_sample: true
  top_p: 0.95
  temperature: 0.8

qnn:
  # Path to your extracted QNN SDK
  sdk_root: ./qairt/2.37.0.250724
  # Optional: override backend path (usually auto-detected)
  backend: null
```

---

## 4. Running Inference

### Single Prompt (NPU)
Use the wrapper script to run a quick check:
```bash
python run_npu_model.py "Why is the sky blue?"
```

### CPU vs. NPU Benchmark
Compare performance directly:
```bash
python compare_cpu_npu.py "Explain quantum computing in simple terms."
```
*Look for "Gen TPS" (Tokens Per Second) in the output. NPU should be significantly higher.*

### API Server
Run the OpenAI-compatible API on the NPU:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
*(Ensure `llama3-qa.yaml` or your active config is set to `device: qnn`)*

---

## 5. Technical Details

### The "Context Binary"
The files named `prompt_*_qnn_ctx.onnx` and `token_*_qnn_ctx.onnx` are wrappers around **QNN Context Binaries**.
*   These contain the model graph *compiled specifically for the Hexagon DSP*.
*   They are non-portable (specific to the Snapdragon architecture).
*   `onnxruntime-genai` loads these binaries and delegates execution directly to the QNN HTP backend.

### Memory Constraints
The Hexagon NPU has limited directly addressable memory (often ~4GB for the NPU subsystem).
*   **Model Size**: 3B parameter models (quantized to INT8) fit comfortably. 7B models may require aggressive quantization or hybrid scheduling (splitting layers between CPU/NPU), which Genie handles but can impact performance.
*   **Quantization**: INT8 is mandatory for best performance. FP16 is supported but much slower on this NPU generation.

---

## 6. Troubleshooting

### Common Errors

**1. `File doesn't exist` or `Load model ... failed`**
*   **Cause**: Missing QNN context binaries.
*   **Fix**: You likely downloaded the `onnx-community` assets repo which is incomplete. Download the full model from `llmware/llama-3.2-3b-onnx-qnn` as described in [Prerequisites](#prerequisites).

**2. `Unknown value "sliding_window_key_value_cache"`**
*   **Cause**: Your `onnxruntime-genai` version is older than the model's config schema.
*   **Fix**:
    *   **Option A**: Upgrade: `pip install --upgrade onnxruntime-genai` (v0.5.0+).
    *   **Option B**: EliteLM automatically patches this in memory for older versions, so simply re-running `run_npu_model.py` should work.

**3. `Unable to load backend` / `QNN HTP backend failed to load`**
*   **Cause**: QNN SDK not found or missing DLL dependencies.
*   **Fix**:
    *   Verify `sdk_root` in `llama3-npu.yaml`.
    *   Install **Visual C++ Redistributable for ARM64**.
    *   Ensure `QnnHtp.dll` is present in the SDK `bin` folder.

**4. `Unknown chip model name ...` (cpuinfo warning)**
*   **Cause**: `py-cpuinfo` library doesn't yet recognize Snapdragon X Elite.
*   **Fix**: Ignore it. This is a benign warning and does not affect NPU inference.

**5. Slow Performance (First Run)**
*   **Cause**: Graph initialization and caching.
*   **Fix**: The first prompt may take a few seconds to start. Subsequent prompts will be much faster.
