# NPU Guide for EliteLM

> ⚠️ **Info: NPU inference is currently NOT recommended for most use cases.**
>
> Benchmarks show that the **CPU is significantly faster** than the NPU for LLM inference with the current Genie runtime:
>
> | Metric            | CPU        | NPU        | Speedup   |
> |-------------------|------------|------------|-----------|
> | Tokens/Sec        | 36.64      | 1.08       | 0.03x     |
> | Time to First (s) | 0.0000     | 0.9710     | 0.00x     |
>
> **The CPU is ~34x faster than the NPU** in tests. This is because Qualcomm optimized the NPU for power efficiency, sacrificing generation speed. 
>
> **Recommendation**: Use CPU inference (`crates/elitelm-backend-llamacpp` or `llamacpp_cpu` backend) for best performance. The NPU backend is provided for experimental purposes only.

---

This guide details how to run Large Language Models (LLMs) on the **Qualcomm Hexagon NPU** using **EliteLM**. It leverages the **Genie** (Gen AI Inference Extensions) architecture natively via the `Genie.dll` dynamic library.

## 1. Why Use the NPU? (Experimental)

The Hexagon NPU (Neural Processing Unit) is a specialized accelerator for INT8 matrix operations. It offloads compute from the CPU, extending battery life.
The power efficiency comes at the cost of performance. 

### Genie Native vs. ONNX Runtime
EliteLM uses the **Genie Native** workflow:
*   **Genie Native**: Interacts directly with `Genie.dll` in-process using dynamic loading (`libloading` in Rust).
*   **Context Binaries**: Executes the entire model as pre-compiled "context binaries" (`*.bin`) on the NPU.

---

## 2. Prerequisites

### Hardware
*   **Device**: Snapdragon X Elite or X Plus (e.g., Surface Laptop 7, Dell XPS 13 9345).
*   **OS**: Windows 11 on Arm.
*   **Memory**: At least 16GB RAM recommended (NPU shares system memory).

### Software
1.  **Visual C++ Redistributable (ARM64)**: Required for QNN libraries.
2.  **Qualcomm AI Engine Direct SDK (QNN)**:
    *   Download from [Qualcomm Developer Network](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).
    *   Extract to `qairt/` in the project root (e.g., `qairt/2.37.0.250724`).

### Model Requirements 
You need a **Genie-compatible model bundle** containing **QNN context binaries** and tokenizers.

**Recommended Model (Llama 3.2 3B):**
We recommend using the `Volko76/Llama-3.2-3B-Genie-Compatible-QNN-Binaries` repository or generating your own using Qualcomm AI Hub.

### Genie Bundle Layout
The runtime expects the following file structure in the model directory (e.g., `cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle`):

#### Executables & Configs
*   `genie_config.json`: Configuration defining model parameters and context binary paths.
*   `htp_backend_ext_config.json`: QNN backend configuration.
*   `tokenizer.json`: Hugging Face style tokenizer definition.

#### Context Binaries
*   `*.bin` (e.g., `llama_v3_2_3b_instruct_part_1_of_3.bin`): The model compiled for Hexagon.

#### Required DLLs
These libraries must be copied to the bundle directory (typically automated by `prepare-genie-bundle` CLI command):

| Category | Files |
| :--- | :--- |
| **Core QNN** | `QnnSystem.dll`, `QnnHtp.dll`, `QnnHtpPrepare.dll` |
| **Genie** | `Genie.dll`, `QnnGenAiTransformer.dll`, `QnnGenAiTransformerModel.dll` |
| **Hexagon V73** | `QnnHtpv73Stub.dll`, `QnnHtpv73CalculatorStub.dll` |
| **Extensions** | `QnnHtpNetRunExtensions.dll` |
| **Helpers** | `PlatformValidatorShared.dll` |

> **Note**: The "V73" in filenames refers to the Hexagon DSP version (Snapdragon X Elite uses v73). Ensure these match your specific device hardware.

---

## 3. Preparation & Configuration

Instead of manually copying QNN SDK libraries and preparing configuration files, use the EliteLM CLI tool to initialize the bundle automatically:

```bash
cargo run --bin elitelm-cli -- prepare-genie-bundle --config elitelm.genie.example.yaml --backend genie_npu
```

This subcommand will:
1. Parse the template files listed in `elitelm.genie.example.yaml`.
2. Generate the actual `genie_config.json` and `htp_backend_ext_config.json` in the bundle directory.
3. Locate the QNN SDK root path and copy all required `*.dll` dependencies into the bundle directory.

---

## 4. Running Inference (Experimental)

### Running Prompts via CLI
To run a prompt on the NPU backend:

```bash
cargo run --bin elitelm-cli -- run genie_npu "Why is the sky blue?" --config elitelm.genie.example.yaml
```

### Running the API Server
To expose the NPU backend via the Axum OpenAI-compatible server:

```bash
cargo run --bin elitelm-server -- serve --config elitelm.genie.example.yaml
```

### CPU vs NPU Benchmark
To run a comparison benchmark:

```bash
cargo run --bin elitelm-cli -- benchmark --config elitelm.genie.example.yaml
```

---

## 5. Technical Details

### The "Context Binary"
The files named `*.bin` in the Genie bundle are **QNN Context Binaries**.
*   These contain the model graph *compiled specifically for the Hexagon DSP*.
*   They are non-portable (specific to the Snapdragon architecture).
*   `Genie.dll` loads these binaries and delegates execution directly to the QNN HTP backend.

### Memory Constraints
The Hexagon NPU has limited directly addressable memory (often ~4GB for the NPU subsystem).
*   **Model Size**: 3B parameter models (quantized to INT8) fit comfortably.
*   **Quantization**: INT8 is mandatory for best performance.

---

## 6. Troubleshooting

### Common Errors

**1. `Genie native library does not exist`**
*   **Cause**: `Genie.dll` is missing from the bundle folder.
*   **Fix**: Run the `prepare-genie-bundle` subcommand to correctly copy all dependencies from your QNN SDK.

**2. `QNN HTP backend failed to load` / `Unable to load backend`**
*   **Cause**: Missing or mismatched DLL dependencies in the bundle folder.
*   **Fix**:
    *   Install **Visual C++ Redistributable for ARM64**.
    *   Verify the `qairt_sdk_root` in your config points to a valid Qualcomm AI SDK (e.g., `2.37.0.250724`).
    *   Run `prepare-genie-bundle` again to refresh copied DLLs.

**3. `[WARN] "Unable to initialize logging in backend extensions."`**
*   **Cause**: Benign warning from the QNN backend.
*   **Fix**: Ignore it.