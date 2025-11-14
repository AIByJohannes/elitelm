# NPU Guide for EliteLM

This guide provides a comprehensive overview of how to set up, use, and test the Hexagon NPU with EliteLM on Snapdragon-powered devices.

## Table of Contents

1.  [**Introduction to NPU Execution**](#introduction-to-npu-execution)
    *   [Why Use the NPU?](#why-use-the-npu)
    *   [Core Technology Stack](#core-technology-stack)
2.  [**Setup and Configuration**](#setup-and-configuration)
    *   [Prerequisites](#prerequisites)
    *   [Model Requirements](#model-requirements)
    *   [Configuration](#configuration)
3.  [**Running on the NPU**](#running-on-the-npu)
    *   [Python Wrapper (Recommended)](#python-wrapper-recommended)
    *   [Direct Genie Execution](#direct-genie-execution)
4.  [**NPU Verification and Testing**](#npu-verification-and-testing)
    *   [Three-Tier Testing Strategy](#three-tier-testing-strategy)
    *   [How to Run Tests](#how-to-run-tests)
    *   [Interpreting Test Results](#interpreting-test-results)
5.  [**Troubleshooting**](#troubleshooting)
    *   [Common Errors](#common-errors)
    *   [Understanding Skip Messages](#understanding-skip-messages)
6.  [**Technical Details**](#technical-details)
    *   [Architecture](#architecture)
    *   [Observable Signals for Verification](#observable-signals-for-verification)

---

## 1. Introduction to NPU Execution

### Why Use the NPU?

The Neural Processing Unit (NPU) is a specialized processor designed to accelerate machine learning workloads. By offloading model inference to the NPU, you can achieve significant performance gains and power savings compared to running on the CPU.

### Core Technology Stack

-   **Qualcomm Neural Processing (QNN) SDK**: The foundational toolkit for running models on Snapdragon NPUs.
-   **ONNX Runtime with QNN Execution Provider**: The primary bridge for Python developers to access the NPU. `onnxruntime-qnn` is the key Python package.
-   **Genie Runtime**: Qualcomm's official high-performance inference engine for running LLMs on the Hexagon NPU.

---

## 2. Setup and Configuration

### Prerequisites

1.  **Hardware**: A PC with a Snapdragon X Elite or Snapdragon X Plus chip.
2.  **Python**: Python 3.11.
3.  **QNN SDK**: Download and extract the Qualcomm AI Engine Direct SDK (QNN SDK). You will need the path to the SDK directory.
4.  **Python Packages**:
    ```bash
    pip install onnxruntime-genai onnxruntime-qnn
    ```
5.  **Visual C++ Redistributable**: Ensure you have the latest Visual C++ Redistributable (ARM64) installed.

### Model Requirements

To run on the Hexagon NPU, a model must be:

1.  **Quantized**: The NPU is optimized for integer arithmetic.
2.  **Static-Shaped**: The model must have fixed input and output shapes.

For this project, a compatible Llama 3.2 3B model is provided under `cpu_and_mobile/llama-3.2-3b-npu-complete/`.

> **Model file naming**: We standardize on `model.onnx` for the exported graph. Older drops may still include `ort_model.onnx`; rename or copy it to `model.onnx` so automation (CLI, tests, and utilities) can find the model without extra configuration. The tooling will still fall back to the legacy filename to ease migrations.

### Configuration

In your `llama3-qa.yaml` file, configure the `device` and `qnn` sections to enable NPU execution.

```yaml
# ... other settings
runtime:
  device: qnn # Use 'qnn' for NPU, 'cpu' for CPU
  # ...

qnn:
  # Path to your extracted QNN SDK
  sdk_root: ./qairt/2.37.0.250724
  # Optional: override backend path if auto-detection fails
  backend: null
```

---

## 3. Running on the NPU

### Python Wrapper (Recommended)

The `run_npu_model.py` script provides an easy way to interact with the NPU-accelerated model.

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Run with a default prompt
python run_npu_model.py

# Run with a custom prompt
python run_npu_model.py "What is artificial intelligence?"
```

### Direct Genie Execution

You can also run the model directly using the Genie runtime executable.

```bash
cd cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle
./genie-t2t-run.exe -c genie_config.json -p "Your prompt here"
```

---

## 4. NPU Verification and Testing

A comprehensive **three-tier testing strategy** is implemented to validate that the NPU is being used correctly.

### Three-Tier Testing Strategy

-   **Tier 1: Build-Time Availability**: Verifies that QNN support is compiled into `onnxruntime-genai`. Runs on all systems.
-   **Tier 2: Session Configuration**: Validates that the QNN provider is configured correctly, without requiring hardware. Runs on all systems.
-   **Tier 3: Actual NPU Execution**: Hardware-specific integration tests that confirm the model is executing on the NPU and not falling back to the CPU. These tests only run on Snapdragon devices.

### How to Run Tests

-   **Run all tests (Tier 3 will skip on non-NPU machines)**:
    ```bash
    pytest tests/
    ```
-   **Run only NPU-specific tests (requires NPU hardware)**:
    ```bash
    pytest tests/test_npu_verification.py -v -s
    ```
-   **Run only non-hardware tests (for CI/CD on CPU-only runners)**:
    ```bash
    pytest tests/ -m "not requires_npu"
    ```

### Interpreting Test Results

-   ✅ **All Pass**: Your system is correctly configured, and the NPU is working.
-   ⏭️ **Some Skips (Tier 3)**: Normal on CPU-only machines. Not an error.
-   ❌ **Failures**: Indicate a problem with your setup, the model, or the hardware.

---

## 5. Troubleshooting

### Common Errors

-   **`Unable to load backend` or `QNN HTP backend failed to load`**:
    -   Ensure the Visual C++ Redistributable (ARM64) is installed.
    -   Verify the `sdk_root` path in your `llama3-qa.yaml` is correct.
    -   Check that `QnnHtp.dll` exists in the specified SDK path.
-   **`RuntimeError: Operator X not supported by QNN EP`**:
    -   The model is not compatible with the NPU. Ensure you are using a correctly quantized model with static shapes.
-   **Slow Performance**:
    -   The first run has cold-start overhead. Subsequent runs should be faster.
    -   Long prompts naturally take longer to process.

### Understanding Skip Messages

-   **`"QNN not available..."`**: `onnxruntime-genai` was not installed with QNN support. Install `onnxruntime-qnn`.
-   **`"QNN SDK not found..."`**: The QNN SDK path is incorrect or the `QNN_SDK_ROOT` environment variable is not set.
-   **`"Test model not found..."`**: The required quantized test model is missing from the `cpu_and_mobile/` directory.

---

## 6. Technical Details

### Architecture

The NPU execution flow is as follows:
1.  The user prompt is tokenized.
2.  The tokenized input is fed into the QNN context binaries (`.bin` files).
3.  The Genie runtime orchestrates the execution on the Hexagon NPU.
4.  The NPU generates output tokens, which are detokenized into text.

Genie is used for its direct optimization for the Hexagon NPU and native support for QNN context binaries, which is Qualcomm's recommended approach.

### Observable Signals for Verification

To be certain the NPU is active, the test suite looks for these signals:

-   **Performance Difference**: The execution time on the NPU should be measurably different from the CPU.
-   **Model Loading**: The model loads successfully using the QNN provider without exceptions.
-   **Inference Execution**: The model generates output without runtime errors.
-   **Output Validity**: The generated tokens are valid and can be decoded into reasonable text.

By combining these checks, we gain high confidence that the NPU is not only active but also producing correct results.
