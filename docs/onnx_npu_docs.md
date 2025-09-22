# Running LLMs on Snapdragon NPUs with Python

## Overview

This document outlines the process for running Large Language Models (LLMs) on the Hexagon NPU of Snapdragon devices using Python. While the native Qualcomm Neural Processing (QNN) SDK is built around a C/C++ API, Python developers can access the NPU by using the ONNX Runtime, which acts as the primary bridge.

The recommended and most common method is to use the ONNX Runtime's QNN Execution Provider, which allows Python applications to execute specially prepared ONNX models on the NPU.

---

## Core Technology Stack

### Qualcomm Neural Processing SDK (QNN)

The QNN SDK is the foundational toolkit for running models on Snapdragon NPUs. Its native interface is a C-style API designed for portability and high performance. Direct Python bindings for the core runtime are not the primary method of interaction; instead, access is typically mediated through higher-level libraries.

### ONNX Runtime with QNN Execution Provider

ONNX Runtime provides a standardized way to run ONNX models across different hardware platforms. For Snapdragons, it uses the **QNN Execution Provider (EP)** to delegate model inference to the Hexagon NPU (also known as the HTP backend).

- **`onnxruntime-qnn`**: This Python package includes the QNN EP and is essential for this workflow.
- **Platform Support**: Pre-built wheels are available for Windows on Arm (WoA), which is the primary target for Python development. Android usage typically requires building from source.

---

## Model Requirements

To run a model on the Hexagon NPU, it must meet two key criteria:

1.  **Quantization**: The model must be quantized. The NPU is optimized for integer arithmetic and does not execute floating-point models.
2.  **Static Shapes**: The model must have fixed input and output shapes. Dynamic sequence lengths, common in LLMs, are not supported and must be handled by padding inputs to a fixed size before inference.

The [ONNX Runtime GenAI documentation](https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html) provides guidance on converting and preparing LLM assets for Snapdragon NPUs.

---

## Python Implementation Guide

The following example demonstrates how to run a streaming LLM on a Snapdragon NPU using the `onnxruntime-genai` library, which is built on top of the ONNX Runtime.

### Prerequisites

1.  **Install Libraries**:
    ```bash
    pip install onnxruntime-genai onnxruntime-qnn
    ```
2.  **Model Assets**: You need a model that has been converted to ONNX and prepared for the QNN EP. This includes a `genai_config.json` file and the associated ONNX binaries.
3.  **QNN SDK**: The path to the QNN HTP backend library (`QnnHtp.dll` on Windows or `libQnnHtp.so` on Linux/Android) is required.

### Code Example

```python
# Minimal streaming text-generation on Snapdragon NPU via ORT-GenAI + QNN EP.
import os
import onnxruntime_genai as og
import numpy as np

# Optional: Verify that the QNN Execution Provider is available
print("QNN available:", og.is_qnn_available())

# 1. Load the model configuration generated during the model preparation step
config = og.Config("path/to/genai_config.json")

# 2. Configure the ONNX Runtime to use the QNN Execution Provider
config.clear_providers()
config.append_provider("QNNExecutionProvider")

# 3. Specify the path to the QNN HTP backend library from the QNN SDK
#    - Windows ARM64: "C:\Qualcomm\AIStack\QAIRT\<ver>\lib\arm64x-windows-msvc\QnnHtp.dll"
#    - Linux/Android: "/path/to/libQnnHtp.so"
config.set_provider_option("backend_path", "path/to/QnnHtp.dll_or_libQnnHtp.so")

# 4. Load the model with the specified configuration
model = og.Model(config)

# 5. Tokenize the prompt
tokenizer = og.Tokenizer(model)
prompt = "You are a helpful assistant. Explain quantization in simple terms:"
input_ids = tokenizer.encode(prompt)

# 6. Set up the generator with search options
params = og.GeneratorParams(model)
params.set_model_input("input_ids", np.array(input_ids, dtype=np.int32))
params.set_search_options(temperature=0.7, top_p=0.9)

# 7. Run the generator and stream the output tokens
generator = og.Generator(model, params)
stream = tokenizer.create_stream()

print("Assistant: ", end="")
while not generator.is_done():
    generator.generate_next_token()
    next_tokens = generator.get_next_tokens()
    if next_tokens.size > 0:
        print(stream.decode(int(next_tokens[-1])), end="", flush=True)

print()
```

### Code Explanation

1.  **Configuration**: The `genai_config.json` file, created when preparing the model, is loaded.
2.  **Provider Setup**: The execution provider is explicitly set to `QNNExecutionProvider`. This directs the ONNX Runtime to use the QNN backend instead of the default CPU provider.
3.  **Backend Path**: The `backend_path` option tells the QNN EP where to find the specific NPU driver library (for the Hexagon Tile Processor).
4.  **Inference**: The `onnxruntime_genai` library handles the tokenization, generation loop, and decoding, providing a simple streaming interface.

---

## Alternative Methods

### Qualcomm AI AppBuilder

Qualcomm provides the **QAI AppBuilder**, which includes Python wheels that wrap the QNN SDK. This is another viable path for running models from Python, especially on Windows on Snapdragon, and it provides its own helper utilities and examples.

---

## Summary

- **Primary Method**: Use Python with the `onnxruntime-qnn` package to run inference on the Snapdragon NPU.
- **Core Component**: The ONNX Runtime's QNN Execution Provider bridges the gap between Python and the native QNN C++ SDK.
- **Requirements**: Models must be quantized and use static input shapes.
- **C/C++**: For maximum control and direct access to the QNN API, C++ remains the canonical interface.

---

## References

- [ONNX Runtime: QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [ONNX Runtime GenAI: Python API](https://onnxruntime.ai/docs/genai/api/python.html)
- [ONNX Runtime GenAI: Build Models for Snapdragon](https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html)
- [Qualcomm QNN API Overview](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/api_overview.html?product=1601111740009302)
- [Qualcomm AI Engine Direct Helper Releases (AppBuilder)](https://github.com/quic/ai-engine-direct-helper/releases)
- [Microsoft Tech Community: "Hello World" NPU on Surface](https://techcommunity.microsoft.com/blog/surfaceitpro/unlocking-the-power-of-npu-on-surface-our-“hello-world”-journey/4149473)
