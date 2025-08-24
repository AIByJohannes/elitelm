# ONNX on NPU Documentation

This document summarizes the findings for running ONNX models on Neural Processing Units (NPUs), specifically the Qualcomm Hexagon NPU.

### Common Concepts

Both Python and C++ rely on the same core components to run ONNX models on a Hexagon NPU:

*   **ONNX Runtime:** A high-performance inference engine for ONNX models. It uses "Execution Providers" to target specific hardware.
*   **QNN Execution Provider:** This is the specific Execution Provider for ONNX Runtime that targets Qualcomm's hardware, including the Hexagon NPU. It uses the Qualcomm AI Engine Direct SDK (QNN SDK) under the hood.
*   **Quantization:** For optimal performance on the NPU, models are often quantized (converted from 32-bit floating-point to 8-bit or 16-bit integers). This is a crucial step.

---

### Python

For Python, the process is generally more straightforward, especially for getting started.

*   **Key Library:** You'll primarily use the `onnxruntime-qnn` package, which can be installed via pip. This package includes the necessary components to run ONNX models on Qualcomm hardware.
    ```bash
    pip install onnxruntime-qnn
    ```
*   **Workflow:**
    1.  Convert your model to the ONNX format.
    2.  Quantize the model.
    3.  Use the `onnxruntime` Python API to load the model and run inference, making sure to specify the QNN Execution Provider.
*   **Resources:**
    *   **ONNX Runtime Documentation (QNN Execution Provider):** [https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
    *   **Qualcomm AI Hub:** [https://aihub.qualcomm.com/](https://aihub.qualcomm.com/) (Provides pre-optimized models and tutorials)
    *   **Qualcomm Developer Network:** [https://developer.qualcomm.com/docs/snpe/overview.html](https://developer.qualcomm.com/docs/snpe/overview.html)

---

### C++

For C++, you'll work more closely with the underlying SDKs, which gives you more control and potentially better performance.

*   **Key APIs:** You will use the ONNX Runtime C++ API to load and run your model.
*   **Workflow:**
    1.  Convert and quantize your ONNX model.
    2.  In your C++ application, include the ONNX Runtime headers.
    3.  When creating an inference session, specify the QNN Execution Provider to target the Hexagon NPU.
    4.  You will likely need to build ONNX Runtime from source and link it against the Qualcomm Neural Processing SDK.
*   **Resources:**
    *   **ONNX Runtime C++ API Documentation:** [https://onnxruntime.ai/docs/api/c.html](https://onnxruntime.ai/docs/api/c.html)
    *   **Qualcomm Neural Processing SDK Documentation:** [https://developer.qualcomm.com/docs/snpe/c-plus-plus-tutorial.html](https://developer.qualcomm.com/docs/snpe/c-plus-plus-tutorial.html)
    *   **Windows on Snapdragon AI Development:** [https://learn.microsoft.com/en-us/windows/ai/develop/snapdragon-npu-onnx-runtime](https://learn.microsoft.com/en-us/windows/ai/develop/snapdragon-npu-onnx-runtime)
