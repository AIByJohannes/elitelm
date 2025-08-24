# Project Details

This document provides a more detailed look into the architecture and components of the EliteLM project.

## Project Overview

EliteLM is designed to be a lightweight and high-performance inference server for large language models, with a focus on leveraging the Qualcomm Hexagon NPU for hardware acceleration. The project is divided into two main parts:

1.  **A command-line interface (CLI)** for interactive testing and experimentation (`llama3-qa.py`).
2.  **A FastAPI server** for exposing the model's functionality as a web API (`api.py`).

## Core Components

### `llama3-qa.py`

This script is the heart of the project. It contains the `QA` class, which is responsible for loading the model, tokenizing the input, and generating the output.

#### The `QA` class

*   **`__init__(self, model_dir, backend)`:** The constructor initializes the `QA` class. It loads the tokenizer and the ONNX model, and sets up the inference session with the specified backend.
*   **`load_model(self)`:** This method loads the tokenizer from the model directory.
*   **`create_inference_session(self, backend)`:** This method creates an ONNX Runtime inference session. It configures the session to use the QNN Execution Provider with the specified backend library (`.dll`).
*   **`generate(self, prompt, top_k, top_p, temperature, repetition_penalty)`:** This method takes a prompt and generation parameters, and returns the generated text. It tokenizes the input, runs the model, and decodes the output.

### ONNX Runtime and QNN Backend

The project uses the [ONNX Runtime](https://onnxruntime.ai/) to run the ONNX model. To leverage the Hexagon NPU, the ONNX Runtime is configured to use the **QNN Execution Provider**. This execution provider is a bridge between the ONNX Runtime and the Qualcomm AI Engine, which includes the Hexagon NPU.

The QNN backend is a library (e.g., `QnnHtp.dll`) that contains the implementation of the QNN Execution Provider. This library is provided by Qualcomm as part of their AI SDK.

## API Server (`api.py`)

The `api.py` file will contain the FastAPI server. The server will expose the functionality of the `QA` class as a web API. The planned endpoints are:

*   **`/generate` (POST):** This endpoint will take a JSON object with a `prompt` and generation parameters, and will return the generated text.

## How to run on NPU

To run the model on the Hexagon NPU, you need to ensure that you have the correct version of the Qualcomm AI Engine installed on your system, and that the `QnnHtp.dll` file is accessible. When you run the `llama3-qa.py` script with the `-b` argument pointing to the `QnnHtp.dll` file, the ONNX Runtime will automatically use the NPU for inference.

For more detailed instructions on setting up the NPU environment, please refer to the `docs/onnx_npu_docs.md` file.
