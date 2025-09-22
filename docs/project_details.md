# Project Details

This document provides a more detailed look into the architecture and components of the EliteLM project.

## Project Overview

EliteLM is designed to be a lightweight and high-performance inference server for large language models, with a focus on leveraging the Qualcomm Hexagon NPU for hardware acceleration. The project is divided into three main parts:

1.  **A reusable runtime module** that handles configuration parsing, model loading, and token generation (`elitelm.py`).
2.  **A command-line interface (CLI)** for interactive testing and experimentation built on top of the runtime (`chat_cli.py`).
3.  **A FastAPI server** for exposing the model's functionality as a web API (`api.py`).

## Core Components

### `elitelm.py`

This module houses the core runtime logic. It provides utilities for configuring the QNN execution provider, loading ONNX Runtime GenAI models, and orchestrating chat-style text generation. The primary entry point is the `ChatSession` class, which keeps track of chat history and performs streaming generation.

#### The `ChatSession` class

*   **`__init__(self, args)`**: Loads the model, tokenizer, and tokenizer stream based on a parsed configuration namespace. It also prepares generation search options and tracks chat history.
*   **`generate(self, user_text, *, timings=False, on_token=None)`**: Builds the prompt (including any previous turns), feeds it into the model, and streams decoded pieces back through the optional callback while collecting timing statistics. Returns a `GenerationResult` with the assistant reply and metrics.
*   **`reset_history(self)`**: Clears the accumulated chat history, allowing a fresh conversation without rebuilding the session.
*   **`device_label` property**: Convenience helper that reports whether the current session is running against the CPU or QNN backend.

In addition to `ChatSession`, the module exposes `load_config(path)` to parse the YAML runtime configuration and several helper functions used by both the CLI and future server work (for example `_configure_qnn_provider`).

### `chat_cli.py`

The CLI is a thin wrapper around `ChatSession`. It loads configuration from disk, prints helpful status messages when `runtime.verbose` is enabled, and streams generated text to STDOUT as tokens arrive. Control+C interruptions propagate through the session so you can stop a long generation without restarting the process.

### ONNX Runtime and QNN Backend

The project uses the [ONNX Runtime](https://onnxruntime.ai/) to run the ONNX model. To leverage the Hexagon NPU, the ONNX Runtime is configured to use the **QNN Execution Provider**. This execution provider is a bridge between the ONNX Runtime and the Qualcomm AI Engine, which includes the Hexagon NPU.

The QNN backend is a library (e.g., `QnnHtp.dll`) that contains the implementation of the QNN Execution Provider. This library is provided by Qualcomm as part of their AI SDK.

## API Server (`api.py`)

The `api.py` file will contain the FastAPI server. The server will expose the functionality of the `ChatSession` class as a web API. The planned endpoints are:

*   **`/generate` (POST):** This endpoint will take a JSON object with a `prompt` and generation parameters, and will return the generated text.

## How to run on NPU

To run the model on the Hexagon NPU, install the matching Qualcomm AI Engine SDK and update your YAML config with `device: qnn`. Populate the `qnn` block (typically `sdk_root` and optionally `backend`) so the runtime can locate the required DLLs and configure the QNN execution provider automatically.

For more detailed instructions on setting up the NPU environment, please refer to the `docs/onnx_npu_docs.md` file.
