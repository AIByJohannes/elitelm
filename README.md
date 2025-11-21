# EliteLM

**EliteLM** is a high-performance inference server for large language models, optimized for PCs equipped with the Qualcomm Hexagon NPU (part of the Snapdragon X Elite chip). It allows you to run state-of-the-art language models locally on your machine with hardware acceleration.

## Features

*   **Hardware Accelerated Inference:** Leverages the Hexagon NPU for efficient and fast inference on Snapdragon X Elite/Plus PCs.
*   **Flexible Runtime:** A reusable, core runtime module (`elitelm.py`) handles configuration, model loading, and token generation.
*   **ONNX Runtime:** Uses the ONNX Runtime to run models, ensuring compatibility and performance.
*   **Interactive CLI:** Includes an interactive command-line interface (`chat_cli.py`) for easy testing and experimentation.
*   **API Server:** A FastAPI server (`api.py`) to expose the model's functionality as a web API.
*   **Llama-3 Support:** Comes with an interactive chat client to run the Llama-3 model out of the box.
*   **Customizable Generation:** Allows you to control the generation process with parameters like temperature, top-k, and top-p.
*   **Comprehensive Testing:** A three-tier testing strategy ensures robust NPU verification.

## Project Architecture

EliteLM is divided into three main parts:

1.  **A reusable runtime module** (`elitelm.py`) that handles configuration parsing, model loading, and token generation.
2.  **A command-line interface (CLI)** (`chat_cli.py`) for interactive testing and experimentation.
3.  **A FastAPI server** (`api.py`) for exposing the model's functionality as a web API.

### Core Components

*   **`elitelm.py`**: This module houses the core runtime logic. The `ChatSession` class manages chat history and streaming generation. It also contains helpers for configuring the QNN execution provider for NPU acceleration.

*   **`chat_cli.py`**: A thin wrapper around `ChatSession` that provides an interactive command-line interface for quick, interactive sessions with the model.

*   **`api.py`**: A skeleton for a FastAPI server that will expose the `ChatSession` functionality as a web API.

*   **ONNX Runtime and QNN Backend**: The project uses the [ONNX Runtime](https://onnxruntime.ai/) to run the ONNX model. To leverage the Hexagon NPU, it's configured to use the **QNN Execution Provider**, which bridges the ONNX Runtime and the Qualcomm AI Engine.

## Roadmap

- [x] **Core Runtime (`elitelm.py`)**
    - [x] Load model and tokenizer.
    - [x] Run inference on CPU.
    - [x] **Run inference on NPU.**
    - [ ] **Add support for more LLMs.**
- [x] **Interactive CLI (`chat_cli.py`)**
    - [x] Implement interactive prompt loop.
- [x] **Inference Server (`api.py`)**
    - [x] **Create a FastAPI application.**
    - [x] **Implement an OpenAI Chat Completions compatible endpoint that takes a prompt and returns a response.**
    - [x] **Integrate functionality from `elitelm.py` into the server.**
    - [ ] **Implement error handling.**
    - [ ] **Add logging.**
- [x] **Advanced Features**
    - [x] **Add a streaming endpoint for real-time generation.**
    - [ ] **Implement a Streamlit app demo for interacting with the LLMs .**

## Requirements

- Python 3.11
- Access to a machine with a Qualcomm Hexagon NPU (for NPU acceleration).

## Setup

1.  **Create and activate the virtual environment (`.venv/`):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download and extract the Qualcomm AI Engine Direct SDK:**
    Download the SDK from [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) and extract the contents to the `qairt/` directory.
4.  **Download the Genie-compatible model:**
    > **Important**: You must use `huggingface-cli` to download the pre-compiled Genie model bundle. Ensure it's installed and authenticated.
    
    ```bash
    huggingface-cli download Volko76/Llama-3.2-3B-Genie-Compatible-QNN-Binaries --local-dir ./cpu_and_mobile/llama-3.2-3b-npu-complete/genie_bundle
    ```
    This downloads the complete Genie bundle including the `genie-t2t-run.exe` executable, QNN context binaries (`*.bin`), DLLs, and tokenizer.
    
    If `huggingface-cli` is not found, install it:
    ```bash
    pip install huggingface-hub
    ```
    
    Then authenticate with your Hugging Face account (optional but recommended for faster downloads):
    ```bash
    huggingface-cli login
    ```
5.  **Create your runtime config:**
    ```bash
    cp llama3-qa.example.yaml llama3-qa.yaml
    ```
    Edit `llama3-qa.yaml` to point to your downloaded model directory and tweak the `generation` values to your preference.

## Run the chat client

The CLI now reads all options from the YAML file. With the default file name you can simply run:

```bash
python chat_cli.py
```

Alternatively, pass an explicit path if you keep multiple configs around:

```bash
python chat_cli.py --config configs/my-experiment.yaml
```

To embed EliteLM into another application, import `ChatSession` from `elitelm` and drive generation directly.

### Running the API Server

EliteLM includes an OpenAI-compatible API server built with FastAPI. To start the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The server loads the configuration from `llama3-qa.yaml` by default (or set `ELITELM_CONFIG` env var). It supports:
-   `/v1/chat/completions` endpoint.
-   Streaming responses (`stream=True`).
-   Hardware acceleration (if configured in YAML).

Example request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Running on the Hexagon NPU

To run the model on the Hexagon NPU, install the matching Qualcomm AI Engine SDK and update your YAML config with `device: qnn`. Populate the `qnn` block so the runtime can locate the required DLLs and configure the QNN execution provider automatically.

Set `device: qnn` in your config and populate the `qnn` block:

```yaml
qnn:
  sdk_root: ./qairt/2.37.0.250724
  backend: null  # optional override when auto-detection does not work
```

The script will automatically update the DLL search path and configure the QNN execution provider. Enable `runtime.verbose: true` to confirm graph compilation in the logs. For more details, see `docs/NPU_GUIDE.md`.

## Benchmarking

To compare CPU vs NPU performance using the same runtime:

```bash
python compare_cpu_npu.py --config llama3-qa.yaml "Your prompt here"
```

This script runs the prompt on both devices (overriding the `device` setting in the config) and reports tokens per second (TPS) and time-to-first-token (TTF).

## Testing

The project includes a comprehensive three-tier testing strategy to validate NPU execution, ensuring that the NPU is correctly utilized. The tests cover:
1.  **Build-time verification**: Checks if QNN is available in the environment.
2.  **Configuration validation**: Ensures the QNN provider is correctly configured.
3.  **Hardware execution**: Verifies that the model runs on the NPU, with performance checks to detect silent CPU fallbacks.

Run tests using `pytest`. For more details on the testing strategy, see `docs/NPU_GUIDE.md`.

## Project Structure

```
.
├── elitelm.py          # Core runtime for model loading and generation
├── chat_cli.py         # Interactive command-line client
├── api.py              # FastAPI server (upcoming)
├── docs/
│   └── NPU_GUIDE.md    # NPU setup and testing guide
├── tests/              # Pytest test suite
├── README.md           # This file
└── requirements.txt    # Python dependencies
```