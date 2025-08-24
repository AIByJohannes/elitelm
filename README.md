# EliteLM

**EliteLM** is a high-performance inference server for large language models, optimized for PCs equipped with the Qualcomm Hexagon NPU (part of the Snapdragon X Elite chip). It allows you to run state-of-the-art language models locally on your machine with hardware acceleration.

## Features

*   **Hardware Accelerated Inference:** Leverages the Hexagon NPU for efficient and fast inference.
*   **ONNX Runtime:** Uses the ONNX Runtime to run models, ensuring compatibility and performance.
*   **Llama-3 Support:** Comes with a script to run the Llama-3 model out of the box.
*   **Interactive QA:** Includes an interactive command-line interface for asking questions to the model.
*   **Customizable Generation:** Allows you to control the generation process with parameters like temperature, top-k, and top-p.

## Roadmap

- [x] **Inference Script (`llama3-qa.py`)**
    - [x] Load model and tokenizer.
    - [x] Implement interactive prompt loop.
    - [x] Run inference on CPU.
    - [ ] **Run inference on NPU.**
    - [ ] **Add support for more models.**
    - [ ] **Improve generation performance.**
- [ ] **Inference Server (`api.py`)**
    - [ ] **Create a FastAPI application.**
    - [ ] **Implement a `/generate` endpoint that takes a prompt and returns a response.**
    - [ ] **Integrate the `QA` class from `llama3-qa.py` into the server.**
    - [ ] **Add request and response models.**
    - [ ] **Implement error handling.**
    - [ ] **Add logging.**
- [ ] **Advanced Features**
    - [ ] **Add a streaming endpoint for real-time generation.**
    - [ ] **Implement a web-based UI for interacting with the model.**
    - [ ] **Package the application as a Docker container.**
    - [ ] **Add support for more backends (e.g., GPU).**

## Requirements

- Python 3.11
- Access to a machine with a Qualcomm Hexagon NPU (for NPU acceleration).

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download the model:**

    ```bash
    huggingface-cli download onnx-community/Llama-3.2-3B-Instruct-ONNX --include cpu_and_mobile/* --local-dir .
    ```

## Usage

To start the interactive question-answering script, run the following command:

```bash
python llama3-qa.py -m ./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -b QnnHtp.dll
```

Once the script is running, you can type your questions and press Enter to get a response from the model.

### Configuration

You can customize the generation process using the following command-line arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `-m`, `--model` | Path to the model directory. | (required) |
| `-b`, `--backend` | Path to the QNN backend library (`.dll`). | (required) |
| `-k`, `--top_k` | Top-k sampling parameter. | 40 |
| `-p`, `--top_p` | Top-p (nucleus) sampling parameter. | 0.95 |
| `-t`, `--temperature` | Temperature for sampling. | 0.8 |
| `-r`, `--repetition_penalty` | Repetition penalty. | 1.0 |

## Project Structure

```
.
├── docs
│   └── onnx_npu_docs.md
├── api.py
├── llama3-qa.py
├── README.md
└── requirements.txt
```

*   **`docs/`**: Contains documentation files.
*   **`api.py`**: The entry point for the upcoming FastAPI server.
*   **`llama3-qa.py`**: A command-line script for interactive question answering.
*   **`README.md`**: This file.
*   **`requirements.txt`**: A list of Python dependencies.