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
    - [x] **Run inference on NPU.**
    - [ ] **Add support for more LLMs.**
- [ ] **Inference Server (`api.py`)**
    - [ ] **Create a FastAPI application.**
    - [ ] **Implement an OpenAI Chat Completions compatible endpoint that takes a prompt and returns a response.**
    - [ ] **Integrate functionality from `llama3-qa.py` into the server.**
    - [ ] **Implement error handling.**
    - [ ] **Add logging.**
- [ ] **Advanced Features**
    - [ ] **Add a streaming endpoint for real-time generation.**
    - [ ] **Implement a Streamlit app demo for interacting with the LLMs .**

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

3.  **Create your runtime config:**

    ```bash
    cp llama3-qa.example.yaml llama3-qa.yaml
    ```

    Edit `llama3-qa.yaml` to point to your downloaded model directory and tweak the `generation` values to your preference.

## Run inference script

The CLI now reads all options from the YAML file. With the default file name you can simply run:

```bash
python llama3-qa.py
```

Alternatively, pass an explicit path if you keep multiple configs around:

```bash
python llama3-qa.py --config configs/my-experiment.yaml
```

### Running on the Hexagon NPU

Set `device: qnn` in your config and populate the `qnn` block:

```yaml
qnn:
  sdk_root: ./qairt/2.37.0.250724
  backend: null  # optional override when auto-detection does not work
```

The script will automatically update the DLL search path and configure the QNN execution provider. Enable `runtime.verbose: true` to confirm graph compilation in the logs.

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
