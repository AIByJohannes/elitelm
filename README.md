# Elite LLM
Inference server for PCs with Hexagon NPU (part of the X Elite chip)

## Roadmap
- [x] Inference script
    - [x] Load model
    - [x] Load input
    - [x] Run inference on CPU
    - [ ] Run inference on NPU
    - [ ] Customize script
- [ ] Inference server
    - [ ] Create FastAPI server
    - [ ] Create endpoint for inference
    - [ ] Connect inference script to server

## Requirements
- Python 3.11

## Setup
```bash
pip install -r requirements.txt
```

## Run inference script

```bash
huggingface-cli download onnx-community/Llama-3.2-3B-Instruct-ONNX --include cpu_and_mobile/* --local-dir .
```

```bash
python llama3-qa.py -m ./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -k 40 -p 0.95 -t 0.8 -r 1.0 -b QnnHtp.dll
```
