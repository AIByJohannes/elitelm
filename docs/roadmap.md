# Roadmap

## NPU Mode (Genie Runtime)

- [ ] **Keep Genie as the NPU runtime**
    - [x] Document the supported Genie bundle layout and required DLLs.
    - [x] Add explicit runtime checks and helpful errors when Genie artifacts are missing.
- [ ] **Genie bindings without PowerShell**
    - [ ] Use existing Python bindings (e.g., `qai_appbuilder`) instead of custom C++/Rust.
    - [ ] Replace PowerShell-based flows with direct Python calls into the Genie runtime.
    - [ ] Add a smoke-test that exercises the binding on a known Genie bundle.
    - [ ] Only consider a custom C++/Rust binding layer if the existing bindings lack required features or performance.
    - [ ] Dependencies: document required Python packages and minimum versions once selected.

## CPU Mode (llama.cpp Custom Build)

- [ ] **Custom llama.cpp build for Snapdragon X Elite (no source changes)**
    - [ ] Define the build flags and toolchain for Windows ARM64.
    - [ ] Add a reproducible build script and binary packaging for the custom build.
    - [ ] Integrate a fast CPU execution path in `session.py` that selects the custom llama.cpp backend.
    - [ ] Add performance benchmarks and a CPU-vs-NPU comparison report.

## Runtime Focus

- [x] **Core Runtime (`session.py`)**
    - [x] Load model and tokenizer.
    - [x] Run inference on CPU.
    - [x] Run inference on NPU.

## CLI + API Server

- [x] **Interactive CLI (`cli.py`)**
    - [x] Implement interactive prompt loop.
- [x] **Inference Server (`api.py`)**
    - [x] Create a FastAPI application.
    - [x] Implement an OpenAI Chat Completions compatible endpoint that takes a prompt and returns a response.
    - [x] Integrate functionality from `session.py` into the server.
    - [ ] Implement error handling.
    - [ ] Add logging.

## Advanced Features

- [x] Add a streaming endpoint for real-time generation.
- [ ] Implement a Streamlit app demo for interacting with the LLMs.
- [ ] Add support for more LLMs.