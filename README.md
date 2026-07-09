# EliteLM

**EliteLM** is a high-performance native inference server for Large Language Models (LLMs), optimized for Snapdragon X Elite / Plus ARM64 PCs. 

It is written entirely in Rust and supports two hardware execution backends:
1. **CPU Mode** (default): High-speed inference using a native FFI wrapper over `llama.cpp`.
2. **NPU Mode** (experimental): Low-energy inference using a native in-process C API bridge over the Qualcomm Genie SDK.

---

## Workspace Architecture

EliteLM is organized as a Cargo workspace:

*   **`crates/elitelm-core`**: Core configuration model parsing, OpenAI compatible schema definitions, and path resolution helpers.
*   **`crates/elitelm-backend-llamacpp` & `elitelm-backend-llamacpp-sys`**: Build system orchestration (CMake/Ninja/clang-cl) and safe RAII wrappers around `llama.h` to drive CPU inference.
*   **`crates/elitelm-backend-genie` & `elitelm-backend-genie-sys`**: Dynamic library wrapper (`libloading`) that loads `Genie.dll` and drives Hexagon NPU inference using direct C API bindings.
*   **`crates/elitelm-cli`**: Command-line interface (`elitelm-cli`) for running prompts and preparing NPU bundles.
*   **`crates/elitelm-server`**: Axum-based HTTP server exposing streaming and non-streaming OpenAI-compatible completion endpoints.

---

## Prerequisites

To build and run EliteLM, ensure you have:

1. **Rust Toolchain**: [rustup](https://rustup.rs/) (edition 2024 / stable).
2. **LLVM Compiler Suite**: Required to compile `llama.cpp` for Windows ARM64 targets (MSVC compiler is not supported by GGML for ARM targets). Run in Administrator terminal:
   ```powershell
   winget install LLVM.LLVM --accept-source-agreements --accept-package-agreements
   ```
3. **Visual Studio 2022 C++ Development Tools**: Specifically, ensure C++ CMake Tools and Ninja build system components are installed.
4. **(NPU Mode Only) Qualcomm AI Engine Direct SDK (QNN)**:
   - Download the SDK from the Qualcomm Developer Network.
   - Extract it to `qairt/` in the project root.

---

## Quick Start

### 1. Initialize Submodules
Clones the bundled `llama.cpp` submodule dependency:
```bash
git submodule update --init --recursive
```

### 2. Build the Workspace
Compiles all crates, including generating FFI bindings and building `llama.cpp` using Ninja and Clang:
```bash
cargo build --workspace
```

### 3. Run Inference via CLI
Configure your model paths in `elitelm.example.yaml` and rename/copy it to `elitelm.yaml`.

Run a prompt against your default configured backend:
```bash
cargo run --bin elitelm-cli -- run --config elitelm.yaml --prompt "Why is the sky blue? Answer in 1 sentence."
```

### 4. Run the API Server
Start the Axum OpenAI-compatible server:
```bash
cargo run --bin elitelm-server -- serve --config elitelm.yaml
```

The server listens on `http://127.0.0.1:8000` by default. You can test it using any OpenAI-compatible client or `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llamacpp_cpu",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Running on the NPU (Snapdragon Hexagon)

To target the NPU:
1. Download a Genie-compatible model bundle containing QNN context binaries (`*.bin`).
2. Run the preparation tool to package the required QNN runtime DLLs and generate configurations:
   ```bash
   cargo run --bin elitelm-cli -- prepare-genie-bundle --config elitelm.yaml --backend genie_npu
   ```
3. Configure `device: genie` inside your `elitelm.yaml` configuration to activate native dynamically-loaded NPU inference.

For details, see [NPU Guide](file:///c:/Users/johan/Code/AIByJohannes/elitelm/docs/NPU_GUIDE.md).
