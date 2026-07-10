# EliteLM Rust Migration Roadmap

Source of truth: TickTick task `[EliteLM] migrate to Rust` (`6a176389a09e53d53fe64e9a`).

This roadmap supersedes the earlier Python/ONNX-centered roadmap. The new target is a Rust product shell with two inference backends:

- Genie backend for Snapdragon NPU / low-energy mode.
- llama.cpp backend for CPU / high-speed mode.

ONNX Runtime GenAI is not part of the core target architecture. Existing Python, FastAPI, and PowerShell code should be treated as reference behavior to port or replace, not as the long-term runtime.

## Target Architecture

```text
elitelm/
|-- crates/
|   |-- elitelm-core/
|   |   `-- Backend trait, chat templates, config, OpenAI request/response types
|   |-- elitelm-backend-genie/
|   |   `-- Safe Rust wrapper for Qualcomm Genie execution
|   |-- elitelm-backend-genie-sys/
|   |   `-- C ABI shim over Qualcomm Genie C++/C SDK
|   |-- elitelm-backend-llamacpp/
|   |   `-- Safe Rust wrapper over llama.cpp
|   |-- elitelm-backend-llamacpp-sys/
|   |   `-- bindgen bindings over llama.h
|   |-- elitelm-server/
|   |   `-- axum OpenAI-compatible API
|   `-- elitelm-cli/
|       `-- clap CLI
|-- models/
|-- qairt/
`-- docs/
```

## Core Decisions

- Keep Genie as the Snapdragon NPU backend.
- Keep llama.cpp as the CPU backend.
- Do not make Genie and llama.cpp share a model format.
- Start Genie integration with a process backend around `genie-t2t-run.exe`.
- Replace the process backend later with a small C ABI bridge over the Genie SDK.
- Bind llama.cpp directly through `llama.h` with bindgen.
- Replace FastAPI with axum.
- Replace the Python CLI with clap.
- Keep C/C++ limited to the inference runtime boundary.

## Backend Contract

The first Rust implementation should establish a small backend trait before optimizing individual runtimes.

```rust
pub trait InferenceBackend: Send {
    fn name(&self) -> &'static str;

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> anyhow::Result<()>,
    ) -> anyhow::Result<GenerateStats>;
}

pub enum Backend {
    Genie(GenieBackend),
    LlamaCpp(LlamaCppBackend),
}
```

The trait should support streaming token callbacks from the beginning so the CLI and OpenAI-compatible API do not need a second interface later.

## Configuration Model

The config must explicitly separate Genie artifacts from llama.cpp artifacts.

```yaml
default_backend: genie_npu

backends:
  genie_npu:
    kind: genie
    bundle_dir: ./models/llama-3.2-3b-genie
    genie_config: ./models/llama-3.2-3b-genie/genie_config.json
    htp_config: ./models/llama-3.2-3b-genie/htp_backend_ext_config.json
    qnn_sdk_root: ./qairt/2.37.0
    soc_model: 60
    dsp_arch: v73

  llamacpp_cpu:
    kind: llamacpp
    model: ./models/llama-3.2-3b.Q4_K_M.gguf
    n_threads: 12
    n_ctx: 4096
    n_batch: 1024
    use_mmap: true
```

Genie configuration should model bundles of QAIRT context binaries, tokenizer files, `genie_config.json`, HTP backend config, and copied runtime libraries. llama.cpp configuration should model GGUF files and llama.cpp runtime parameters.

## Phase 0: Preserve Current Behavior

- [x] Document current Python CLI behavior, FastAPI endpoints, config fields, and expected streaming response shape.
- [x] Capture the current Genie bundle layout under `cpu_and_mobile/llama-3.2-3b-npu-complete`.
- [x] Capture the current PowerShell setup flow from `RunLlm.ps1`, `LlmUtils.ps1`, and related scripts.
- [x] Define smoke prompts and expected success criteria for CPU and NPU paths.
- [x] Decide which Python tests are behavior references for the Rust rewrite.

Exit criteria:

- Current behavior is documented enough that Rust parity can be verified without re-reading the old implementation.

## Phase 1: Rust Workspace and Product Shell

- [x] Add a Cargo workspace with `elitelm-core`, `elitelm-cli`, and `elitelm-server`.
- [x] Implement config loading and validation in `elitelm-core`.
- [x] Implement OpenAI-compatible request/response types in `elitelm-core`.
- [x] Implement chat-template and message formatting behavior needed by the current models.
- [x] Add a placeholder backend implementation for CLI/server wiring tests.
- [x] Add `elitelm run --backend <name>`.
- [x] Add `elitelm serve --backend <name>`.
- [x] Add parity tests for config validation and OpenAI response serialization.

Exit criteria:

- The Rust CLI and server compile and can exercise a fake backend through the same public commands intended for real inference.

## Phase 2: Process-Based Genie Backend

- [x] Add `elitelm-backend-genie`.
- [x] Port PowerShell Genie bundle preparation into Rust as `elitelm prepare-genie-bundle`.
- [x] Validate required Genie files before execution.
- [x] Copy or locate Windows `hexagon-v73` files, `aarch64-windows-msvc` libraries, and `genie-t2t-run.exe` according to the existing bundle flow.
- [x] Implement `elitelm run --backend genie` using `std::process::Command` around `genie-t2t-run.exe`.
- [x] Implement server generation through the process backend.
- [x] Surface clear errors for missing SDK roots, missing configs, missing binaries, and nonzero Genie exits.
- [x] Add a Snapdragon Windows smoke test for a known Genie bundle.

Exit criteria:

- Rust can run the real Genie NPU path without Python or PowerShell.

## Phase 3: llama.cpp CPU Backend

- [x] Add `elitelm-backend-llamacpp-sys`.
- [x] Generate bindgen bindings over `llama.cpp/include/llama.h`.
- [x] Add build/link instructions for the checked-out llama.cpp submodule.
- [x] Add `elitelm-backend-llamacpp` as a safe wrapper over model load, context creation, tokenization, decode, sampling, and teardown.
- [x] Implement `elitelm run --backend llamacpp`.
- [x] Implement `elitelm serve --backend llamacpp`.
- [x] Add CPU smoke tests using a small GGUF model fixture or documented local test model.
- [x] Add benchmarking for tokens/sec, time to first token, and memory footprint.

Exit criteria:

- Rust can run a GGUF model through native llama.cpp FFI and expose it through the same CLI/server surface as Genie.

## Phase 4: OpenAI-Compatible axum Server

- [x] Replace the FastAPI runtime with `elitelm-server`.
- [x] Implement `/v1/chat/completions`.
- [x] Support streaming and non-streaming responses.
- [x] Preserve the current request compatibility surface where practical.
- [x] Add structured logging and backend selection diagnostics.
- [x] Add integration tests for request validation, response schema, streaming chunks, and backend errors.

Exit criteria:

- Existing API clients can point at the Rust server for chat completions with either backend.

## Phase 5: Native Genie Bridge

- [x] Add `elitelm-backend-genie-sys`.
- [x] Create a stable C ABI bridge:

```text
crates/elitelm-backend-genie-sys/
|-- build.rs
|-- bridge/
|   |-- elite_genie_bridge.h
|   `-- elite_genie_bridge.cpp
`-- src/lib.rs
```

- [x] Compile the bridge with the `cc` crate using `.cpp(true)`.
- [x] Read `QNN_SDK_ROOT` or explicit config paths during build/runtime validation.
- [x] Keep the Rust boundary C-compatible instead of binding Qualcomm C++ types directly.
- [x] Implement safe Rust ownership around native handles.
- [x] Add parity tests comparing process-based Genie output behavior with native bridge behavior.
- [x] Retire the process backend only after the native bridge proves stable.

Exit criteria:

- Genie execution no longer shells out to `genie-t2t-run.exe`, while keeping the same `InferenceBackend` contract.

## Phase 6: Migration Cutover

- [x] Update README and setup docs to make Rust the primary installation and execution path.
- [x] Move Python implementation into an archived/reference location or remove it after parity is proven.
- [x] Remove ONNX Runtime GenAI from primary dependencies.
- [x] Remove PowerShell runtime dependency from normal operation.
- [x] Keep scripts only where they are still useful for developer setup or diagnostics.
- [x] Add release packaging for Windows ARM64.
- [x] Add CI jobs for formatting, clippy, unit tests, and server API tests.

Exit criteria:

- The repository presents EliteLM as a Rust project with Genie NPU and llama.cpp CPU backends, and no longer depends on Python/PowerShell for normal inference.

## Verification Matrix

- Config validation: missing files, wrong backend kind, unsupported model format, invalid runtime paths.
- CLI: `prepare-genie-bundle`, `run --backend genie`, `run --backend llamacpp`.
- Server: non-streaming chat completions, streaming chat completions, backend selection, error responses.
- Genie NPU: bundle validation, real prompt smoke test, failed process/native call diagnostics.
- llama.cpp CPU: model load, prompt generation, context limits, sampler configuration, teardown.
- Performance: tokens/sec, time to first token, memory footprint, CPU-vs-NPU comparison.

## Immediate Next Work

1. Create the Rust workspace and fake backend.
2. Port config parsing and OpenAI request/response types into `elitelm-core`.
3. Implement `elitelm run` and `elitelm serve` against the fake backend.
4. Add process-based Genie execution around `genie-t2t-run.exe`.
5. Add llama.cpp bindings and CPU execution.

