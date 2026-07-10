# EliteLM Test Results

- Date: 2026-07-10 (Europe/Berlin)
- Commit tested: `fa44800`
- Platform: Windows 11 aarch64 (Snapdragon X Elite), PowerShell
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Discovered VS CMake: `4.2.3-msvc3`
- Discovered VS Ninja: `1.13.2`
- Discovered LLVM clang-cl: `22.1.8`

## Summary

EliteLM built and ran successfully. The repository's CI-equivalent format,
lint, and workspace test commands all passed.

| Check | Result | Details |
| --- | --- | --- |
| CLI smoke run | PASS | Returned `EliteLM fake response: smoke test` |
| Formatting | PASS | `cargo fmt --check` |
| Clippy | PASS | `cargo clippy --workspace -- -D warnings` |
| Workspace tests | PASS | 21 passed, 0 failed, 1 ignored |
| Real CPU inference | PASS | Local Gemma 2 2B Q4_K_M GGUF produced `Hello. \n`; 21 prompt tokens and 4 completion tokens |

## Commands Run

```powershell
cargo run --bin elitelm-cli -- run --config elitelm.example.yaml --backend fake --prompt "smoke test"
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace -- --nocapture
```

## Test Breakdown

| Test target | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| `elitelm-backend-genie` unit tests | 5 | 0 | 0 |
| Genie hardware integration test | 0 | 0 | 1 |
| `elitelm-backend-llamacpp` integration tests | 2 | 0 | 0 |
| `elitelm-cli` integration tests | 2 | 0 | 0 |
| `elitelm-core` unit tests | 9 | 0 | 0 |
| `elitelm-server` API integration tests | 3 | 0 | 0 |
| **Total** | **21** | **0** | **1** |

All crate unit-test targets and doc-test targets that contain zero tests also
completed successfully.

## Notes

- The ignored test is `runs_real_local_genie_bundle`. It explicitly requires
  Snapdragon Windows hardware, the QAIRT SDK, and a local Genie bundle.
- Real llama.cpp CPU inference was exercised because the test suite found a
  local Gemma 2 2B GGUF model.
- `cmake` and `ninja` were not available on the global PowerShell `PATH` during this run.
  However, the build script successfully discovered CMake 4.2.3 and Ninja 1.13.2 bundled
  with Visual Studio, as well as LLVM clang-cl 22.1.8, allowing the genuinely clean native rebuild
  to succeed in an isolated cargo target directory without modifying the system PATH.
