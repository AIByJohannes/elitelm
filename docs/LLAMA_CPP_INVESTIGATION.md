# Llama.cpp Compilation & Setup Investigation (Snapdragon X Elite)

This document records the attempts to compile or acquire an optimized `llama.cpp` for the Snapdragon X Elite (ARM64 Windows).

## Repository Setup

- `llama.cpp` is tracked as a git submodule at `llama.cpp/` to keep upstream history and enable clean updates.

## Compilation Attempts

### 1. MSVC ARM64 Native Build
Attempted to use CMake with the Visual Studio 17 2022 generator targeting ARM64.
- **Command:** `cmake .. -G "Visual Studio 17 2022" -A ARM64 -DGGML_OPENMP=OFF`
- **Result:** **Failed**. 
- **Error:** `MSVC is not supported for ARM, use clang`. 
- **Context:** `llama.cpp`'s GGML CPU backend requires Clang for ARM Windows to utilize specific optimizations (NEON/DotProd).

### 2. Clang/LLVM Investigation
Searched for existing Clang installations within Visual Studio.
- **Result:** `clang.exe` was not found in standard VS 2022 paths.
- **Requirement:** Building on Snapdragon X Elite optimally requires the `C++ Clang Compiler for Windows` and `MS-Build Support for LLVM-Toolset (clang)` components to be installed via the Visual Studio Installer.

## Binary Acquisition Attempts

### 1. GitHub Release Assets (ggml-org/llama.cpp)
Attempted to download the pre-compiled ARM64 Windows binary from the official releases.
- **Version:** `b7801`
- **Asset Pattern:** `llama-b7801-bin-win-cpu-arm64.zip`
- **Result:** **Unsuccessful via automated tools**. `Invoke-WebRequest` returned 404 (likely due to redirect/auth handling) and `gh release download` failed to match patterns despite the asset appearing in web scans.

## Summary of Findings
- **Optimized Path:** To use `llama.cpp` on Snapdragon X Elite, it MUST be compiled with **Clang** to enable ARM-specific instructions. MSVC is currently unsupported for the GGML ARM backend.
- **Next Step:** Ensure LLVM/Clang is installed via VS Installer (or a manual LLVM install), then build from the submodule using the clang toolset or the `arm64-windows-llvm-release` CMake preset (if available in the submodule).
