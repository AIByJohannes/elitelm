use std::path::PathBuf;

fn main() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2) // crates/elitelm-backend-llamacpp-sys → crates → workspace root
        .expect("could not find workspace root")
        .to_path_buf();

    let llama_src = workspace_root.join("llama.cpp");
    let header = llama_src.join("include").join("llama.h");

    // ── Re-run triggers ───────────────────────────────────────────────────────
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed=build.rs");

    // ── 1. Locate build tools ─────────────────────────────────────────────────
    let cmake_exe = find_cmake().expect("cmake not found. Install CMake and add it to PATH.");
    let ninja = find_ninja().expect("ninja not found. Install Ninja and add it to PATH.");

    // Add ninja's parent directory to PATH so CMake can find it automatically
    if let Some(ninja_dir) = ninja.parent() {
        let mut paths = vec![ninja_dir.to_path_buf()];
        paths.extend(
            std::env::var_os("PATH")
                .map_or_else(Vec::new, |path| std::env::split_paths(&path).collect()),
        );
        unsafe {
            std::env::set_var("PATH", std::env::join_paths(paths).expect("valid PATH"));
        }
    }

    // ── 2. Build llama.cpp via CMake (Ninja + clang-cl) ──────────────────────
    let mut cfg = cmake::Config::new(&llama_src);
    cfg.define("CMAKE_BUILD_TYPE", "Release")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        // Disable GPU backends — CPU only
        .define("GGML_CUDA", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_VULKAN", "OFF")
        // Disable OpenMP to prevent unresolved external linker issues (omp_*)
        .define("GGML_OPENMP", "OFF")
        .define("LLAMA_OPENMP", "OFF")
        .define("CMAKE_MAKE_PROGRAM", ninja.display().to_string())
        // Use the Ninja generator
        .generator("Ninja")
        // Tell the cmake crate where cmake lives
        .env("CMAKE", cmake_exe.display().to_string());

    #[cfg(windows)]
    {
        let clang_cl = find_clang_cl().expect(
            "clang-cl not found. Install LLVM: winget install LLVM.LLVM. \
             llama.cpp requires Clang for ARM64/aarch64-windows targets.",
        );
        cfg.define("CMAKE_C_COMPILER", clang_cl.display().to_string())
            .define("CMAKE_CXX_COMPILER", clang_cl.display().to_string())
            // Enable C++ exceptions (required by gguf.cpp).
            .cxxflag("/EHsc");
    }

    let dst = cfg.build();

    // ── 3. Link instructions ──────────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Windows system libs required by llama.cpp / ggml.
    #[cfg(windows)]
    for library in ["kernel32", "advapi32", "shell32", "ole32", "user32"] {
        println!("cargo:rustc-link-lib={library}");
    }

    // ── 4. Generate bindgen bindings ──────────────────────────────────────────
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            llama_src.join("ggml").join("include").display()
        ))
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .prepend_enum_name(false)
        .derive_debug(true)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings for llama.h");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}

// ── Tool-finder helpers ───────────────────────────────────────────────────────

#[cfg(windows)]
fn vs_install_path() -> Option<PathBuf> {
    let vswhere =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let output = std::process::Command::new(&vswhere)
        .args(["-latest", "-property", "installationPath"])
        .output()
        .ok()?;
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn find_cmake() -> Option<PathBuf> {
    #[cfg(windows)]
    if let Some(vs) = vs_install_path() {
        let bundled = vs
            .join("Common7")
            .join("IDE")
            .join("CommonExtensions")
            .join("Microsoft")
            .join("CMake")
            .join("CMake")
            .join("bin")
            .join("cmake.exe");
        if bundled.exists() {
            return Some(bundled);
        }
    }
    which(if cfg!(windows) { "cmake.exe" } else { "cmake" })
}

#[cfg(windows)]
fn find_clang_cl() -> Option<PathBuf> {
    let standalone = PathBuf::from(r"C:\Program Files\LLVM\bin\clang-cl.exe");
    if standalone.exists() {
        return Some(standalone);
    }
    if let Some(vs) = vs_install_path() {
        for sub in [
            "VC\\Tools\\Llvm\\ARM64\\bin",
            "VC\\Tools\\Llvm\\bin",
            "VC\\Tools\\Llvm\\x64\\bin",
        ] {
            let p = vs.join(sub).join("clang-cl.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    which("clang-cl.exe")
}

fn find_ninja() -> Option<PathBuf> {
    #[cfg(windows)]
    if let Some(vs) = vs_install_path() {
        let bundled = vs
            .join("Common7")
            .join("IDE")
            .join("CommonExtensions")
            .join("Microsoft")
            .join("CMake")
            .join("Ninja")
            .join("ninja.exe");
        if bundled.exists() {
            return Some(bundled);
        }
    }
    which(if cfg!(windows) { "ninja.exe" } else { "ninja" })
}

fn which(exe: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = PathBuf::from(dir).join(exe);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
