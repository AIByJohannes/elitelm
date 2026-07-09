use elitelm_backend_llamacpp::LlamaCppBackend;
use elitelm_core::{ChatMessage, GenerateRequest, InferenceBackend, LlamaCppBackendConfig};
use std::path::PathBuf;

/// Returns the GGUF model path to use for integration tests.
/// Priority order:
///   1. GGUF_TEST_MODEL env var (set explicitly by the user/CI)
///   2. Well-known local path from LM Studio cache
fn test_model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GGUF_TEST_MODEL") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    // Fallback: Gemma model in LM Studio cache
    let fallback = PathBuf::from(r"C:\Users\johan\.cache\lm-studio\models\lmstudio-community\gemma-2-2b-it-GGUF\gemma-2-2b-it-Q4_K_M.gguf");
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

/// Confirms that the backend validates a non-existent model path without panicking.
#[test]
fn rejects_missing_model() {
    let config = LlamaCppBackendConfig {
        model: PathBuf::from("/nonexistent/path/model.gguf"),
        n_threads: None,
        n_ctx: None,
        n_batch: None,
        use_mmap: true,
    };
    let result = LlamaCppBackend::new("test", config);
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("does not exist"), "unexpected error: {msg}");
}

/// Runs real inference against a GGUF model if one is available.
/// Skipped automatically when no model file can be found.
#[test]
fn real_inference_produces_output() {
    let model_path = match test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP real_inference_produces_output: no GGUF model found");
            eprintln!("  Set GGUF_TEST_MODEL=/path/to/model.gguf to enable this test.");
            return;
        }
    };

    let config = LlamaCppBackendConfig {
        model: model_path,
        n_threads: Some(4),
        n_ctx: Some(512),
        n_batch: Some(512),
        use_mmap: true,
    };
    let mut backend = LlamaCppBackend::new("llamacpp", config).expect("backend creation failed");

    let mut output = String::new();
    let stats = backend
        .generate(
            GenerateRequest {
                messages: vec![ChatMessage::new("user", "Say hello in one word.")],
                max_tokens: Some(20),
                temperature: None,
                top_p: None,
            },
            &mut |piece| {
                output.push_str(piece);
                Ok(())
            },
        )
        .expect("generate failed");

    assert!(!output.trim().is_empty(), "expected non-empty output");
    assert!(stats.completion_tokens > 0);
    assert!(stats.prompt_tokens > 0);
    eprintln!("Output: {output:?}");
    eprintln!("Stats: prompt={} completion={}", stats.prompt_tokens, stats.completion_tokens);
}
