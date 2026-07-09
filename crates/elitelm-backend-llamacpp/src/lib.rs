use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr::NonNull;

use anyhow::{Context, Result as AnyResult, anyhow};
use elitelm_core::{GenerateRequest, GenerateStats, InferenceBackend, LlamaCppBackendConfig};
use elitelm_backend_llamacpp_sys as sys;
use thiserror::Error;

/// Errors specific to the llama.cpp backend.
#[derive(Debug, Error)]
pub enum LlamaCppError {
    #[error("model file does not exist: {path}")]
    MissingModel { path: PathBuf },
    #[error("llama_model_load_from_file returned null for {path}")]
    LoadFailed { path: PathBuf },
    #[error("llama_init_from_model returned null")]
    ContextFailed,
    #[error("tokenization failed: return code {0}")]
    TokenizeFailed(i32),
    #[error("llama_decode failed: return code {0}")]
    DecodeFailed(i32),
    #[error("messages cannot be empty")]
    EmptyMessages,
}

// ── RAII wrappers ─────────────────────────────────────────────────────────────

struct LlamaModel(NonNull<sys::llama_model>);

impl LlamaModel {
    fn load(path: &std::path::Path, params: sys::llama_model_params) -> AnyResult<Self> {
        if !path.exists() {
            return Err(LlamaCppError::MissingModel {
                path: path.to_path_buf(),
            }
            .into());
        }
        let c_path = CString::new(path.to_string_lossy().replace('\\', "/"))
            .context("model path contains a null byte")?;
        let raw = unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), params) };
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| {
                LlamaCppError::LoadFailed {
                    path: path.to_path_buf(),
                }
                .into()
            })
    }

    fn as_ptr(&self) -> *mut sys::llama_model {
        self.0.as_ptr()
    }

    fn vocab(&self) -> *const sys::llama_vocab {
        unsafe { sys::llama_model_get_vocab(self.as_ptr()) }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { sys::llama_model_free(self.0.as_ptr()) };
    }
}

struct LlamaContext(NonNull<sys::llama_context>);

impl LlamaContext {
    fn new(model: &LlamaModel, params: sys::llama_context_params) -> AnyResult<Self> {
        let raw = unsafe { sys::llama_init_from_model(model.as_ptr(), params) };
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| LlamaCppError::ContextFailed.into())
    }

    fn as_ptr(&self) -> *mut sys::llama_context {
        self.0.as_ptr()
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { sys::llama_free(self.0.as_ptr()) };
    }
}

struct LlamaSampler(NonNull<sys::llama_sampler>);

impl LlamaSampler {
    fn greedy() -> Self {
        let chain_params = unsafe { sys::llama_sampler_chain_default_params() };
        let chain = unsafe { sys::llama_sampler_chain_init(chain_params) };
        let chain = NonNull::new(chain).expect("llama_sampler_chain_init returned null");
        // Add greedy sampler to the chain
        let greedy = unsafe { sys::llama_sampler_init_greedy() };
        unsafe { sys::llama_sampler_chain_add(chain.as_ptr(), greedy) };
        Self(chain)
    }

    fn with_temperature(temperature: f32, top_p: f32) -> Self {
        let chain_params = unsafe { sys::llama_sampler_chain_default_params() };
        let chain = unsafe { sys::llama_sampler_chain_init(chain_params) };
        let chain = NonNull::new(chain).expect("llama_sampler_chain_init returned null");
        unsafe {
            sys::llama_sampler_chain_add(chain.as_ptr(), sys::llama_sampler_init_top_p(top_p, 1));
            sys::llama_sampler_chain_add(chain.as_ptr(), sys::llama_sampler_init_temp(temperature));
        }
        Self(chain)
    }

    fn sample(&self, ctx: &LlamaContext, idx: i32) -> sys::llama_token {
        unsafe { sys::llama_sampler_sample(self.0.as_ptr(), ctx.as_ptr(), idx) }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe { sys::llama_sampler_free(self.0.as_ptr()) };
    }
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct LlamaCppBackend {
    name: String,
    config: LlamaCppBackendConfig,
}

impl LlamaCppBackend {
    pub fn new(name: impl Into<String>, config: LlamaCppBackendConfig) -> AnyResult<Self> {
        if !config.model.exists() {
            return Err(LlamaCppError::MissingModel {
                path: config.model.clone(),
            }
            .into());
        }
        Ok(Self {
            name: name.into(),
            config,
        })
    }

    fn build_prompt(request: &GenerateRequest) -> AnyResult<String> {
        // Simple chat-to-text formatting that works for most instruction models.
        // For production, this should be driven by the tokenizer's chat template.
        let mut parts = Vec::new();
        for msg in &request.messages {
            match msg.role.as_str() {
                "system" => parts.push(format!("<|system|>\n{}<|end|>\n", msg.content)),
                "user" => parts.push(format!("<|user|>\n{}<|end|>\n", msg.content)),
                "assistant" => parts.push(format!("<|assistant|>\n{}<|end|>\n", msg.content)),
                _ => parts.push(format!("{}: {}\n", msg.role, msg.content)),
            }
        }
        parts.push("<|assistant|>\n".to_string());
        Ok(parts.join(""))
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> AnyResult<()>,
    ) -> AnyResult<GenerateStats> {
        if request.messages.is_empty() {
            return Err(LlamaCppError::EmptyMessages.into());
        }

        let prompt = Self::build_prompt(&request)?;

        // ── Initialise llama backend (idempotent) ─────────────────────────
        unsafe { sys::llama_backend_init() };

        // ── Load model ────────────────────────────────────────────────────
        let mut mparams = unsafe { sys::llama_model_default_params() };
        mparams.use_mmap = self.config.use_mmap;
        mparams.n_gpu_layers = 0; // CPU only

        let model = LlamaModel::load(&self.config.model, mparams)?;

        // ── Create context ────────────────────────────────────────────────
        let mut cparams = unsafe { sys::llama_context_default_params() };
        if let Some(n_ctx) = self.config.n_ctx {
            cparams.n_ctx = n_ctx;
        }
        if let Some(n_batch) = self.config.n_batch {
            cparams.n_batch = n_batch;
        }
        if let Some(n_threads) = self.config.n_threads {
            cparams.n_threads = n_threads as i32;
            cparams.n_threads_batch = n_threads as i32;
        }

        let ctx = LlamaContext::new(&model, cparams)?;

        // ── Tokenize prompt ───────────────────────────────────────────────
        let c_prompt = CString::new(prompt.as_str()).context("prompt contains a null byte")?;
        // First call to size the buffer
        let n_vocab = cparams.n_ctx.max(2048) as usize;
        let mut tokens: Vec<sys::llama_token> = vec![0; n_vocab];
        let n_tokens = unsafe {
            sys::llama_tokenize(
                model.vocab(),
                c_prompt.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,  // add_special (BOS)
                true,  // parse_special
            )
        };
        if n_tokens < 0 {
            return Err(LlamaCppError::TokenizeFailed(n_tokens).into());
        }
        tokens.truncate(n_tokens as usize);
        let prompt_token_count = tokens.len() as u32;

        // ── Choose sampler ────────────────────────────────────────────────
        let sampler = match (request.temperature, request.top_p) {
            (Some(t), top_p) if t > 0.0 => {
                LlamaSampler::with_temperature(t, top_p.unwrap_or(0.9))
            }
            _ => LlamaSampler::greedy(),
        };

        // ── Auto-calculate max tokens ─────────────────────────────────────
        let n_ctx = cparams.n_ctx as usize;
        let max_new = request
            .max_tokens
            .map(|m| m as usize)
            .unwrap_or_else(|| n_ctx.saturating_sub(tokens.len()).min(2048));

        // ── Evaluate prompt batch ─────────────────────────────────────────
        {
            let batch = unsafe {
                sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32)
            };
            let rc = unsafe { sys::llama_decode(ctx.as_ptr(), batch) };
            if rc != 0 {
                return Err(LlamaCppError::DecodeFailed(rc).into());
            }
        }

        // ── Generate tokens ───────────────────────────────────────────────
        let vocab = model.vocab();
        let mut completion_tokens = 0u32;
        let mut response_buf = [0i8; 256];

        for _ in 0..max_new {
            let token = sampler.sample(&ctx, -1);

            // Check for end-of-generation
            if unsafe { sys::llama_vocab_is_eog(vocab, token) } {
                break;
            }

            // Decode to text
            let n_written = unsafe {
                sys::llama_token_to_piece(
                    vocab,
                    token,
                    response_buf.as_mut_ptr(),
                    response_buf.len() as i32,
                    0,
                    true,
                )
            };
            if n_written > 0 {
                let piece = unsafe {
                    CStr::from_ptr(response_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                // Trim the piece to the number of bytes actually written
                let piece = &piece[..n_written.min(piece.len() as i32) as usize];
                on_token(piece)?;
                completion_tokens += 1;
            }

            // Feed generated token back
            let mut next = [token];
            let batch = unsafe { sys::llama_batch_get_one(next.as_mut_ptr(), 1) };
            let rc = unsafe { sys::llama_decode(ctx.as_ptr(), batch) };
            if rc != 0 {
                return Err(LlamaCppError::DecodeFailed(rc).into());
            }
        }

        unsafe { sys::llama_backend_free() };

        Ok(GenerateStats {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        })
    }
}
