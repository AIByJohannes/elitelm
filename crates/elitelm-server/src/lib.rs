use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use elitelm_backend_genie::GenieBackend;
use elitelm_backend_llamacpp::LlamaCppBackend;
use elitelm_core::{
    AppConfig, BackendConfig, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    CoreError, GenerateRequest, GenerateStats, InferenceBackend, ModelListResponse, ModelObject,
    create_fake_backend, get_elitelm_models_dir, model_filename, sse_done_line, sse_json_line,
};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    config: Arc<AppConfig>,
    backend_name: Option<String>,
}

pub fn build_router(config: AppConfig, backend_name: Option<String>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(retrieve_model))
        .with_state(AppState {
            config: Arc::new(config),
            backend_name,
        })
}

async fn create_chat_completion(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    if request.messages.is_empty() {
        return Err(AppError::bad_request("messages cannot be empty"));
    }

    let model = request.model.clone().unwrap_or_else(|| {
        state
            .backend_name
            .clone()
            .unwrap_or_else(|| state.config.default_backend.clone())
    });
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let config = Arc::clone(&state.config);
    let backend_name = state.backend_name.clone();
    let generate_request = request.into_generate_request();
    let generated = tokio::task::spawn_blocking(move || {
        let mut backend = create_backend_for_server(&config, backend_name.as_deref())?;
        generate_text(&mut *backend, generate_request)
    })
    .await
    .map_err(|error| AppError::internal(format!("generation task failed: {error}")))??;
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_timestamp();

    if stream {
        let body = stream_body(
            &id,
            created,
            &model,
            &generated.pieces,
            generated.stats,
            include_usage,
        )?;
        Ok((
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from(body),
        )
            .into_response())
    } else {
        let response =
            ChatCompletionResponse::new(id, created, model, generated.text, generated.stats);
        Ok(Json(response).into_response())
    }
}

async fn list_models(
    State(state): State<AppState>,
) -> Json<ModelListResponse> {
    let mut data = Vec::new();
    let created = unix_timestamp();
    for backend_name in state.config.backends.keys() {
        data.push(ModelObject {
            id: backend_name.clone(),
            object: "model".to_string(),
            created,
            owned_by: "elitelm".to_string(),
        });
    }

    // Include downloaded registry models
    let models_dir = get_elitelm_models_dir();
    if models_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "gguf") {
                    if let Some(stem) = path.file_stem() {
                        let filename = stem.to_string_lossy().to_string();
                        let reconstructed = if let Some(idx) = filename.rfind('_') {
                            let mut r = filename.clone();
                            r.replace_range(idx..=idx, ":");
                            r
                        } else {
                            filename.clone()
                        };

                        if !state.config.backends.contains_key(&reconstructed) {
                            data.push(ModelObject {
                                id: reconstructed,
                                object: "model".to_string(),
                                created,
                                owned_by: "elitelm".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

async fn retrieve_model(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<ModelObject>, AppError> {
    let exists_in_config = state.config.backends.contains_key(&model_name);
    let mut exists_on_disk = false;

    if !exists_in_config {
        let models_dir = get_elitelm_models_dir();
        let filename = model_filename(&model_name);
        if models_dir.join(&filename).exists() {
            exists_on_disk = true;
        }
    }

    if exists_in_config || exists_on_disk {
        let created = unix_timestamp();
        Ok(Json(ModelObject {
            id: model_name,
            object: "model".to_string(),
            created,
            owned_by: "elitelm".to_string(),
        }))
    } else {
        Err(AppError::not_found(format!("model '{}' not found", model_name)))
    }
}

struct GeneratedText {
    text: String,
    pieces: Vec<String>,
    stats: GenerateStats,
}

fn generate_text(
    backend: &mut dyn elitelm_core::InferenceBackend,
    request: GenerateRequest,
) -> anyhow::Result<GeneratedText> {
    let mut text = String::new();
    let mut pieces = Vec::new();
    let stats = backend.generate(request, &mut |piece| {
        text.push_str(piece);
        pieces.push(piece.to_string());
        Ok(())
    })?;

    Ok(GeneratedText {
        text,
        pieces,
        stats,
    })
}

fn stream_body(
    id: &str,
    created: u64,
    model: &str,
    pieces: &[String],
    stats: GenerateStats,
    include_usage: bool,
) -> anyhow::Result<String> {
    let mut body = String::new();

    // 1. Send first chunk declaring the role
    let first_chunk = ChatCompletionChunk::first_chunk(id, created, model);
    body.push_str(&sse_json_line(&first_chunk)?);

    // 2. Send token chunks
    for piece in pieces {
        let chunk = ChatCompletionChunk::token(id, created, model, piece);
        body.push_str(&sse_json_line(&chunk)?);
    }

    // 3. Send done chunk
    let final_chunk = ChatCompletionChunk::done(id, created, model);
    body.push_str(&sse_json_line(&final_chunk)?);

    // 4. Send usage chunk if requested
    if include_usage {
        let usage_chunk = ChatCompletionChunk::usage_chunk(id, created, model, stats.into());
        body.push_str(&sse_json_line(&usage_chunk)?);
    }

    body.push_str(sse_done_line());
    Ok(body)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl From<CoreError> for AppError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::UnknownBackendName { .. } | CoreError::UnsupportedBackendKind { .. } => {
                Self {
                    status: StatusCode::BAD_REQUEST,
                    message: error.to_string(),
                }
            }
            _ => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: error.to_string(),
            },
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

fn create_backend_for_server(
    config: &AppConfig,
    requested_backend: Option<&str>,
) -> anyhow::Result<Box<dyn InferenceBackend>> {
    if let Some(name) = requested_backend {
        if config.backends.contains_key(name) {
            // Exist in config file, load as configured
            let (_, backend_config) = config.backend(Some(name))?;
            return build_backend_from_config(name, backend_config);
        }

        // Not in config file. Check if it exists as a pulled GGUF model.
        let models_dir = get_elitelm_models_dir();
        let filename = model_filename(name);
        let local_path = models_dir.join(&filename);
        if local_path.exists() {
            let llama_config = elitelm_core::LlamaCppBackendConfig {
                model: local_path,
                n_threads: None,
                n_ctx: None,
                n_batch: None,
                use_mmap: true,
            };
            return Ok(Box::new(LlamaCppBackend::new(
                name.to_string(),
                llama_config,
            )?));
        }
    }

    let (name, backend_config) = config.backend(requested_backend)?;
    build_backend_from_config(name, backend_config)
}

fn build_backend_from_config(
    name: &str,
    backend_config: &BackendConfig,
) -> anyhow::Result<Box<dyn InferenceBackend>> {
    match backend_config {
        BackendConfig::Fake(fake_config) => Ok(create_fake_backend(name, fake_config)),
        BackendConfig::Genie(genie_config) => Ok(Box::new(GenieBackend::new(
            name,
            genie_config.as_ref().clone(),
        )?)),
        BackendConfig::LlamaCpp(llama_config) => Ok(Box::new(LlamaCppBackend::new(
            name,
            llama_config.as_ref().clone(),
        )?)),
    }
}
