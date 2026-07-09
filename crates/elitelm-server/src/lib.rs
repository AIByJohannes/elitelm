use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use elitelm_core::{
    AppConfig, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CoreError,
    GenerateStats, create_backend, sse_done_line, sse_json_line,
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

    let mut backend = create_backend(&state.config, state.backend_name.as_deref())?;
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| backend.name().to_string());
    let stream = request.stream;
    let generated = generate_text(&mut *backend, request.into_generate_request())?;
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_timestamp();

    if stream {
        let body = stream_body(&id, created, &model, &generated.pieces)?;
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

struct GeneratedText {
    text: String,
    pieces: Vec<String>,
    stats: GenerateStats,
}

fn generate_text(
    backend: &mut dyn elitelm_core::InferenceBackend,
    request: elitelm_core::GenerateRequest,
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

fn stream_body(id: &str, created: u64, model: &str, pieces: &[String]) -> anyhow::Result<String> {
    let mut body = String::new();
    for piece in pieces {
        let chunk = ChatCompletionChunk::token(id, created, model, piece);
        body.push_str(&sse_json_line(&chunk)?);
    }
    let final_chunk = ChatCompletionChunk::done(id, created, model);
    body.push_str(&sse_json_line(&final_chunk)?);
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
