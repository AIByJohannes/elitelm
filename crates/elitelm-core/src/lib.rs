use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result as AnyResult, anyhow};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("failed to read config {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config: {0}")]
    ParseConfig(#[from] serde_yaml::Error),
    #[error("config is missing default_backend")]
    MissingDefaultBackend,
    #[error("backend '{name}' is not configured")]
    UnknownBackendName { name: String },
    #[error("backend '{name}' uses kind '{kind}', which is not supported yet")]
    UnsupportedBackendKind { name: String, kind: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerateStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub trait InferenceBackend: Send {
    fn name(&self) -> &str;

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> AnyResult<()>,
    ) -> AnyResult<GenerateStats>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub default_backend: String,
    pub backends: BTreeMap<String, BackendConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendConfig {
    pub kind: String,
    #[serde(default)]
    pub response_prefix: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawAppConfig {
    default_backend: Option<String>,
    #[serde(default)]
    backends: BTreeMap<String, BackendConfig>,
}

impl AppConfig {
    pub fn from_yaml_str(input: &str) -> Result<Self, CoreError> {
        let raw: RawAppConfig = serde_yaml::from_str(input)?;
        let default_backend = raw
            .default_backend
            .filter(|value| !value.trim().is_empty())
            .ok_or(CoreError::MissingDefaultBackend)?;

        let config = Self {
            default_backend,
            backends: raw.backends,
        };
        config.resolve_backend_name(None)?;
        Ok(config)
    }

    pub fn resolve_backend_name<'a>(
        &'a self,
        requested: Option<&str>,
    ) -> Result<&'a str, CoreError> {
        let name = requested.unwrap_or(&self.default_backend);
        self.backends
            .get_key_value(name)
            .map(|(key, _)| key.as_str())
            .ok_or_else(|| CoreError::UnknownBackendName {
                name: name.to_string(),
            })
    }

    pub fn backend(&self, requested: Option<&str>) -> Result<(&str, &BackendConfig), CoreError> {
        let name = self.resolve_backend_name(requested)?;
        Ok(self
            .backends
            .get_key_value(name)
            .map(|(key, backend)| (key.as_str(), backend))
            .expect("resolve_backend_name verified backend presence"))
    }
}

pub fn load_config_file(path: impl AsRef<Path>) -> Result<AppConfig, CoreError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| CoreError::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    AppConfig::from_yaml_str(&input)
}

pub fn create_backend(
    config: &AppConfig,
    requested_backend: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, CoreError> {
    let (name, backend) = config.backend(requested_backend)?;
    match backend.kind.as_str() {
        "fake" => Ok(Box::new(FakeBackend::new(
            name,
            backend.response_prefix.as_deref(),
        ))),
        "genie" | "llamacpp" => Err(CoreError::UnsupportedBackendKind {
            name: name.to_string(),
            kind: backend.kind.clone(),
        }),
        other => Err(CoreError::UnsupportedBackendKind {
            name: name.to_string(),
            kind: other.to_string(),
        }),
    }
}

#[derive(Debug, Clone)]
pub struct FakeBackend {
    name: String,
    response_prefix: String,
}

impl FakeBackend {
    pub fn new(name: impl Into<String>, response_prefix: Option<&str>) -> Self {
        Self {
            name: name.into(),
            response_prefix: response_prefix
                .unwrap_or("EliteLM fake response: ")
                .to_string(),
        }
    }

    fn response_for(&self, request: &GenerateRequest) -> AnyResult<String> {
        let prompt = request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .or_else(|| request.messages.last())
            .ok_or_else(|| anyhow!("messages cannot be empty"))?;
        Ok(format!("{}{}", self.response_prefix, prompt.content))
    }
}

impl InferenceBackend for FakeBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> AnyResult<()>,
    ) -> AnyResult<GenerateStats> {
        let response = self.response_for(&request)?;
        let prompt_tokens = request
            .messages
            .iter()
            .map(|message| count_tokens(&message.content))
            .sum();
        let completion_tokens = count_tokens(&response);

        for piece in stream_pieces(&response) {
            on_token(piece)?;
        }

        Ok(GenerateStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        })
    }
}

fn count_tokens(input: &str) -> u32 {
    input
        .split_whitespace()
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn stream_pieces(input: &str) -> Vec<&str> {
    if input.is_empty() {
        return vec![""];
    }

    let mut pieces = Vec::new();
    let mut start = 0;
    for (idx, ch) in input.char_indices() {
        if ch.is_whitespace() && idx > start {
            pieces.push(&input[start..idx]);
            pieces.push(&input[idx..idx + ch.len_utf8()]);
            start = idx + ch.len_utf8();
        }
    }
    if start < input.len() {
        pieces.push(&input[start..]);
    }
    pieces
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

impl ChatCompletionRequest {
    pub fn into_generate_request(self) -> GenerateRequest {
        GenerateRequest {
            messages: self.messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<GenerateStats> for ChatCompletionUsage {
    fn from(stats: GenerateStats) -> Self {
        Self {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            total_tokens: stats.total_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionResponseChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(rename = "object")]
    pub object_type: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseChoice>,
    pub usage: ChatCompletionUsage,
}

impl ChatCompletionResponse {
    pub fn new(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: impl Into<String>,
        stats: GenerateStats,
    ) -> Self {
        Self {
            id: id.into(),
            object_type: "chat.completion".to_string(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionResponseChoice {
                index: 0,
                message: ChatMessage::new("assistant", content),
                finish_reason: Some("stop".to_string()),
            }],
            usage: stats.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionChunk {
    pub id: String,
    #[serde(rename = "object")]
    pub object_type: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

impl ChatCompletionChunk {
    pub fn token(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object_type: "chat.completion.chunk".to_string(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: Some(content.into()),
                },
                finish_reason: None,
            }],
        }
    }

    pub fn done(id: impl Into<String>, created: u64, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object_type: "chat.completion.chunk".to_string(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        }
    }
}

pub fn sse_json_line<T: Serialize>(value: &T) -> AnyResult<String> {
    Ok(format!("data: {}\n\n", serde_json::to_string(value)?))
}

pub fn sse_done_line() -> &'static str {
    "data: [DONE]\n\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_config() -> AppConfig {
        AppConfig::from_yaml_str(
            r#"
default_backend: fake
backends:
  fake:
    kind: fake
"#,
        )
        .unwrap()
    }

    #[test]
    fn config_requires_default_backend() {
        let error = AppConfig::from_yaml_str(
            r#"
backends:
  fake:
    kind: fake
"#,
        )
        .unwrap_err();

        assert!(matches!(error, CoreError::MissingDefaultBackend));
    }

    #[test]
    fn unknown_backend_name_is_rejected() {
        let config = fake_config();
        let error = match create_backend(&config, Some("missing")) {
            Ok(_) => panic!("expected missing backend to fail"),
            Err(error) => error,
        };

        assert!(matches!(error, CoreError::UnknownBackendName { .. }));
        assert_eq!(error.to_string(), "backend 'missing' is not configured");
    }

    #[test]
    fn unsupported_backend_kind_is_rejected() {
        let config = AppConfig::from_yaml_str(
            r#"
default_backend: genie_npu
backends:
  genie_npu:
    kind: genie
"#,
        )
        .unwrap();

        let error = match create_backend(&config, None) {
            Ok(_) => panic!("expected unsupported backend to fail"),
            Err(error) => error,
        };
        assert!(matches!(error, CoreError::UnsupportedBackendKind { .. }));
        assert_eq!(
            error.to_string(),
            "backend 'genie_npu' uses kind 'genie', which is not supported yet"
        );
    }

    #[test]
    fn fake_backend_streams_deterministic_response() {
        let config = fake_config();
        let mut backend = create_backend(&config, None).unwrap();
        let mut text = String::new();

        let stats = backend
            .generate(
                GenerateRequest {
                    messages: vec![ChatMessage::new("user", "hello rust")],
                    max_tokens: None,
                    temperature: None,
                    top_p: None,
                },
                &mut |piece| {
                    text.push_str(piece);
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(text, "EliteLM fake response: hello rust");
        assert_eq!(stats.prompt_tokens, 2);
        assert_eq!(stats.completion_tokens, 5);
    }

    #[test]
    fn serializes_openai_response() {
        let response = ChatCompletionResponse::new(
            "chatcmpl-test",
            123,
            "fake",
            "hello",
            GenerateStats {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
        );
        let value = serde_json::to_value(response).unwrap();

        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["usage"]["total_tokens"], 2);
    }

    #[test]
    fn serializes_streaming_chunk_line() {
        let chunk = ChatCompletionChunk::token("chatcmpl-test", 123, "fake", "hel");
        let line = sse_json_line(&chunk).unwrap();

        assert!(line.starts_with("data: {"));
        assert!(line.contains(r#""object":"chat.completion.chunk""#));
        assert!(line.contains(r#""content":"hel""#));
        assert_eq!(sse_done_line(), "data: [DONE]\n\n");
    }
}
