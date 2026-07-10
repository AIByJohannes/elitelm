use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use elitelm_core::AppConfig;
use elitelm_server::build_router;
use http_body_util::BodyExt;
use tempfile::TempDir;
use tower::ServiceExt;

fn test_config() -> AppConfig {
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

#[tokio::test]
async fn non_streaming_chat_completions() {
    let app = build_router(test_config(), Some("fake".to_string()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"fake","messages":[{"role":"user","content":"hello api"}]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "EliteLM fake response: hello api"
    );
}

#[tokio::test]
async fn streaming_chat_completions() {
    let app = build_router(test_config(), Some("fake".to_string()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"fake","stream":true,"messages":[{"role":"user","content":"hello stream"}]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("data: {"));
    assert!(text.contains(r#""object":"chat.completion.chunk""#));
    assert!(text.contains(r#""content":"hello""#));
    assert!(text.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn genie_backend_chat_completions_uses_process_output() {
    let temp = TempDir::new().unwrap();
    let bundle = temp.path().join("bundle");
    std::fs::create_dir_all(&bundle).unwrap();
    let genie_config = bundle.join("genie_config.json");
    std::fs::write(&genie_config, "{}").unwrap();
    let fake_executable = create_fake_executable(&bundle);
    let config = AppConfig::from_yaml_str(&format!(
        r#"
default_backend: genie_npu
backends:
  genie_npu:
    kind: genie
    bundle_dir: {}
    genie_config: {}
    htp_config: {}
    qnn_sdk_root: {}
    tokenizer_path: {}
    genie_config_template: {}
    htp_config_template: {}
    genie_executable: {}
"#,
        yaml_path(&bundle),
        yaml_path(&genie_config),
        yaml_path(&bundle.join("htp_backend_ext_config.json")),
        yaml_path(temp.path()),
        yaml_path(&bundle.join("tokenizer.json")),
        yaml_path(&bundle.join("genie-template.json")),
        yaml_path(&bundle.join("htp-template.json")),
        yaml_path(&fake_executable),
    ))
    .unwrap();
    let app = build_router(config, Some("genie_npu".to_string()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"genie_npu","messages":[{"role":"user","content":"hello genie"}]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("fake genie output")
    );
}

#[tokio::test]
async fn list_models() {
    let app = build_router(test_config(), Some("fake".to_string()));
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["object"], "list");
    let data = value["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "fake");
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["owned_by"], "elitelm");
    assert!(data[0]["created"].is_number());
}

#[tokio::test]
async fn retrieve_model() {
    let app = build_router(test_config(), Some("fake".to_string()));
    
    // Success
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models/fake")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["id"], "fake");
    assert_eq!(value["object"], "model");
    assert_eq!(value["owned_by"], "elitelm");

    // Not Found
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models/unknown")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn streaming_chat_completions_with_usage() {
    let app = build_router(test_config(), Some("fake".to_string()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"fake","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello stream with usage"}]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    
    // Verify first chunk has role: "assistant" and content: ""
    assert!(text.contains(r#""delta":{"role":"assistant","content":""}"#));
    
    // Verify we have a usage chunk at the end with empty choices
    assert!(text.contains(r#""choices":[]"#));
    assert!(text.contains(r#""usage":{"prompt_tokens":"#));
    assert!(text.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn chat_completions_max_completion_tokens() {
    let app = build_router(test_config(), Some("fake".to_string()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"model":"fake","max_completion_tokens":10,"messages":[{"role":"user","content":"hello api"}]}"#,
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["object"], "chat.completion");
}

fn yaml_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn create_fake_executable(dir: &std::path::Path) -> std::path::PathBuf {
    #[cfg(windows)]
    {
        let path = dir.join("fake-genie-ok.cmd");
        std::fs::write(&path, "@echo off\r\necho fake genie output %*\r\n").unwrap();
        path
    }

    #[cfg(not(windows))]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake-genie-ok.sh");
        std::fs::write(&path, "#!/bin/sh\necho fake genie output \"$@\"\n").unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }
}
