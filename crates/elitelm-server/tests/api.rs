use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use elitelm_core::AppConfig;
use elitelm_server::build_router;
use http_body_util::BodyExt;
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
