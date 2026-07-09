use elitelm_backend_genie::{GenieBackend, prepare_genie_bundle};
use elitelm_core::{
    BackendConfig, ChatMessage, GenerateRequest, InferenceBackend, load_config_file,
};

#[test]
#[ignore = "requires Snapdragon Windows hardware, QAIRT SDK, and the local Genie bundle"]
fn runs_real_local_genie_bundle() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let config = load_config_file(workspace_root.join("elitelm.genie.example.yaml")).unwrap();

    let (_, backend) = config.backend(None).unwrap();
    let BackendConfig::Genie(genie_config) = backend else {
        panic!("expected genie backend");
    };

    prepare_genie_bundle(genie_config).unwrap();
    let mut backend = GenieBackend::new("genie_npu", genie_config.as_ref().clone()).unwrap();
    let mut output = String::new();
    backend
        .generate(
            GenerateRequest {
                messages: vec![ChatMessage::new(
                    "user",
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSay hello in five words.<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                )],
                max_tokens: None,
                temperature: None,
                top_p: None,
            },
            &mut |piece| {
                output.push_str(piece);
                Ok(())
            },
        )
        .unwrap();

    assert!(!output.trim().is_empty());
}
