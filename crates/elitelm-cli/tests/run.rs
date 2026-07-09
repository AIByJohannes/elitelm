use assert_cmd::Command;

#[test]
fn run_uses_fake_backend() {
    let mut cmd = Command::cargo_bin("elitelm-cli").unwrap();
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();

    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("--config")
        .arg("elitelm.example.yaml")
        .arg("--backend")
        .arg("fake")
        .arg("--prompt")
        .arg("smoke test")
        .assert()
        .success()
        .stdout("EliteLM fake response: smoke test\n");
}
