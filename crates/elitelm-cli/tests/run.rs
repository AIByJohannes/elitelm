use assert_cmd::Command;
use tempfile::TempDir;

#[test]
fn run_uses_fake_backend() {
    let mut cmd = Command::cargo_bin("elitelm-cli").unwrap();
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();

    let output = cmd.current_dir(workspace_root)
        .arg("run")
        .arg("--config")
        .arg("elitelm.example.yaml")
        .arg("--backend")
        .arg("fake")
        .arg("--prompt")
        .arg("smoke test")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("EliteLM fake response: smoke test"));
    assert!(stdout.contains("Inference Statistics:"));
    assert!(stdout.contains("Prompt tokens:     2"));
    assert!(stdout.contains("Completion tokens: 5"));
}

#[test]
fn prepare_genie_bundle_uses_fixture_paths() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let bundle = root.join("bundle");
    let sdk = root.join("qairt");
    let configs = root.join("configs");
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::create_dir_all(&configs).unwrap();
    create_sdk_fixture(&sdk, "v73");
    std::fs::write(bundle.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(bundle.join("part_1.bin"), "ctx1").unwrap();
    std::fs::write(configs.join("htp.json.template"), htp_template()).unwrap();
    std::fs::write(configs.join("genie.json"), genie_template()).unwrap();
    let config_path = root.join("elitelm.genie.yaml");
    std::fs::write(
        &config_path,
        format!(
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
"#,
            yaml_path(&bundle),
            yaml_path(&bundle.join("genie_config.json")),
            yaml_path(&bundle.join("htp_backend_ext_config.json")),
            yaml_path(&sdk),
            yaml_path(&bundle.join("tokenizer.json")),
            yaml_path(&configs.join("genie.json")),
            yaml_path(&configs.join("htp.json.template")),
        ),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("elitelm-cli").unwrap();
    cmd.arg("prepare-genie-bundle")
        .arg("--config")
        .arg(&config_path)
        .arg("--backend")
        .arg("genie_npu")
        .assert()
        .success();

    assert!(bundle.join("genie_config.json").exists());
    assert!(bundle.join("htp_backend_ext_config.json").exists());
    assert!(bundle.join("Genie.dll").exists());
    assert!(bundle.join("tokenizer.json").exists());
}

fn create_sdk_fixture(root: &std::path::Path, dsp_arch: &str) {
    let arch_dir = root.join("bin").join("aarch64-windows-msvc");
    let lib_dir = root.join("lib").join("aarch64-windows-msvc");
    let hexagon_dir = root
        .join("lib")
        .join(format!("hexagon-{dsp_arch}"))
        .join("unsigned");
    std::fs::create_dir_all(&arch_dir).unwrap();
    std::fs::create_dir_all(&lib_dir).unwrap();
    std::fs::create_dir_all(&hexagon_dir).unwrap();

    for file in [
        arch_dir.join("genie-t2t-run.exe"),
        arch_dir.join("qnn-platform-validator.exe"),
        lib_dir.join("Genie.dll"),
        lib_dir.join("PlatformValidatorShared.dll"),
        lib_dir.join("QnnGenAiTransformer.dll"),
        lib_dir.join("QnnGenAiTransformerModel.dll"),
        lib_dir.join("QnnHtp.dll"),
        lib_dir.join("QnnHtpNetRunExtensions.dll"),
        lib_dir.join("QnnHtpPrepare.dll"),
        lib_dir.join(format!("QnnHtp{dsp_arch}Stub.dll")),
        lib_dir.join(format!("QnnHtp{dsp_arch}CalculatorStub.dll")),
        lib_dir.join("QnnSystem.dll"),
        hexagon_dir.join("libCalculator_skel.so"),
        hexagon_dir.join(format!("libQnnHtp{dsp_arch}.cat")),
        hexagon_dir.join(format!("libQnnHtp{dsp_arch}Skel.so")),
    ] {
        std::fs::write(file, "runtime").unwrap();
    }
}

fn yaml_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn genie_template() -> &'static str {
    r#"{
  "dialog": {
    "tokenizer": {
      "path": "tokenizer.json"
    },
    "engine": {
      "backend": {
        "QnnHtp": {
          "use-mmap": true
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "binary": {
          "ctx-bins": ["part_1.bin"]
        }
      }
    }
  }
}"#
}

fn htp_template() -> &'static str {
    r#"{
  "devices": [
    {
      "soc_model": <TODO>,
      "dsp_arch": "<TODO>"
    }
  ]
}"#
}
