use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result as AnyResult, anyhow};
use elitelm_core::{GenerateRequest, GenerateStats, GenieBackendConfig, InferenceBackend};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenieError {
    #[error("{label} does not exist: {path}")]
    MissingPath { label: &'static str, path: PathBuf },
    #[error("{label} is not a file: {path}")]
    NotFile { label: &'static str, path: PathBuf },
    #[error("{label} is not a directory: {path}")]
    NotDirectory { label: &'static str, path: PathBuf },
    #[error("failed to parse Genie config template {path}: {source}")]
    ParseGenieConfig {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("Genie config template is missing dialog.engine.model.binary.ctx-bins")]
    MissingCtxBins,
    #[error("Genie exited with status {status}: {stderr}")]
    ProcessFailed { status: String, stderr: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedGenieBundle {
    pub genie_config: PathBuf,
    pub htp_config: PathBuf,
    pub copied_files: Vec<PathBuf>,
    pub context_bins: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct GenieBackend {
    name: String,
    config: GenieBackendConfig,
}

impl GenieBackend {
    pub fn new(name: impl Into<String>, config: GenieBackendConfig) -> AnyResult<Self> {
        validate_file("Genie config", &config.genie_config)?;
        validate_dir("Genie bundle directory", &config.bundle_dir)?;
        let executable = effective_genie_executable(&config);
        validate_file("Genie executable", &executable)?;

        Ok(Self {
            name: name.into(),
            config,
        })
    }

    fn prompt_from(request: &GenerateRequest) -> AnyResult<String> {
        let prompt = request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .or_else(|| request.messages.last())
            .ok_or_else(|| anyhow!("messages cannot be empty"))?;
        Ok(prompt.content.clone())
    }
}

impl InferenceBackend for GenieBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> AnyResult<()>,
    ) -> AnyResult<GenerateStats> {
        let prompt = Self::prompt_from(&request)?;
        let executable = effective_genie_executable(&self.config);
        let output = Command::new(&executable)
            .current_dir(&self.config.bundle_dir)
            .arg("-c")
            .arg(&self.config.genie_config)
            .arg("-p")
            .arg(&prompt)
            .env("ADSP_LIBRARY_PATH", &self.config.bundle_dir)
            .output()
            .with_context(|| {
                format!("failed to spawn Genie executable {}", executable.display())
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(GenieError::ProcessFailed {
                status: output.status.to_string(),
                stderr,
            }
            .into());
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        on_token(&text)?;
        let prompt_tokens = count_tokens(&prompt);
        let completion_tokens = count_tokens(&text);
        Ok(GenerateStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        })
    }
}

pub fn prepare_genie_bundle(config: &GenieBackendConfig) -> AnyResult<PreparedGenieBundle> {
    validate_prepare_inputs(config)?;
    fs::create_dir_all(&config.bundle_dir).with_context(|| {
        format!(
            "failed to create bundle directory {}",
            config.bundle_dir.display()
        )
    })?;

    let ctx_bin_names = read_context_bin_names(&config.genie_config_template)?;
    let context_bins = validate_context_bins(&config.bundle_dir, &ctx_bin_names)?;
    let htp_config = render_htp_config(config)?;
    fs::write(&config.htp_config, htp_config)
        .with_context(|| format!("failed to write {}", config.htp_config.display()))?;

    let genie_config = render_genie_config(config, &ctx_bin_names)?;
    fs::write(&config.genie_config, genie_config)
        .with_context(|| format!("failed to write {}", config.genie_config.display()))?;

    let copied_files = copy_runtime_files(config)?;

    Ok(PreparedGenieBundle {
        genie_config: config.genie_config.clone(),
        htp_config: config.htp_config.clone(),
        copied_files,
        context_bins,
    })
}

pub fn required_runtime_files(config: &GenieBackendConfig) -> Vec<(PathBuf, PathBuf)> {
    let arch = "aarch64-windows-msvc";
    let dsp_arch = &config.dsp_arch;
    let sdk = &config.qnn_sdk_root;
    let bundle = &config.bundle_dir;
    let mut files = vec![
        (
            sdk.join("bin").join(arch).join("genie-t2t-run.exe"),
            bundle.join("genie-t2t-run.exe"),
        ),
        (
            sdk.join("bin")
                .join(arch)
                .join("qnn-platform-validator.exe"),
            bundle.join("qnn-platform-validator.exe"),
        ),
        (
            sdk.join("lib").join(arch).join("Genie.dll"),
            bundle.join("Genie.dll"),
        ),
        (
            sdk.join("lib")
                .join(arch)
                .join("PlatformValidatorShared.dll"),
            bundle.join("PlatformValidatorShared.dll"),
        ),
        (
            sdk.join("lib").join(arch).join("QnnGenAiTransformer.dll"),
            bundle.join("QnnGenAiTransformer.dll"),
        ),
        (
            sdk.join("lib")
                .join(arch)
                .join("QnnGenAiTransformerModel.dll"),
            bundle.join("QnnGenAiTransformerModel.dll"),
        ),
        (
            sdk.join("lib").join(arch).join("QnnHtp.dll"),
            bundle.join("QnnHtp.dll"),
        ),
        (
            sdk.join("lib")
                .join(arch)
                .join("QnnHtpNetRunExtensions.dll"),
            bundle.join("QnnHtpNetRunExtensions.dll"),
        ),
        (
            sdk.join("lib").join(arch).join("QnnHtpPrepare.dll"),
            bundle.join("QnnHtpPrepare.dll"),
        ),
        (
            sdk.join("lib")
                .join(arch)
                .join(format!("QnnHtp{dsp_arch}Stub.dll")),
            bundle.join(format!("QnnHtp{dsp_arch}Stub.dll")),
        ),
        (
            sdk.join("lib")
                .join(arch)
                .join(format!("QnnHtp{dsp_arch}CalculatorStub.dll")),
            bundle.join(format!("QnnHtp{dsp_arch}CalculatorStub.dll")),
        ),
        (
            sdk.join("lib").join(arch).join("QnnSystem.dll"),
            bundle.join("QnnSystem.dll"),
        ),
    ];

    let hexagon = sdk
        .join("lib")
        .join(format!("hexagon-{dsp_arch}"))
        .join("unsigned");
    files.extend([
        (
            hexagon.join("libCalculator_skel.so"),
            bundle.join("libCalculator_skel.so"),
        ),
        (
            hexagon.join(format!("libQnnHtp{dsp_arch}.cat")),
            bundle.join(format!("libQnnHtp{dsp_arch}.cat")),
        ),
        (
            hexagon.join(format!("libQnnHtp{dsp_arch}Skel.so")),
            bundle.join(format!("libQnnHtp{dsp_arch}Skel.so")),
        ),
    ]);
    files
}

fn validate_prepare_inputs(config: &GenieBackendConfig) -> AnyResult<()> {
    validate_dir("QNN SDK root", &config.qnn_sdk_root)?;
    validate_dir("Genie bundle directory", &config.bundle_dir)?;
    validate_file("Genie config template", &config.genie_config_template)?;
    validate_file("HTP config template", &config.htp_config_template)?;
    validate_file("Tokenizer", &config.tokenizer_path)?;

    for (source, _) in required_runtime_files(config) {
        validate_file("QNN/Genie runtime file", &source)?;
    }

    Ok(())
}

fn validate_context_bins(bundle_dir: &Path, names: &[String]) -> AnyResult<Vec<PathBuf>> {
    names
        .iter()
        .map(|name| {
            let path = bundle_dir.join(name);
            validate_file("Genie context binary", &path)?;
            Ok(path)
        })
        .collect()
}

fn read_context_bin_names(path: &Path) -> AnyResult<Vec<String>> {
    let config = read_genie_config(path)?;
    let bins = config
        .pointer("/dialog/engine/model/binary/ctx-bins")
        .and_then(Value::as_array)
        .ok_or(GenieError::MissingCtxBins)?;

    let names = bins
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow!("ctx-bins must contain only strings"))
        })
        .collect::<AnyResult<Vec<_>>>()?;

    if names.is_empty() {
        Err(GenieError::MissingCtxBins.into())
    } else {
        Ok(names)
    }
}

fn render_htp_config(config: &GenieBackendConfig) -> AnyResult<String> {
    let template = fs::read_to_string(&config.htp_config_template)
        .with_context(|| format!("failed to read {}", config.htp_config_template.display()))?;
    Ok(template
        .replace(
            "\"soc_model\": <TODO>",
            &format!("\"soc_model\": {}", config.soc_model),
        )
        .replace(
            "\"dsp_arch\": \"<TODO>\"",
            &format!("\"dsp_arch\": \"{}\"", config.dsp_arch),
        ))
}

fn render_genie_config(config: &GenieBackendConfig, ctx_bin_names: &[String]) -> AnyResult<String> {
    let mut value = read_genie_config(&config.genie_config_template)?;
    *value
        .pointer_mut("/dialog/engine/backend/QnnHtp/use-mmap")
        .ok_or_else(|| anyhow!("Genie config template is missing QnnHtp.use-mmap"))? =
        Value::Bool(false);
    *value
        .pointer_mut("/dialog/tokenizer/path")
        .ok_or_else(|| anyhow!("Genie config template is missing tokenizer.path"))? =
        Value::String(path_for_json(&config.tokenizer_path));
    *value
        .pointer_mut("/dialog/engine/backend/extensions")
        .ok_or_else(|| anyhow!("Genie config template is missing backend.extensions"))? =
        Value::String(path_for_json(&config.htp_config));

    let bins = value
        .pointer_mut("/dialog/engine/model/binary/ctx-bins")
        .and_then(Value::as_array_mut)
        .ok_or(GenieError::MissingCtxBins)?;
    bins.clear();
    bins.extend(
        ctx_bin_names
            .iter()
            .map(|name| Value::String(path_for_json(&config.bundle_dir.join(name)))),
    );

    Ok(format!("{}\n", serde_json::to_string_pretty(&value)?))
}

fn read_genie_config(path: &Path) -> AnyResult<Value> {
    let input =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&input).map_err(|source| {
        GenieError::ParseGenieConfig {
            path: path.to_path_buf(),
            source,
        }
        .into()
    })
}

fn copy_runtime_files(config: &GenieBackendConfig) -> AnyResult<Vec<PathBuf>> {
    let mut copied = Vec::new();
    for (source, destination) in required_runtime_files(config) {
        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to copy runtime file {} to {}",
                source.display(),
                destination.display()
            )
        })?;
        copied.push(destination);
    }
    Ok(copied)
}

fn effective_genie_executable(config: &GenieBackendConfig) -> PathBuf {
    config
        .genie_executable
        .clone()
        .unwrap_or_else(|| config.bundle_dir.join("genie-t2t-run.exe"))
}

fn validate_file(label: &'static str, path: &Path) -> AnyResult<()> {
    if !path.exists() {
        return Err(GenieError::MissingPath {
            label,
            path: path.to_path_buf(),
        }
        .into());
    }
    if !path.is_file() {
        return Err(GenieError::NotFile {
            label,
            path: path.to_path_buf(),
        }
        .into());
    }
    Ok(())
}

fn validate_dir(label: &'static str, path: &Path) -> AnyResult<()> {
    if !path.exists() {
        return Err(GenieError::MissingPath {
            label,
            path: path.to_path_buf(),
        }
        .into());
    }
    if !path.is_dir() {
        return Err(GenieError::NotDirectory {
            label,
            path: path.to_path_buf(),
        }
        .into());
    }
    Ok(())
}

fn path_for_json(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

fn count_tokens(input: &str) -> u32 {
    input
        .split_whitespace()
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    #[cfg(not(windows))]
    use std::io::Write;

    use super::*;
    use tempfile::TempDir;

    struct Fixture {
        _temp: TempDir,
        config: GenieBackendConfig,
    }

    impl Fixture {
        fn new() -> Self {
            let temp = TempDir::new().unwrap();
            let root = temp.path();
            let bundle = root.join("bundle");
            let sdk = root.join("qairt");
            let configs = root.join("configs");
            fs::create_dir_all(&bundle).unwrap();
            fs::create_dir_all(&configs).unwrap();
            create_sdk_fixture(&sdk, "v73");

            fs::write(bundle.join("tokenizer.json"), "{}").unwrap();
            fs::write(bundle.join("part_1.bin"), "ctx1").unwrap();
            fs::write(bundle.join("part_2.bin"), "ctx2").unwrap();
            fs::write(
                configs.join("htp_backend_ext_config.json.template"),
                htp_template(),
            )
            .unwrap();
            fs::write(configs.join("genie.json"), genie_template()).unwrap();

            let config = GenieBackendConfig {
                bundle_dir: bundle.clone(),
                genie_config: bundle.join("genie_config.json"),
                htp_config: bundle.join("htp_backend_ext_config.json"),
                qnn_sdk_root: sdk,
                tokenizer_path: bundle.join("tokenizer.json"),
                genie_config_template: configs.join("genie.json"),
                htp_config_template: configs.join("htp_backend_ext_config.json.template"),
                soc_model: 60,
                dsp_arch: "v73".to_string(),
                genie_executable: None,
            };

            Self {
                _temp: temp,
                config,
            }
        }
    }

    #[test]
    fn prepare_generates_configs_and_copies_runtime_files() {
        let fixture = Fixture::new();

        let prepared = prepare_genie_bundle(&fixture.config).unwrap();

        assert_eq!(prepared.context_bins.len(), 2);
        assert_eq!(prepared.copied_files.len(), 15);
        let htp = fs::read_to_string(&fixture.config.htp_config).unwrap();
        assert!(htp.contains("\"soc_model\": 60"));
        assert!(htp.contains("\"dsp_arch\": \"v73\""));

        let genie: Value =
            serde_json::from_str(&fs::read_to_string(&fixture.config.genie_config).unwrap())
                .unwrap();
        assert_eq!(
            genie.pointer("/dialog/engine/backend/QnnHtp/use-mmap"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            genie
                .pointer("/dialog/tokenizer/path")
                .unwrap()
                .as_str()
                .unwrap(),
            path_for_json(&fixture.config.tokenizer_path)
        );
        assert!(
            genie
                .pointer("/dialog/engine/model/binary/ctx-bins/0")
                .unwrap()
                .as_str()
                .unwrap()
                .ends_with("part_1.bin")
        );
        assert!(fixture.config.bundle_dir.join("Genie.dll").exists());
        assert!(fixture.config.bundle_dir.join("tokenizer.json").exists());
    }

    #[test]
    fn prepare_rejects_missing_context_binary() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.config.bundle_dir.join("part_2.bin")).unwrap();

        let error = prepare_genie_bundle(&fixture.config).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Genie context binary does not exist")
        );
    }

    #[test]
    fn required_runtime_files_match_powershell_reference_count() {
        let fixture = Fixture::new();
        let files = required_runtime_files(&fixture.config);

        assert_eq!(files.len(), 15);
        assert!(
            files
                .iter()
                .any(|(source, _)| source.ends_with("bin/aarch64-windows-msvc/genie-t2t-run.exe"))
        );
        assert!(
            files
                .iter()
                .any(|(source, _)| source.ends_with("lib/hexagon-v73/unsigned/libQnnHtpv73Skel.so"))
        );
    }

    #[test]
    fn process_backend_streams_stdout() {
        let fixture = Fixture::new();
        prepare_genie_bundle(&fixture.config).unwrap();
        let fake = create_fake_executable(&fixture.config.bundle_dir, false);
        let mut config = fixture.config.clone();
        config.genie_executable = Some(fake);
        let mut backend = GenieBackend::new("genie_npu", config).unwrap();
        let mut text = String::new();

        let stats = backend
            .generate(
                GenerateRequest {
                    messages: vec![elitelm_core::ChatMessage::new("user", "hello genie")],
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

        assert!(text.contains("fake genie output"));
        assert_eq!(stats.prompt_tokens, 2);
        assert!(stats.completion_tokens > 0);
    }

    #[test]
    fn process_backend_reports_nonzero_exit() {
        let fixture = Fixture::new();
        prepare_genie_bundle(&fixture.config).unwrap();
        let fake = create_fake_executable(&fixture.config.bundle_dir, true);
        let mut config = fixture.config.clone();
        config.genie_executable = Some(fake);
        let mut backend = GenieBackend::new("genie_npu", config).unwrap();

        let error = backend
            .generate(
                GenerateRequest {
                    messages: vec![elitelm_core::ChatMessage::new("user", "hello genie")],
                    max_tokens: None,
                    temperature: None,
                    top_p: None,
                },
                &mut |_| Ok(()),
            )
            .unwrap_err();

        assert!(error.to_string().contains("Genie exited with status"));
        assert!(error.to_string().contains("fake genie failure"));
    }

    fn create_sdk_fixture(root: &Path, dsp_arch: &str) {
        let arch_dir = root.join("bin").join("aarch64-windows-msvc");
        let lib_dir = root.join("lib").join("aarch64-windows-msvc");
        let hexagon_dir = root
            .join("lib")
            .join(format!("hexagon-{dsp_arch}"))
            .join("unsigned");
        fs::create_dir_all(&arch_dir).unwrap();
        fs::create_dir_all(&lib_dir).unwrap();
        fs::create_dir_all(&hexagon_dir).unwrap();

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
            fs::write(file, "runtime").unwrap();
        }
    }

    fn create_fake_executable(dir: &Path, fail: bool) -> PathBuf {
        #[cfg(windows)]
        {
            let path = dir.join(if fail {
                "fake-genie-fail.cmd"
            } else {
                "fake-genie-ok.cmd"
            });
            let body = if fail {
                "@echo off\r\necho fake genie failure 1>&2\r\nexit /b 7\r\n"
            } else {
                "@echo off\r\necho fake genie output %*\r\n"
            };
            fs::write(&path, body).unwrap();
            path
        }

        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = dir.join(if fail {
                "fake-genie-fail.sh"
            } else {
                "fake-genie-ok.sh"
            });
            let body = if fail {
                "#!/bin/sh\necho fake genie failure >&2\nexit 7\n"
            } else {
                "#!/bin/sh\necho fake genie output \"$@\"\n"
            };
            let mut file = fs::File::create(&path).unwrap();
            file.write_all(body.as_bytes()).unwrap();
            let mut permissions = file.metadata().unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).unwrap();
            path
        }
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
          "ctx-bins": ["part_1.bin", "part_2.bin"]
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
}
