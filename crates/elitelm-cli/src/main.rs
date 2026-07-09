use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use elitelm_backend_genie::{GenieBackend, prepare_genie_bundle};
use elitelm_backend_llamacpp::LlamaCppBackend;
use elitelm_core::{
    BackendConfig, ChatMessage, GenerateRequest, InferenceBackend, create_fake_backend,
    load_config_file,
};

#[derive(Debug, Parser)]
#[command(name = "elitelm")]
#[command(about = "EliteLM Rust CLI shell")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run(RunArgs),
    PrepareGenieBundle(PrepareGenieBundleArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    #[arg(long, default_value = "elitelm.example.yaml")]
    config: PathBuf,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long, default_value = "Hello from EliteLM")]
    prompt: String,
}

#[derive(Debug, Parser)]
struct PrepareGenieBundleArgs {
    #[arg(long, default_value = "elitelm.genie.example.yaml")]
    config: PathBuf,
    #[arg(long)]
    backend: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::PrepareGenieBundle(args) => prepare(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    let mut backend = create_backend_for_cli(&config, args.backend.as_deref())?;
    let request = GenerateRequest {
        messages: vec![ChatMessage::new("user", args.prompt)],
        max_tokens: None,
        temperature: None,
        top_p: None,
    };

    backend.generate(request, &mut |piece| {
        print!("{piece}");
        Ok(())
    })?;
    println!();
    Ok(())
}

fn prepare(args: PrepareGenieBundleArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    let (_, backend_config) = config.backend(Some(&args.backend))?;
    let BackendConfig::Genie(genie_config) = backend_config else {
        return Err(anyhow!(
            "backend '{}' uses kind '{}', expected genie",
            args.backend,
            backend_config.kind()
        ));
    };

    let prepared = prepare_genie_bundle(genie_config)?;
    println!("Generated {}", prepared.htp_config.display());
    println!("Generated {}", prepared.genie_config.display());
    println!("Copied {} runtime files", prepared.copied_files.len());
    println!("Validated {} context binaries", prepared.context_bins.len());
    Ok(())
}

fn create_backend_for_cli(
    config: &elitelm_core::AppConfig,
    requested_backend: Option<&str>,
) -> Result<Box<dyn InferenceBackend>> {
    let (name, backend_config) = config.backend(requested_backend)?;
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
