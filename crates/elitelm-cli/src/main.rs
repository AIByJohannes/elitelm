use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use elitelm_core::{ChatMessage, GenerateRequest, create_backend, load_config_file};

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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    let mut backend = create_backend(&config, args.backend.as_deref())?;
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
