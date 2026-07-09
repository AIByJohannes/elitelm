use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use elitelm_core::load_config_file;
use elitelm_server::build_router;

#[derive(Debug, Parser)]
#[command(name = "elitelm-server")]
#[command(about = "EliteLM Rust API server shell")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(ServeArgs),
}

#[derive(Debug, Parser)]
struct ServeArgs {
    #[arg(long, default_value = "elitelm.example.yaml")]
    config: PathBuf,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve(args).await,
    }
}

async fn serve(args: ServeArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    let app = build_router(config, args.backend);
    let address: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;

    println!("EliteLM server listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}
