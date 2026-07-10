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
    Benchmark(BenchmarkArgs),
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

#[derive(Debug, Parser)]
struct BenchmarkArgs {
    #[arg(long, default_value = "elitelm.example.yaml")]
    config: PathBuf,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long, default_value = "Why is the sky blue? Answer in 1 sentence.")]
    prompt: String,
    #[arg(long, default_value = "3")]
    runs: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::PrepareGenieBundle(args) => prepare(args),
        Command::Benchmark(args) => benchmark(args),
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

    let stats = backend.generate(request, &mut |piece| {
        print!("{piece}");
        use std::io::Write;
        std::io::stdout().flush()?;
        Ok(())
    })?;
    println!();

    println!("\nInference Statistics:");
    println!("  Prompt tokens:     {}", stats.prompt_tokens);
    println!("  Completion tokens: {}", stats.completion_tokens);
    println!("  Total tokens:      {}", stats.total_tokens);
    if let Some(ttft) = stats.time_to_first_token_ms {
        println!("  Time to First Token (TTFT): {} ms", ttft);
    }
    if let Some(total_time) = stats.generation_time_ms {
        println!("  Total generation time:      {} ms", total_time);
        if stats.completion_tokens > 0 {
            let tokens_per_sec = (stats.completion_tokens as f64) / (total_time as f64 / 1000.0);
            println!("  Generation speed:           {:.2} tokens/sec", tokens_per_sec);
        }
    }
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

struct BackendBenchResult {
    backend_name: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    avg_ttft: Option<f64>,
    avg_total_time: f64,
    avg_speed: Option<f64>,
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    if args.runs == 0 {
        return Err(anyhow!("Number of runs must be greater than 0"));
    }

    let config = load_config_file(&args.config)?;
    
    // Determine which backends to benchmark
    let backends_to_bench: Vec<String> = if let Some(ref b) = args.backend {
        vec![b.clone()]
    } else {
        // Benchmark all backends defined in the config
        config.backends.keys().cloned().collect()
    };

    if backends_to_bench.is_empty() {
        return Err(anyhow!("No backends found to benchmark"));
    }

    println!("Starting benchmarks (trials per backend: {})...", args.runs);
    println!("Prompt: {:?}", args.prompt);
    println!();

    let mut results = Vec::new();

    for backend_name in &backends_to_bench {
        println!("Benchmarking backend: {}...", backend_name);
        
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut ttft_samples = Vec::new();
        let mut total_time_samples = Vec::new();
        let mut speed_samples = Vec::new();
        let mut failed = false;

        for run_idx in 1..=args.runs {
            print!("  Run {}/{}... ", run_idx, args.runs);
            use std::io::Write;
            std::io::stdout().flush()?;

            let mut backend = match create_backend_for_cli(&config, Some(backend_name)) {
                Ok(b) => b,
                Err(e) => {
                    println!("Failed to initialize backend: {e}");
                    failed = true;
                    break;
                }
            };

            let request = GenerateRequest {
                messages: vec![ChatMessage::new("user", args.prompt.clone())],
                max_tokens: None,
                temperature: None,
                top_p: None,
            };

            // Capture output during benchmark run, but discard it to keep stdout clean
            let run_result = backend.generate(request, &mut |_| Ok(()));

            match run_result {
                Ok(stats) => {
                    prompt_tokens = stats.prompt_tokens;
                    completion_tokens = stats.completion_tokens;
                    
                    if let Some(ttft) = stats.time_to_first_token_ms {
                        ttft_samples.push(ttft as f64);
                    }
                    if let Some(total_time) = stats.generation_time_ms {
                        total_time_samples.push(total_time as f64);
                        if stats.completion_tokens > 0 {
                            let speed = (stats.completion_tokens as f64) / (total_time as f64 / 1000.0);
                            speed_samples.push(speed);
                        }
                    }
                    println!("OK");
                }
                Err(e) => {
                    println!("FAILED: {e}");
                    failed = true;
                    break;
                }
            }
        }

        if !failed && !total_time_samples.is_empty() {
            let avg_ttft = if !ttft_samples.is_empty() {
                Some(ttft_samples.iter().sum::<f64>() / ttft_samples.len() as f64)
            } else {
                None
            };
            let avg_total_time = total_time_samples.iter().sum::<f64>() / total_time_samples.len() as f64;
            let avg_speed = if !speed_samples.is_empty() {
                Some(speed_samples.iter().sum::<f64>() / speed_samples.len() as f64)
            } else {
                None
            };

            results.push(BackendBenchResult {
                backend_name: backend_name.clone(),
                prompt_tokens,
                completion_tokens,
                avg_ttft,
                avg_total_time,
                avg_speed,
            });
        }
    }

    // Print comparison table
    println!("\n=== Benchmark Results ===");
    println!("| Backend | Prompt Tokens | Gen Tokens | Avg TTFT (ms) | Avg Total Time (ms) | Avg Speed (tokens/s) |");
    println!("|---|---|---|---|---|---|");
    for res in &results {
        let ttft_str = res.avg_ttft.map_or("N/A".to_string(), |v| format!("{:.1}", v));
        let speed_str = res.avg_speed.map_or("N/A".to_string(), |v| format!("{:.2}", v));
        println!(
            "| {} | {} | {} | {} | {:.1} | {} |",
            res.backend_name,
            res.prompt_tokens,
            res.completion_tokens,
            ttft_str,
            res.avg_total_time,
            speed_str
        );
    }
    println!();

    Ok(())
}
