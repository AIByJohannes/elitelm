use std::io::{stdin, stdout, Write};
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use elitelm_backend_genie::{GenieBackend, prepare_genie_bundle};
use elitelm_backend_llamacpp::LlamaCppBackend;
use elitelm_core::{
    BackendConfig, ChatMessage, GenerateRequest, InferenceBackend, create_fake_backend,
    get_elitelm_models_dir, load_config_file, model_filename,
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
    /// Run a model and start an interactive chat session or generate a response
    Run(RunArgs),
    /// Start the OpenAI-compatible API server
    Serve(ServeArgs),
    /// List available models configured in the config file
    #[command(visible_alias = "ls")]
    List(ListArgs),
    /// Show detailed information about a model/backend
    Show(ShowArgs),
    /// Pull a model from the registry
    Pull(PullArgs),
    /// Remove a downloaded model
    Rm(RmArgs),
    /// Prepare a bundle for Genie NPU backend
    #[command(name = "prepare-genie-bundle")]
    PrepareGenieBundle(PrepareGenieBundleArgs),
    /// Benchmark configured backends
    Benchmark(BenchmarkArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    /// The name of the backend (model) to run (positional).
    backend_pos: Option<String>,
    /// The prompt to send (positional).
    prompt_pos: Option<String>,
    /// Override/specify the backend to run as a flag.
    #[arg(long, short = 'b')]
    backend: Option<String>,
    /// Override/specify the prompt to send as a flag.
    #[arg(long, short = 'p')]
    prompt: Option<String>,
    /// Path to the configuration file.
    #[arg(long, short, default_value = "elitelm.example.yaml")]
    config: PathBuf,
}

#[derive(Debug, Parser)]
struct ServeArgs {
    /// Path to the configuration file.
    #[arg(long, short, default_value = "elitelm.example.yaml")]
    config: PathBuf,
    /// Override default backend to use.
    #[arg(long)]
    backend: Option<String>,
    /// Host interface to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

#[derive(Debug, Parser)]
struct ListArgs {
    /// Path to the configuration file.
    #[arg(long, short, default_value = "elitelm.example.yaml")]
    config: PathBuf,
}

#[derive(Debug, Parser)]
struct ShowArgs {
    /// The name of the backend (model) to show.
    backend: String,
    /// Path to the configuration file.
    #[arg(long, short, default_value = "elitelm.example.yaml")]
    config: PathBuf,
}

#[derive(Debug, Parser)]
struct PullArgs {
    /// The name of the model to pull (e.g. qwen2.5:0.5b).
    model: String,
}

#[derive(Debug, Parser)]
struct RmArgs {
    /// The name of the model to remove.
    model: String,
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

#[derive(serde::Deserialize)]
struct RegistryManifest {
    layers: Vec<RegistryLayer>,
}

#[derive(serde::Deserialize)]
struct RegistryLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args).await,
        Command::Serve(args) => serve(args).await,
        Command::List(args) => list_models(args),
        Command::Show(args) => show_model(args),
        Command::Pull(args) => pull_cmd(args).await,
        Command::Rm(args) => remove_model(args),
        Command::PrepareGenieBundle(args) => prepare(args),
        Command::Benchmark(args) => benchmark(args),
    }
}

async fn run(args: RunArgs) -> Result<()> {
    let backend_name_opt = args.backend.or(args.backend_pos);
    let prompt = args.prompt.or(args.prompt_pos);
    let config = load_config_file(&args.config)?;

    // Determine the backend name
    let backend_name = if let Some(ref name) = backend_name_opt {
        if config.backends.contains_key(name) {
            config.resolve_backend_name(Some(name))?.to_string()
        } else {
            // It's a registry/downloaded model
            let models_dir = get_elitelm_models_dir();
            let filename = model_filename(name);
            let local_path = models_dir.join(&filename);
            if !local_path.exists() {
                println!("Model '{}' not found locally.", name);
                pull_model(name).await?;
            }
            name.clone()
        }
    } else {
        config.resolve_backend_name(None)?.to_string()
    };

    let backend = create_backend_for_cli(&config, Some(&backend_name))?;

    if let Some(prompt) = prompt {
        let mut backend = backend;
        let request = GenerateRequest {
            messages: vec![ChatMessage::new("user", prompt)],
            max_tokens: None,
            temperature: None,
            top_p: None,
        };

        let stats = backend.generate(request, &mut |piece| {
            print!("{piece}");
            stdout().flush()?;
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
    } else {
        run_interactive(backend, &backend_name)?;
    }
    Ok(())
}

fn run_interactive(mut backend: Box<dyn InferenceBackend>, backend_name: &str) -> Result<()> {
    println!("EliteLM Interactive Chat Session (Backend: {backend_name})");
    println!("Type /bye or /exit to quit, /help for help.");
    println!();

    let mut history = Vec::new();

    loop {
        print!(">>> ");
        stdout().flush()?;

        let mut input = String::new();
        let bytes_read = stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            // EOF
            println!("/bye");
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "/bye" || trimmed == "/exit" || trimmed == "/quit" {
            println!("Goodbye!");
            break;
        }

        if trimmed == "/help" {
            println!("Available commands:");
            println!("  /help - Show this help message");
            println!("  /bye, /exit, /quit - Exit the chat session");
            println!();
            continue;
        }

        history.push(ChatMessage::new("user", trimmed.to_string()));

        let request = GenerateRequest {
            messages: history.clone(),
            max_tokens: None,
            temperature: None,
            top_p: None,
        };

        let mut assistant_reply = String::new();
        // Print response start
        let stats_res = backend.generate(request, &mut |piece| {
            print!("{piece}");
            stdout().flush()?;
            assistant_reply.push_str(piece);
            Ok(())
        });
        println!(); // ensure trailing newline

        match stats_res {
            Ok(stats) => {
                history.push(ChatMessage::new("assistant", assistant_reply));
                
                // Print a small status line
                if let Some(total_time) = stats.generation_time_ms {
                    if stats.completion_tokens > 0 {
                        let speed = (stats.completion_tokens as f64) / (total_time as f64 / 1000.0);
                        println!("[Speed: {:.2} tokens/sec, Prompt tokens: {}, Gen tokens: {}]", speed, stats.prompt_tokens, stats.completion_tokens);
                    }
                }
                println!();
            }
            Err(e) => {
                println!("Error generating response: {e}");
                // Remove the user message from history as it failed
                history.pop();
            }
        }
    }

    Ok(())
}

async fn serve(args: ServeArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    let app = elitelm_server::build_router(config, args.backend);
    let address: std::net::SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;

    println!("EliteLM server listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn list_models(args: ListArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    println!("{:<25} {:<15} {}", "NAME", "KIND", "DETAILS");
    println!("{}", "-".repeat(70));
    
    for (name, backend) in &config.backends {
        let details = match backend {
            BackendConfig::Fake(_) => "-".to_string(),
            BackendConfig::Genie(g) => format!("bundle_dir: {}", g.bundle_dir.display()),
            BackendConfig::LlamaCpp(l) => format!("model: {}", l.model.display()),
        };
        println!("{:<25} {:<15} {}", name, backend.kind(), details);
    }

    // Include downloaded registry models
    let models_dir = get_elitelm_models_dir();
    if models_dir.exists() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "gguf") {
                if let Some(stem) = path.file_stem() {
                    let filename = stem.to_string_lossy().to_string();
                    let reconstructed = if let Some(idx) = filename.rfind('_') {
                        let mut r = filename.clone();
                        r.replace_range(idx..=idx, ":");
                        r
                    } else {
                        filename.clone()
                    };

                    if !config.backends.contains_key(&reconstructed) {
                        let size_bytes = entry.metadata()?.len();
                        let size_mb = size_bytes as f64 / 1024.0 / 1024.0;
                        println!("{:<25} {:<15} registry (size: {:.2} MB)", reconstructed, "llamacpp", size_mb);
                    }
                }
            }
        }
    }

    Ok(())
}

fn show_model(args: ShowArgs) -> Result<()> {
    let config = load_config_file(&args.config)?;
    if config.backends.contains_key(&args.backend) {
        let (name, backend) = config.backend(Some(&args.backend))?;
        println!("Model: {}", name);
        println!("Kind:  {}", backend.kind());
        match backend {
            BackendConfig::Fake(fake) => {
                println!("Response prefix: {:?}", fake.response_prefix.as_deref().unwrap_or("None"));
            }
            BackendConfig::Genie(genie) => {
                println!("Bundle directory:   {}", genie.bundle_dir.display());
                println!("Genie config:       {}", genie.genie_config.display());
                println!("HTP config:         {}", genie.htp_config.display());
                println!("QNN SDK root:       {}", genie.qnn_sdk_root.display());
                println!("Tokenizer path:     {}", genie.tokenizer_path.display());
                println!("Genie template:     {}", genie.genie_config_template.display());
                println!("HTP template:       {}", genie.htp_config_template.display());
                println!("SoC model:          {}", genie.soc_model);
                println!("DSP arch:           {}", genie.dsp_arch);
                if let Some(ref exec) = genie.genie_executable {
                    println!("Genie executable:   {}", exec.display());
                }
            }
            BackendConfig::LlamaCpp(llamacpp) => {
                println!("Model path:         {}", llamacpp.model.display());
                if let Some(threads) = llamacpp.n_threads {
                    println!("Threads:            {}", threads);
                }
                if let Some(ctx) = llamacpp.n_ctx {
                    println!("Context size:       {}", ctx);
                }
                if let Some(batch) = llamacpp.n_batch {
                    println!("Batch size:         {}", batch);
                }
                println!("Use mmap:           {}", llamacpp.use_mmap);
            }
        }
    } else {
        // Check if registry model
        let models_dir = get_elitelm_models_dir();
        let filename = model_filename(&args.backend);
        let local_path = models_dir.join(&filename);
        if local_path.exists() {
            let metadata = local_path.metadata()?;
            println!("Model: {}", args.backend);
            println!("Kind:  llamacpp (registry)");
            println!("Path:  {}", local_path.display());
            println!("Size:  {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
        } else {
            return Err(anyhow!("Model '{}' not found", args.backend));
        }
    }
    Ok(())
}

async fn pull_cmd(args: PullArgs) -> Result<()> {
    pull_model(&args.model).await
}

fn remove_model(args: RmArgs) -> Result<()> {
    let models_dir = get_elitelm_models_dir();
    let filename = model_filename(&args.model);
    let local_path = models_dir.join(&filename);
    if local_path.exists() {
        std::fs::remove_file(&local_path)?;
        println!("Removed model '{}'", args.model);
    } else {
        println!("Model '{}' is not downloaded locally", args.model);
    }
    Ok(())
}

fn get_manifest_url(model: &str, tag: &str) -> String {
    if model.contains('/') {
        format!("https://registry.ollama.ai/v2/{}/manifests/{}", model, tag)
    } else {
        format!("https://registry.ollama.ai/v2/library/{}/manifests/{}", model, tag)
    }
}

fn get_blob_url(model: &str, digest: &str) -> String {
    if model.contains('/') {
        format!("https://registry.ollama.ai/v2/{}/blobs/{}", model, digest)
    } else {
        format!("https://registry.ollama.ai/v2/library/{}/blobs/{}", model, digest)
    }
}

async fn pull_model(model_name: &str) -> Result<()> {
    let (name, tag) = if let Some(idx) = model_name.find(':') {
        (&model_name[..idx], &model_name[idx+1..])
    } else {
        (model_name, "latest")
    };

    println!("Pulling manifest for {}:{}...", name, tag);

    let client = reqwest::Client::new();
    let manifest_url = get_manifest_url(name, tag);
    
    let resp = client.get(&manifest_url)
        .header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(anyhow!("Failed to pull manifest for {}:{}. Status: {}", name, tag, resp.status()));
    }

    let manifest: RegistryManifest = resp.json().await?;

    // Find the model weights layer
    let model_layer = manifest.layers.iter()
        .find(|layer| layer.media_type == "application/vnd.ollama.image.model")
        .ok_or_else(|| anyhow!("No model weights layer found in manifest"))?;

    println!("Found model weights layer:");
    println!("  Digest: {}", model_layer.digest);
    println!("  Size:   {:.2} MB", model_layer.size as f64 / 1024.0 / 1024.0);

    // Determine where to save it
    let models_dir = get_elitelm_models_dir();
    tokio::fs::create_dir_all(&models_dir).await?;
    
    let dest_filename = model_filename(model_name);
    let dest_path = models_dir.join(&dest_filename);

    println!("Downloading to {}...", dest_path.display());

    // Fetch the blob
    let mut blob_resp = client.get(&get_blob_url(name, &model_layer.digest))
        .send()
        .await?;

    if !blob_resp.status().is_success() {
        return Err(anyhow!("Failed to download model blob. Status: {}", blob_resp.status()));
    }

    // Write to file with progress
    let mut dest_file = tokio::fs::File::create(&dest_path).await?;
    let mut downloaded = 0u64;
    let total_size = model_layer.size;
    
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = blob_resp.chunk().await? {
        dest_file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        let percent = (downloaded as f64 / total_size as f64) * 100.0;
        let downloaded_mb = downloaded as f64 / 1024.0 / 1024.0;
        let total_mb = total_size as f64 / 1024.0 / 1024.0;
        
        print!("\rDownloading: {:.2}% ({:.2} MB / {:.2} MB)", percent, downloaded_mb, total_mb);
        stdout().flush()?;
    }
    println!("\nDownload complete!");

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
    if let Some(name) = requested_backend {
        if config.backends.contains_key(name) {
            // Exist in config file, load as configured
            let (_, backend_config) = config.backend(Some(name))?;
            return build_backend_from_config(name, backend_config);
        }

        // Not in config file. Check if it exists as a pulled GGUF model.
        let models_dir = get_elitelm_models_dir();
        let filename = model_filename(name);
        let local_path = models_dir.join(&filename);
        if local_path.exists() {
            let llama_config = elitelm_core::LlamaCppBackendConfig {
                model: local_path,
                n_threads: None,
                n_ctx: None,
                n_batch: None,
                use_mmap: true,
            };
            return Ok(Box::new(LlamaCppBackend::new(
                name.to_string(),
                llama_config,
            )?));
        }
    }

    let (name, backend_config) = config.backend(requested_backend)?;
    build_backend_from_config(name, backend_config)
}

fn build_backend_from_config(
    name: &str,
    backend_config: &BackendConfig,
) -> Result<Box<dyn InferenceBackend>> {
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
