//! strata-embed: Generate embeddings from a GGUF embedding model.

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use strata_inference::cli;
use strata_inference::EmbeddingEngine;

#[derive(Parser)]
#[command(name = "strata-embed", about = "Generate embeddings from a GGUF model")]
struct Args {
    /// Model name or path to GGUF file (e.g., "miniLM" or "./model.gguf")
    #[arg(short = 'm', long)]
    model: String,

    /// Text to embed
    #[arg(short = 'p', long, conflicts_with = "file")]
    prompt: Option<String>,

    /// Read text from file
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// Normalization: 2=L2 (only L2 currently supported)
    #[arg(long, default_value = "2", value_parser = validate_normalize)]
    embd_normalize: i32,

    /// Output format: json, array, or raw
    #[arg(long, default_value = "json", value_parser = validate_embd_format)]
    embd_output_format: String,

    /// Separator for multiple inputs
    #[arg(long, default_value = "\n")]
    embd_separator: String,

    /// Compute backend: auto, cpu, metal, cuda
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Suppress all logging
    #[arg(long)]
    log_disable: bool,
}

fn validate_normalize(s: &str) -> Result<i32, String> {
    let n: i32 = s.parse().map_err(|_| format!("'{}' is not a valid integer", s))?;
    match n {
        2 => Ok(n),
        _ => Err(format!(
            "Unsupported normalization '{}'. Only 2 (L2) is currently supported",
            n
        )),
    }
}

fn validate_embd_format(s: &str) -> Result<String, String> {
    match s {
        "json" | "array" | "raw" => Ok(s.to_string()),
        _ => Err(format!(
            "Unknown output format '{}'. Options: json, array, raw",
            s
        )),
    }
}

#[derive(Serialize)]
struct Timings {
    load_ms: f64,
    inference_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct JsonOutput {
    model: String,
    embeddings: Vec<Vec<f32>>,
    embedding_dim: usize,
    num_embeddings: usize,
    timings: Timings,
}

fn main() {
    let args = Args::parse();
    cli::init_logging(args.log_disable);

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = cli::model::resolve_model(&args.model)?;
    let input = cli::read_input(args.prompt.as_deref(), args.file.as_deref(), false)?;

    let total_start = Instant::now();

    // Load model
    let load_start = Instant::now();
    let backend = cli::backend::resolve_backend(Some(&args.backend))?;
    let engine = EmbeddingEngine::from_gguf(&model_path, backend)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Split input by separator
    let texts: Vec<&str> = input
        .split(&args.embd_separator)
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();

    if texts.is_empty() {
        return Err("No non-empty inputs found".into());
    }

    // Embed
    let infer_start = Instant::now();
    let embeddings = engine.embed_batch(&texts)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let embedding_dim = engine.embedding_dim();

    match args.embd_output_format.as_str() {
        "json" => {
            let output = JsonOutput {
                model: args.model.clone(),
                num_embeddings: embeddings.len(),
                embedding_dim,
                embeddings,
                timings: Timings {
                    load_ms,
                    inference_ms,
                    total_ms,
                },
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "array" => {
            println!("{}", serde_json::to_string(&embeddings)?);
        }
        "raw" => {
            for emb in &embeddings {
                let vals: Vec<String> = emb.iter().map(|v| format!("{}", v)).collect();
                println!("{}", vals.join(" "));
            }
        }
        _ => unreachable!(), // validated by clap
    }

    Ok(())
}
