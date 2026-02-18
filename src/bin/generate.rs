//! strata-generate: Generate text from a GGUF causal language model.

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use strata_inference::cli;
use strata_inference::engine::generate::GenerationConfig;
use strata_inference::engine::sampler::SamplingConfig;
use strata_inference::GenerationEngine;

#[derive(Parser)]
#[command(name = "strata-generate", about = "Generate text from a GGUF model")]
struct Args {
    /// Path to GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Prompt text
    #[arg(short = 'p', long, conflicts_with = "file")]
    prompt: Option<String>,

    /// Read prompt from file
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// Maximum tokens to generate (-1 = until EOS or context limit)
    #[arg(short = 'n', long, default_value = "-1")]
    max_tokens: i64,

    /// Temperature (0.0 = greedy)
    #[arg(long, default_value = "0.8")]
    temp: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "40")]
    top_k: usize,

    /// Top-p (nucleus) sampling
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Random seed for sampling
    #[arg(short = 's', long)]
    seed: Option<u64>,

    /// Don't echo the prompt in output
    #[arg(long)]
    no_display_prompt: bool,

    /// Output format: text or json
    #[arg(long, default_value = "text", value_parser = validate_output_format)]
    output_format: String,

    /// Compute backend: auto, cpu, metal, cuda
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Suppress all logging
    #[arg(long)]
    log_disable: bool,

    /// Enable per-token profiling (prints to stderr)
    #[arg(long)]
    profile: bool,
}

fn validate_output_format(s: &str) -> Result<String, String> {
    match s {
        "text" | "json" => Ok(s.to_string()),
        _ => Err(format!("Unknown output format '{}'. Options: text, json", s)),
    }
}

#[derive(Serialize)]
struct Timings {
    load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    prefill_tok_per_sec: f64,
    decode_tok_per_sec: f64,
}

#[derive(Serialize)]
struct ConfigOutput {
    max_tokens: i64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct JsonOutput {
    model: String,
    prompt: String,
    output: String,
    generated_tokens: usize,
    stop_reason: String,
    timings: Timings,
    config: ConfigOutput,
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
    if args.profile {
        unsafe { std::env::set_var("STRATA_PROFILE", "1"); }
    }
    let input = cli::read_input(args.prompt.as_deref(), args.file.as_deref(), false)?;

    let total_start = Instant::now();

    // Load model
    let load_start = Instant::now();
    let backend = cli::backend::resolve_backend(Some(&args.backend))?;
    let engine = GenerationEngine::from_gguf(&args.model, backend)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Build generation config
    // -1 means generate until EOS or context limit; use a large number
    let max_tokens = if args.max_tokens < 0 {
        usize::MAX
    } else {
        args.max_tokens as usize
    };

    let gen_config = GenerationConfig {
        max_tokens,
        stop_tokens: Vec::new(),
        sampling: SamplingConfig {
            temperature: args.temp,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
        },
    };

    // Generate with timing
    let decode_start_cell: std::cell::Cell<Option<Instant>> = std::cell::Cell::new(None);

    let output = engine.generate_stream_full(&input, &gen_config, |_token_id| {
        if decode_start_cell.get().is_none() {
            decode_start_cell.set(Some(Instant::now()));
        }
        true
    })?;

    let gen_end = Instant::now();

    // Timing calculations — prefill comes from the engine (excludes tokenization)
    let prefill_ms = output.prefill_duration.as_secs_f64() * 1000.0;

    let decode_ms = decode_start_cell
        .get()
        .map(|t| (gen_end - t).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = output.token_ids.len();
    let generated_text = engine.decode(&output.token_ids);

    let prefill_tok_per_sec = if prefill_ms > 0.0 {
        (output.prompt_tokens as f64) / (prefill_ms / 1000.0)
    } else {
        0.0
    };

    let decode_tok_per_sec = if decode_ms > 0.0 {
        (generated_tokens as f64) / (decode_ms / 1000.0)
    } else {
        0.0
    };

    match args.output_format.as_str() {
        "json" => {
            let json = JsonOutput {
                model: args.model.display().to_string(),
                prompt: input.clone(),
                output: generated_text,
                generated_tokens,
                stop_reason: output.stop_reason.to_string(),
                timings: Timings {
                    load_ms,
                    prefill_ms,
                    decode_ms,
                    total_ms,
                    prefill_tok_per_sec,
                    decode_tok_per_sec,
                },
                config: ConfigOutput {
                    max_tokens: args.max_tokens,
                    temperature: args.temp,
                    top_k: args.top_k,
                    top_p: args.top_p,
                    seed: args.seed,
                },
            };
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            if !args.no_display_prompt {
                print!("{}", input);
            }
            println!("{}", generated_text);
        }
    }

    Ok(())
}
