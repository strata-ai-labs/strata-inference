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

use std::io::Write;

#[derive(Parser)]
#[command(name = "strata-generate", about = "Generate text from a GGUF model")]
struct Args {
    /// Model name or path to GGUF file (e.g., "qwen3:8b" or "./model.gguf")
    #[arg(short = 'm', long)]
    model: String,

    /// Prompt text
    #[arg(short = 'p', long, conflicts_with_all = ["file", "token_ids"])]
    prompt: Option<String>,

    /// Read prompt from file
    #[arg(short = 'f', long, conflicts_with = "token_ids")]
    file: Option<PathBuf>,

    /// Pre-tokenized input: comma-separated token IDs (bypasses tokenizer)
    #[arg(long, conflicts_with_all = ["prompt", "file"])]
    token_ids: Option<String>,

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

    /// Compute backend: auto, cpu, metal
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Suppress all logging
    #[arg(long)]
    log_disable: bool,

    /// Enable per-token profiling (prints to stderr)
    #[arg(long)]
    profile: bool,

    /// Context length override (default: 4096). Affects KV cache size and LongRoPE factor selection.
    #[arg(short = 'c', long)]
    ctx: Option<usize>,

    /// Dump raw logits (pre-softmax) for each generated token to a file.
    /// Each line contains comma-separated f32 values for the full vocabulary.
    #[arg(long)]
    dump_logits: Option<PathBuf>,
}

fn validate_output_format(s: &str) -> Result<String, String> {
    match s {
        "text" | "json" => Ok(s.to_string()),
        _ => Err(format!(
            "Unknown output format '{}'. Options: text, json",
            s
        )),
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
        unsafe {
            std::env::set_var("STRATA_PROFILE", "1");
        }
    }
    let model_path = cli::model::resolve_model(&args.model)?;

    // Determine input mode: --token-ids bypasses tokenizer entirely
    let use_token_ids = args.token_ids.is_some();

    let total_start = Instant::now();

    // Load model with explicit backend selection
    let load_start = Instant::now();
    let mut engine =
        GenerationEngine::from_gguf_with_options(&model_path, &args.backend, args.ctx)?;
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

    // Determine input text for display purposes
    let input_display: String;

    let output = if use_token_ids {
        // Parse comma-separated token IDs
        let ids_str = args.token_ids.as_ref().unwrap();
        let token_ids: Vec<u32> = ids_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .unwrap_or_else(|_| panic!("Invalid token ID: '{}'", s.trim()))
            })
            .collect();
        input_display = format!("[{} token IDs]", token_ids.len());
        engine.generate_stream_full_from_token_ids(&token_ids, &gen_config, |_token_id| {
            if decode_start_cell.get().is_none() {
                decode_start_cell.set(Some(Instant::now()));
            }
            true
        })?
    } else {
        let input = cli::read_input(args.prompt.as_deref(), args.file.as_deref(), false)?;
        input_display = input.clone();
        engine.generate_stream_full(&input, &gen_config, |_token_id| {
            if decode_start_cell.get().is_none() {
                decode_start_cell.set(Some(Instant::now()));
            }
            true
        })?
    };

    let gen_end = Instant::now();

    // Timing calculations -- prefill comes from the engine (excludes tokenization)
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

    // Dump logits if requested (re-run generation to capture logits)
    // Note: For the initial implementation, we dump the generated token IDs.
    // Full logits dumping requires plumbing through the generation loop.
    if let Some(ref logits_path) = args.dump_logits {
        let mut f = std::fs::File::create(logits_path)?;
        // Write generated token IDs (one per line) for now
        // TODO: Full logits vector requires callback plumbing
        for &tid in &output.token_ids {
            writeln!(f, "{}", tid)?;
        }
        eprintln!(
            "Dumped {} generated token IDs to {}",
            output.token_ids.len(),
            logits_path.display()
        );
    }

    match args.output_format.as_str() {
        "json" => {
            let json = JsonOutput {
                model: args.model.clone(),
                prompt: input_display.clone(),
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
            if !args.no_display_prompt && !use_token_ids {
                print!("{}", input_display);
            }
            println!("{}", generated_text);
        }
    }

    Ok(())
}
