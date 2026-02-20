//! strata-metal-generate: Generate text using the graph-based Metal fast path.

use std::process;
use std::time::Instant;

use clap::Parser;

use strata_inference::cli;
use strata_inference::engine::generate::GenerationConfig;
use strata_inference::engine::sampler::SamplingConfig;
use strata_inference::GenerationEngine;

#[derive(Parser)]
#[command(name = "strata-metal-generate", about = "Generate text via Metal graph engine")]
struct Args {
    /// Model name or path to GGUF file (e.g., "tinyllama" or "./model.gguf")
    #[arg(short = 'm', long)]
    model: String,

    /// Prompt text
    #[arg(short = 'p', long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value = "32")]
    max_tokens: usize,

    /// Temperature (0.0 = greedy)
    #[arg(long, default_value = "0.0")]
    temp: f32,

    /// Random seed for sampling
    #[arg(short = 's', long, default_value = "42")]
    seed: u64,

    /// Enable per-token profiling
    #[arg(long)]
    profile: bool,

    /// Debug: dump first N logits from graph decode
    #[arg(long, default_value = "0")]
    debug_logits: usize,

    /// Context size (KV cache length). Defaults to min(model context_length, 4096).
    #[arg(long)]
    ctx: Option<usize>,
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.profile {
        unsafe { std::env::set_var("STRATA_PROFILE", "1"); }
    }

    let model_path = cli::model::resolve_model(&args.model)?;
    let total_start = Instant::now();

    // Load model via unified GenerationEngine with Metal backend
    let load_start = Instant::now();
    let mut engine = GenerationEngine::from_gguf_with_options(&model_path, "metal", args.ctx)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[timing] model load: {:.1}ms", load_ms);

    // Print weight dtypes for debugging
    eprintln!("[debug] config: {}", engine.config().arch_name);
    eprintln!("[debug] vocab_size: {}", engine.config().vocab_size);

    let gen_config = GenerationConfig {
        max_tokens: args.max_tokens,
        stop_tokens: Vec::new(),
        sampling: SamplingConfig {
            temperature: args.temp,
            top_k: 40,
            top_p: 0.95,
            seed: Some(args.seed),
        },
    };

    let output = engine.generate_stream_full(&args.prompt, &gen_config, |tok| {
        eprint!("[tok={}] ", tok);
        true
    })?;
    eprintln!();

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = output.prefill_duration.as_secs_f64() * 1000.0;
    let decode_ms = total_ms - load_ms - prefill_ms;
    let n = output.token_ids.len();

    eprintln!("\n[summary]");
    eprintln!("  prompt tokens: {}", output.prompt_tokens);
    eprintln!("  generated tokens: {}", n);
    eprintln!("  stop reason: {}", output.stop_reason);
    eprintln!("  load: {:.1}ms", load_ms);
    eprintln!("  prefill: {:.1}ms ({:.1} tok/s)", prefill_ms, output.prompt_tokens as f64 / (prefill_ms / 1000.0));
    if n > 0 {
        eprintln!("  decode: {:.1}ms ({:.1} ms/tok, {:.1} tok/s)", decode_ms, decode_ms / n as f64, n as f64 / (decode_ms / 1000.0));
    }
    eprintln!("  total: {:.1}ms", total_ms);

    // Print token IDs
    eprintln!("  token_ids: {:?}", &output.token_ids);

    // Print decoded text
    let text = engine.decode(&output.token_ids);
    println!("{}{}", args.prompt, text);

    Ok(())
}
