//! strata-tokenize: Tokenize text using a GGUF model's vocabulary.

use std::path::PathBuf;
use std::process;

use clap::Parser;
use serde::Serialize;

use strata_inference::cli;
use strata_inference::gguf::GgufFile;
use strata_inference::tokenizer::create_tokenizer_from_gguf;

#[derive(Parser)]
#[command(name = "strata-tokenize", about = "Tokenize text using a GGUF model")]
struct Args {
    /// Model name or path to GGUF file (e.g., "tinyllama" or "./model.gguf")
    #[arg(short = 'm', long)]
    model: String,

    /// Text to tokenize
    #[arg(short = 'p', long, conflicts_with_all = ["file", "stdin"])]
    prompt: Option<String>,

    /// Read text from file
    #[arg(short = 'f', long, conflicts_with = "stdin")]
    file: Option<PathBuf>,

    /// Read text from stdin
    #[arg(long)]
    stdin: bool,

    /// Output only token IDs in list format: [1, 2, 3]
    #[arg(long)]
    ids: bool,

    /// Print total token count
    #[arg(long)]
    show_count: bool,

    /// Don't prepend BOS token
    #[arg(long)]
    no_bos: bool,

    /// Output format: text or json
    #[arg(long, default_value = "text", value_parser = validate_output_format)]
    output_format: String,

    /// Suppress all logging
    #[arg(long)]
    log_disable: bool,
}

fn validate_output_format(s: &str) -> Result<String, String> {
    match s {
        "text" | "json" => Ok(s.to_string()),
        _ => Err(format!("Unknown output format '{}'. Options: text, json", s)),
    }
}

#[derive(Serialize)]
struct TokenEntry {
    id: u32,
    text: String,
}

#[derive(Serialize)]
struct JsonOutput {
    tokens: Vec<TokenEntry>,
    count: usize,
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
    let input = cli::read_input(
        args.prompt.as_deref(),
        args.file.as_deref(),
        args.stdin,
    )?;

    let gguf = GgufFile::open(&model_path)?;
    let tokenizer = create_tokenizer_from_gguf(&gguf)?;

    let add_special_tokens = !args.no_bos;
    let token_ids = tokenizer.encode(&input, add_special_tokens);

    match args.output_format.as_str() {
        "json" => {
            let entries: Vec<TokenEntry> = token_ids
                .iter()
                .map(|&id| TokenEntry {
                    id,
                    text: tokenizer.decode(&[id]),
                })
                .collect();
            let output = JsonOutput {
                count: entries.len(),
                tokens: entries,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            if args.ids {
                let id_strs: Vec<String> = token_ids.iter().map(|id| id.to_string()).collect();
                println!("[{}]", id_strs.join(", "));
            } else {
                for &id in &token_ids {
                    let text = tokenizer.decode(&[id]);
                    println!("{:>5} -> '{}'", id, text);
                }
            }

            if args.show_count {
                println!("Total number of tokens: {}", token_ids.len());
            }
        }
    }

    Ok(())
}
