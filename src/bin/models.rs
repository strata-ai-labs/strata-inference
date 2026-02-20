//! strata-models: Manage the strata model registry.

use std::process;

use clap::{Parser, Subcommand};

use strata_inference::registry::{format_size, ModelRegistry, ModelTask};

#[derive(Parser)]
#[command(name = "strata-models", about = "Manage strata model registry")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List available models
    List {
        /// Only show locally available models
        #[arg(long)]
        local: bool,
    },
    /// Download a model from HuggingFace
    Pull {
        /// Model name (e.g., "qwen3:8b", "miniLM")
        name: String,
    },
    /// Print the local path for a model
    Path {
        /// Model name
        name: String,
    },
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelRegistry::new();

    match args.command {
        Command::List { local } => cmd_list(&registry, local),
        Command::Pull { name } => cmd_pull(&registry, &name),
        Command::Path { name } => cmd_path(&registry, &name),
    }
}

fn cmd_list(registry: &ModelRegistry, local_only: bool) -> Result<(), Box<dyn std::error::Error>> {
    let models = if local_only {
        registry.list_local()
    } else {
        registry.list_available()
    };

    if models.is_empty() {
        if local_only {
            eprintln!("No models downloaded yet.");
            eprintln!("Run `strata-models list` to see available models.");
            eprintln!("Run `strata-models pull <name>` to download one.");
        } else {
            eprintln!("No models in catalog.");
        }
        return Ok(());
    }

    // Print header
    println!(
        "{:<16} {:<10} {:<10} {}",
        "NAME", "TASK", "SIZE", "STATUS"
    );

    for m in &models {
        let task = match m.task {
            ModelTask::Embed => "embed",
            ModelTask::Generate => "generate",
        };
        let size = format_size(m.size_bytes);
        let status = if m.is_local { "downloaded" } else { "-" };

        println!("{:<16} {:<10} {:<10} {}", m.name, task, size, status);
    }

    Ok(())
}

fn cmd_pull(registry: &ModelRegistry, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "registry")]
    {
        use std::io::Write;

        eprintln!("Resolving '{}'...", name);

        let path = registry.pull_with_progress(name, |downloaded, total| {
            if total > 0 {
                let pct = (downloaded as f64 / total as f64 * 100.0) as u64;
                let dl = format_size(downloaded);
                let tot = format_size(total);
                eprint!("\r  {} / {} ({}%)", dl, tot, pct);
                let _ = std::io::stderr().flush();
            }
        })?;

        eprintln!();
        println!("{}", path.display());
        Ok(())
    }

    #[cfg(not(feature = "registry"))]
    {
        let _ = (registry, name);
        Err(
            "Model downloading requires the 'registry' feature. Build with:\n  \
             cargo build --features cli,registry --bin strata-models"
                .into(),
        )
    }
}

fn cmd_path(registry: &ModelRegistry, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = registry.resolve(name)?;
    println!("{}", path.display());
    Ok(())
}
