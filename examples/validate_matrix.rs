//! Validate a SAAQ run matrix TOML file.
//!
//! Usage:
//!   cargo run --example validate_matrix --no-default-features -- <path/to/matrix.toml>
//!
//! Exits with code 0 on success, 1 on validation failure.

use corinth_canal::RunMatrix;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: validate_matrix <path/to/matrix.toml>");
        process::exit(1);
    }

    let path = &args[1];
    let contents = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading '{}': {}", path, e);
            process::exit(1);
        }
    };

    #[derive(serde::Deserialize)]
    struct MatrixFile {
        #[serde(rename = "run")]
        runs: Vec<corinth_canal::RunEntry>,
    }

    let file: MatrixFile = match toml::from_str(&contents) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error parsing '{}': {}", path, e);
            process::exit(1);
        }
    };

    let matrix = RunMatrix { runs: file.runs };

    println!("Loaded {} run(s) from '{}'", matrix.runs.len(), path);

    if let Err(e) = matrix.validate() {
        eprintln!("Validation failed: {}", e);
        process::exit(1);
    }

    println!("Validation passed.");

    // Summary
    let enabled = matrix
        .runs
        .iter()
        .filter(|r| r.skip_reason.is_none())
        .count();
    let skipped = matrix.runs.len() - enabled;
    println!("  Enabled runs: {}", enabled);
    println!("  Skipped runs: {}", skipped);

    if skipped > 0 {
        for run in &matrix.runs {
            if let Some(reason) = &run.skip_reason {
                println!("    - {}: {}", run.run_id, reason);
            }
        }
    }
}
