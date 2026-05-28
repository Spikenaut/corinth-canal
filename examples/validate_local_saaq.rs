//! Dry-run validation for the local SAAQ experiment matrix.
//!
//! Usage:
//!   cargo run --example validate_local_saaq --no-default-features -- <matrix.toml> [--check-paths]
//!
//! Validates matrix entries for:
//! - Structural correctness (RunMatrix::validate)
//! - Path existence (when --check-paths is given)
//! - Model family/format consistency
//!
//! Exits 0 on success, 1 on validation failure.

use corinth_canal::RunMatrix;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: validate_local_saaq <matrix.toml> [--check-paths]");
        process::exit(1);
    }

    let matrix_path = &args[1];
    let check_paths = args.iter().any(|a| a == "--check-paths");

    let contents = match fs::read_to_string(matrix_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading '{}': {}", matrix_path, e);
            process::exit(1);
        }
    };

    let matrix: RunMatrix = match toml::from_str(&contents) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error parsing '{}': {}", matrix_path, e);
            process::exit(1);
        }
    };

    println!("Loaded {} run(s) from '{}'", matrix.runs.len(), matrix_path);

    if let Err(e) = matrix.validate() {
        eprintln!("Validation failed: {}", e);
        process::exit(1);
    }

    let mut warnings = Vec::new();
    let mut path_errors = 0;
    let mut enabled = 0;
    let mut skipped = 0;

    for run in &matrix.runs {
        if run.skip_reason.is_some() {
            skipped += 1;
            continue;
        }
        enabled += 1;

        if !["gguf", "safetensors", "custom_artifact"].contains(&run.source_format.as_str()) {
            warnings.push(format!(
                "run '{}': unknown source_format '{}' (expected gguf|safetensors|custom_artifact)",
                run.run_id, run.source_format
            ));
        }

        if !["protect", "quantize", "drop"].contains(&run.router_policy.as_str()) {
            warnings.push(format!(
                "run '{}': unexpected router_policy '{}' (expected protect|quantize|drop)",
                run.run_id, run.router_policy
            ));
        }

        if !["protect", "quantize", "drop"].contains(&run.norm_policy.as_str()) {
            warnings.push(format!(
                "run '{}': unexpected norm_policy '{}' (expected protect|quantize|drop)",
                run.run_id, run.norm_policy
            ));
        }

        if check_paths {
            let model_path = Path::new(&run.model_id_or_path);
            if !model_path.exists() && run.source_format != "custom_artifact" {
                warnings.push(format!(
                    "run '{}': model path not found: {}",
                    run.run_id, run.model_id_or_path
                ));
                path_errors += 1;
            }

            if let Some(env_val) = run.model_id_or_path.strip_prefix("$") {
                if let Some(var_name) = env_val.split('/').next() {
                    if std::env::var(var_name).is_err() {
                        warnings.push(format!(
                            "run '{}': env var '{}' not set (from model_id_or_path '{}')",
                            run.run_id, var_name, run.model_id_or_path
                        ));
                        path_errors += 1;
                    }
                }
            }

            let output_root = Path::new(&run.output_root);
            if output_root.exists() {
                let run_dir = output_root.join(&run.run_id);
                if run_dir.exists() {
                    warnings.push(format!(
                        "run '{}': output directory already exists: {}",
                        run.run_id,
                        run_dir.display()
                    ));
                }
            }
        }
    }

    println!("Validation passed.");
    println!("  Enabled runs: {}", enabled);
    println!("  Skipped runs: {}", skipped);

    if !warnings.is_empty() {
        println!();
        println!("Warnings ({}):", warnings.len());
        for w in &warnings {
            println!("  - {}", w);
        }
    }

    if path_errors > 0 {
        eprintln!(
            "Path check found {} missing or unresolved path(s).",
            path_errors
        );
        process::exit(1);
    }
}
