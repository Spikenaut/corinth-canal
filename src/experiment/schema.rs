// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Standardized experiment output schema for SAAQ quantization runs.
//!
//! This module defines the canonical JSON structures emitted by
//! `saaq_latent_calibration` and consumed by downstream tools such as
//! `Surrogate_Viz.jl`.

use crate::error::{HybridError, Result};
use crate::types::ModelFamily;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Top-level manifest for a single SAAQ experiment run.
///
/// Emitted as `run_manifest.json` inside every run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_tag: Option<String>,
    pub created_at: String,
    pub repo: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_sha: Option<String>,

    pub model_slug: String,
    pub model_family: String,
    pub architecture: String,
    pub checkpoint_path: String,
    pub checkpoint_format: String,
    pub routing_tensor_name: String,
    pub synapse_source: String,

    pub prompt_embedding_source: String,
    pub prompt_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_text: Option<String>,

    pub ticks: usize,
    pub saaq_rule: String,
    pub saaq_primary_rule: String,
    pub saaq_dual_emit: bool,

    pub telemetry_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_csv_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_row_count: Option<usize>,

    pub wraparound_enabled: bool,
    pub wraparound_loops: u64,
    pub ticks_effective: usize,

    pub run_dir: String,
    pub output_root: String,
    pub repeat_idx: usize,
    pub repeat_count: usize,

    pub validation_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_mode: Option<String>,

    /// Projector strategy used for this run (`rate_sum`,
    /// `temporal_histogram`, `membrane_snapshot`, `spiking_ternary`).
    /// Absent on historical manifests written before GH#162.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_mode: Option<String>,

    pub generated_files: Vec<String>,
}

/// Compact, stable per-run summary consumed by downstream aggregators.
///
/// Emitted as `summary.json` inside every run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_tag: Option<String>,
    pub model_slug: String,
    pub model_family: String,
    pub telemetry_source: String,
    pub repeat_idx: usize,
    pub repeat_count: usize,
    pub saaq_rule: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_mode: Option<String>,
    pub validation_status: String,
    pub run_dir: String,
    pub manifest_path: String,
    pub tick_telemetry_path: String,
    pub latent_telemetry_path: String,
    pub metrics: ExperimentMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_determinism: Option<String>,
}

/// Quantitative metrics for a single experiment run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub ticks_completed: usize,
    pub latent_rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_tick_elapsed_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_mse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_output_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_output_rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_top1_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_top2_set_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_load_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_load_js_divergence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_estimate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_gib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_count: Option<usize>,
}

/// Standardized warning record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentWarning {
    pub category: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_name: Option<String>,
}

/// Output bundle emitted by every SAAQ run.
///
/// This is the top-level structure that `Surrogate_Viz.jl` expects to ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentBundle {
    pub manifest: ExperimentManifest,
    pub summary: ExperimentSummary,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<ExperimentWarning>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub metadata: BTreeMap<String, String>,
}

// ── Model adapter configuration ───────────────────────────────────────────

/// Static per-model-family configuration used by the run matrix.
///
/// Emitted as `model_adapter_configs.toml` and consumed by the run matrix
/// validator and `Surrogate_Viz.jl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdapterConfig {
    pub model_family: String,
    pub model_id_or_local_path: String,
    pub format: String,
    pub loader_hint: String,
    pub router_policy: String,
    pub norm_policy: String,
    pub expert_policy: String,
    pub supports_route_metrics: bool,
    pub supports_block_metrics: bool,
    #[serde(default)]
    pub preferred_quant_modes: Vec<String>,
    #[serde(default)]
    pub minimum_expected_artifacts: Vec<String>,
    #[serde(default)]
    pub known_risks: Vec<String>,
}

/// Dynamic per-run entry in the SAAQ run matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEntry {
    pub run_id: String,
    pub model_family: String,
    pub model_id_or_path: String,
    pub source_format: String,
    pub quant_mode: String,
    pub saaq_formula_version: String,
    pub router_policy: String,
    pub norm_policy: String,
    pub telemetry_source: String,
    pub output_root: String,
    pub max_runtime_minutes: u64,
    pub max_disk_gib: u64,
    #[serde(default)]
    pub expected_artifacts: Vec<String>,
    #[serde(default)]
    pub success_metrics: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

/// Full run matrix with validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMatrix {
    #[serde(rename = "run")]
    pub runs: Vec<RunEntry>,
}

impl RunMatrix {
    /// Validate the run matrix.
    ///
    /// Checks:
    /// - Every enabled run has explicit runtime and disk caps.
    /// - Every enabled run has explicit router and norm policy.
    /// - Grok-1 cannot be selected unless `GROK1_ARTIFACT_READY=1`.
    /// - Unknown model families are rejected.
    ///
    /// Reads the Grok-1 readiness flag from the environment. That is the only
    /// environment read in this module, and it is a readiness flag rather than
    /// a filesystem path, per the invariant in CLAUDE.md. Use
    /// [`validate_with`](Self::validate_with) to supply the flag directly.
    pub fn validate(&self) -> Result<()> {
        let grok_ready = std::env::var("GROK1_ARTIFACT_READY").is_ok_and(|v| v == "1");
        self.validate_with(grok_ready)
    }

    /// Validation proper, with the Grok-1 gate passed in.
    ///
    /// Split out so every rejection branch is testable without mutating
    /// process-global environment state from a test.
    pub fn validate_with(&self, grok_ready: bool) -> Result<()> {
        for run in &self.runs {
            // Skip validation for explicitly skipped runs.
            if run.skip_reason.is_some() {
                continue;
            }

            // Runtime and disk caps must be present (non-zero).
            if run.max_runtime_minutes == 0 {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' missing max_runtime_minutes",
                    run.run_id
                )));
            }
            if run.max_disk_gib == 0 {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' missing max_disk_gib",
                    run.run_id
                )));
            }

            // Router and norm policies must be explicit.
            if run.router_policy.trim().is_empty() {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' missing router_policy",
                    run.run_id
                )));
            }
            if run.norm_policy.trim().is_empty() {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' missing norm_policy",
                    run.run_id
                )));
            }

            // Grok-1 gate: only block runs that specifically reference grok-1.
            if run.model_family == "grok" && run.model_id_or_path.contains("grok-1") && !grok_ready
            {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' selects grok-1 but GROK1_ARTIFACT_READY is not set to 1",
                    run.run_id
                )));
            }

            // Validate through the enum rather than a hand-maintained list.
            // This was a third copy of the family names — after
            // `ModelFamily::slug` and `configs/model_adapter_configs.toml` —
            // and a new variant silently failed to reach it.
            //
            // Requiring `slug() == model_family` keeps the accepted set exactly
            // the canonical slugs: `from_alias` also accepts shorthands such as
            // "qwen", which must not be valid in a matrix file.
            let known_family = ModelFamily::from_alias(&run.model_family)
                .is_some_and(|family| family.slug() == run.model_family);
            if !known_family {
                return Err(HybridError::InvalidConfig(format!(
                    "run '{}' uses unknown model_family '{}'",
                    run.run_id, run.model_family
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ExperimentMetrics, ExperimentSummary};

    fn sample_summary(projection_mode: Option<String>) -> ExperimentSummary {
        ExperimentSummary {
            run_id: "t".into(),
            run_tag: None,
            model_slug: "olmoe".into(),
            model_family: "Olmoe".into(),
            telemetry_source: "synthetic".into(),
            repeat_idx: 0,
            repeat_count: 1,
            saaq_rule: "SaaqV1_5SqrtRate".into(),
            projection_mode,
            validation_status: "completed".into(),
            run_dir: "artifacts/x".into(),
            manifest_path: "artifacts/x/run_manifest.json".into(),
            tick_telemetry_path: "artifacts/x/tick_telemetry.txt".into(),
            latent_telemetry_path: "artifacts/x/latent_telemetry.csv".into(),
            metrics: ExperimentMetrics::default(),
            repeat_determinism: None,
        }
    }

    #[test]
    fn experiment_summary_stamps_projection_mode() {
        let summary = sample_summary(Some("rate_sum".into()));
        let json = serde_json::to_value(&summary).expect("serialize summary");
        assert_eq!(json["projection_mode"], "rate_sum");
    }

    #[test]
    fn experiment_summary_missing_projection_mode_deserializes() {
        let summary = sample_summary(Some("spiking_ternary".into()));
        let json = serde_json::to_value(&summary).expect("serialize summary");
        let mut obj = json
            .as_object()
            .cloned()
            .expect("summary serializes as object");
        obj.remove("projection_mode");
        let parsed: ExperimentSummary = serde_json::from_value(serde_json::Value::Object(obj))
            .expect("old summary without field");
        assert_eq!(parsed.projection_mode, None);
    }
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    /// Every string `KNOWN_FAMILIES` accepts must round-trip through the enum.
    /// If this holds, the hand-maintained list is redundant with `ModelFamily`.
    #[test]
    fn known_families_round_trip_through_the_enum() {
        const KNOWN: &[&str] = &[
            "olmoe",
            "qwen3_moe",
            "gemma4",
            "deepseek2",
            "llama_moe",
            "moonlight_16b_a3b",
            "granite_3_1_a800m",
            "nemotron",
            "lfm2_moe",
            "slim_moe",
            "zaya",
            "glm4",
            "gpt_oss",
            "step",
            "minimax",
            "cohere",
            "grin",
            "skyworks",
            "trinity",
            "grok",
        ];
        let mut unmatched = Vec::new();
        for slug in KNOWN {
            match ModelFamily::from_alias(slug) {
                Some(family) if family.slug() == *slug => {}
                Some(family) => {
                    unmatched.push(format!("{slug} -> {:?} (slug {})", family, family.slug()))
                }
                None => unmatched.push(format!("{slug} -> None")),
            }
        }
        assert!(
            unmatched.is_empty(),
            "these do not round-trip: {unmatched:?}"
        );
    }

    fn valid_run() -> RunEntry {
        RunEntry {
            run_id: "r1".into(),
            model_family: "olmoe".into(),
            model_id_or_path: "/models/olmoe".into(),
            source_format: "gguf".into(),
            quant_mode: "q8_0".into(),
            saaq_formula_version: "v1_5".into(),
            router_policy: "top_k".into(),
            norm_policy: "rms".into(),
            telemetry_source: "synthetic".into(),
            output_root: "artifacts".into(),
            max_runtime_minutes: 30,
            max_disk_gib: 8,
            expected_artifacts: Vec::new(),
            success_metrics: Vec::new(),
            skip_reason: None,
        }
    }

    fn matrix(run: RunEntry) -> RunMatrix {
        RunMatrix { runs: vec![run] }
    }

    fn rejection(run: RunEntry, grok_ready: bool) -> String {
        matrix(run)
            .validate_with(grok_ready)
            .expect_err("expected this run to be rejected")
            .to_string()
    }

    #[test]
    fn a_fully_specified_run_validates() {
        assert!(matrix(valid_run()).validate_with(false).is_ok());
    }

    #[test]
    fn missing_runtime_cap_is_rejected() {
        let mut run = valid_run();
        run.max_runtime_minutes = 0;
        assert!(rejection(run, false).contains("missing max_runtime_minutes"));
    }

    #[test]
    fn missing_disk_cap_is_rejected() {
        let mut run = valid_run();
        run.max_disk_gib = 0;
        assert!(rejection(run, false).contains("missing max_disk_gib"));
    }

    #[test]
    fn blank_router_policy_is_rejected() {
        let mut run = valid_run();
        run.router_policy = "   ".into();
        assert!(rejection(run, false).contains("missing router_policy"));
    }

    #[test]
    fn blank_norm_policy_is_rejected() {
        let mut run = valid_run();
        run.norm_policy = "\t".into();
        assert!(rejection(run, false).contains("missing norm_policy"));
    }

    #[test]
    fn unknown_model_family_is_rejected() {
        let mut run = valid_run();
        run.model_family = "not_a_family".into();
        assert!(rejection(run, false).contains("unknown model_family"));
    }

    #[test]
    fn a_non_canonical_alias_is_not_a_valid_matrix_family() {
        // `from_alias` accepts "qwen", but a matrix file must use the
        // canonical slug, so validation requires slug() == model_family.
        let mut run = valid_run();
        run.model_family = "qwen".into();
        assert!(rejection(run, false).contains("unknown model_family"));

        let mut canonical = valid_run();
        canonical.model_family = "qwen3_moe".into();
        assert!(matrix(canonical).validate_with(false).is_ok());
    }

    #[test]
    fn grok1_is_gated_on_readiness_but_other_grok_paths_are_not() {
        let mut grok1 = valid_run();
        grok1.model_family = "grok".into();
        grok1.model_id_or_path = "/models/grok-1".into();
        assert!(
            rejection(grok1.clone(), false).contains("GROK1_ARTIFACT_READY"),
            "grok-1 must be gated when not ready"
        );
        assert!(
            matrix(grok1).validate_with(true).is_ok(),
            "grok-1 must pass once ready"
        );

        // The gate keys on the grok-1 path specifically, not the family.
        let mut other_grok = valid_run();
        other_grok.model_family = "grok".into();
        other_grok.model_id_or_path = "/models/grok-2".into();
        assert!(matrix(other_grok).validate_with(false).is_ok());
    }

    /// Parse and validate the matrix actually shipped in the repository.
    ///
    /// `experiments/*/matrix.toml` was previously unchecked against this
    /// schema — nothing parsed it, so a field rename here or a typo there
    /// would only surface when someone ran the sweep. `CARGO_MANIFEST_DIR` is
    /// resolved at compile time, so this introduces no runtime path discovery.
    #[test]
    fn the_shipped_experiment_matrix_parses_and_validates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/vultr-2026-05-28/matrix.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            // The matrix is a research artifact, not a build input; skip
            // rather than fail if a checkout does not carry it.
            eprintln!("skipping: {} not present", path.display());
            return;
        };

        let parsed: RunMatrix =
            toml::from_str(&text).expect("shipped matrix must parse as RunMatrix");
        assert!(!parsed.runs.is_empty(), "shipped matrix has no runs");

        // grok_ready = true so the Grok-1 gate does not mask other problems.
        parsed
            .validate_with(true)
            .expect("shipped matrix must satisfy its own schema");
    }

    #[test]
    fn a_skipped_run_bypasses_every_check() {
        let mut run = valid_run();
        run.skip_reason = Some("checkpoint unavailable".into());
        run.max_runtime_minutes = 0;
        run.max_disk_gib = 0;
        run.router_policy = String::new();
        run.norm_policy = String::new();
        run.model_family = "not_a_family".into();
        assert!(
            matrix(run).validate_with(false).is_ok(),
            "skip_reason must short-circuit validation"
        );
    }
}
