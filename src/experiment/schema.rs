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

/// Static per-model-family configuration.
///
/// One `[[adapter]]` entry in `configs/model_adapter_configs.toml`. Parse the
/// file as [`ModelAdapterConfigs`], not as a bare sequence of these — the file
/// wraps them in a table array.
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

/// The whole of `configs/model_adapter_configs.toml`.
///
/// The file is a table array, so [`ModelAdapterConfig`] alone cannot represent
/// it — without this wrapper the type existed but could not parse the only
/// file it describes, and the 300-odd lines of that file went unchecked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdapterConfigs {
    #[serde(rename = "adapter")]
    pub adapters: Vec<ModelAdapterConfig>,
}

impl ModelAdapterConfigs {
    /// Check every entry names a canonical [`ModelFamily`] slug.
    ///
    /// Same rule as [`RunMatrix::validate`]: `from_alias` also accepts
    /// shorthands such as "qwen", so the slug must round-trip exactly.
    pub fn validate(&self) -> Result<()> {
        for adapter in &self.adapters {
            let known = ModelFamily::from_alias(&adapter.model_family)
                .is_some_and(|family| family.slug() == adapter.model_family);
            if !known {
                return Err(HybridError::InvalidConfig(format!(
                    "adapter entry '{}' uses unknown model_family '{}'",
                    adapter.model_id_or_local_path, adapter.model_family
                )));
            }
        }
        Ok(())
    }
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
    pub fn validate(&self) -> Result<()> {
        let grok_ready = std::env::var("GROK1_ARTIFACT_READY").is_ok_and(|v| v == "1");

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

            const KNOWN_FAMILIES: &[&str] = &[
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
            if !KNOWN_FAMILIES.contains(&run.model_family.as_str()) {
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
mod adapter_config_tests {
    use super::*;

    /// Parse and validate the adapter config actually shipped in the repo.
    ///
    /// Before `ModelAdapterConfigs` existed, `ModelAdapterConfig` described a
    /// file it could not parse — the entries are a `[[adapter]]` table array —
    /// so none of that file was ever checked against the type meant to model
    /// it. `CARGO_MANIFEST_DIR` is resolved at compile time, so this adds no
    /// runtime path discovery.
    #[test]
    fn the_shipped_adapter_config_parses_and_validates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("configs/model_adapter_configs.toml");
        // No skip branch: the file is tracked, so absence is a real failure.
        // A test that returns green when its fixture is missing reports
        // "passed" for something it never checked.
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()));

        let parsed: ModelAdapterConfigs =
            toml::from_str(&text).expect("shipped adapter config must parse");
        // Count table headers only: the file's header comment also mentions
        // "[[adapter]]" in prose, so a substring count over-reports by one.
        let declared = text
            .lines()
            .filter(|line| line.trim() == "[[adapter]]")
            .count();
        assert_eq!(
            parsed.adapters.len(),
            declared,
            "parsed {} of {declared} [[adapter]] entries",
            parsed.adapters.len()
        );
        assert!(!parsed.adapters.is_empty(), "no adapter entries parsed");

        parsed
            .validate()
            .expect("every shipped adapter entry must name a canonical family slug");
    }

    #[test]
    fn an_unknown_family_is_rejected() {
        let configs = ModelAdapterConfigs {
            adapters: vec![ModelAdapterConfig {
                model_family: "not_a_family".into(),
                model_id_or_local_path: "/models/x".into(),
                format: "gguf".into(),
                loader_hint: "gguf".into(),
                router_policy: "top_k".into(),
                norm_policy: "rms".into(),
                expert_policy: "dense".into(),
                supports_route_metrics: true,
                supports_block_metrics: true,
                preferred_quant_modes: Vec::new(),
                minimum_expected_artifacts: Vec::new(),
                known_risks: Vec::new(),
            }],
        };
        let err = configs.validate().expect_err("unknown family must reject");
        assert!(
            err.to_string().contains("unknown model_family"),
            "unexpected error: {err}"
        );
    }
}
