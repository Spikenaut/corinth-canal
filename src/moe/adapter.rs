// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Model-family adapter resolution for the GGUF and Safetensors router host.

use super::checkpoint::{GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint};
use super::safetensors::{MappedSafetensorsCheckpoint, SafetensorsMetadata};
use super::{GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
use crate::error::{HybridError, Result};
use crate::types::ModelFamily;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SynapseSource {
    Real,
    DequantizedQ8_0,
    DequantizedQ5K,
    DequantizedQ6K,
    DequantizedIQ3M,
    RoutingF32,
    DequantizedInt4,
    SyntheticFallback,
}

#[derive(Debug, Clone)]
pub(super) struct ModelAdapter {
    pub(super) family: ModelFamily,
    pub(super) architecture: String,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) num_experts: usize,
    pub(super) expert_used_count: usize,
    pub(super) token_embedding_tensor: String,
    pub(super) routing_tensor: String,
    pub(super) preferred_gpu_synapse_tensor: Option<String>,
    pub(super) real_gpu_synapse_tensor: Option<String>,
    pub(super) dequant_q8_0_synapse_tensor: Option<String>,
    pub(super) dequant_q5_k_synapse_tensor: Option<String>,
    // Staged for future CUDA synapse paths; only written, never read at runtime.
    #[allow(dead_code)]
    pub(super) dequant_q6_k_synapse_tensor: Option<String>,
    // Staged for future IQ3_M CUDA dequant path; only read in tests.
    #[allow(dead_code)]
    pub(super) dequant_iq3_m_synapse_tensor: Option<String>,
    #[allow(dead_code)]
    pub(super) routing_f32_synapse_tensor: Option<String>,
    // Staged for future Int4 CUDA dequant path; only written, never read at runtime.
    #[allow(dead_code)]
    pub(super) dequant_int4_synapse_tensor: Option<String>,
    pub(super) synapse_source: SynapseSource,
    pub(super) quantization: String,
}

impl ModelAdapter {
    pub(super) fn synapse_source_label(&self) -> &'static str {
        match self.synapse_source {
            SynapseSource::Real => "real",
            SynapseSource::DequantizedQ8_0 => "dequantized-q8_0",
            SynapseSource::DequantizedQ5K => "dequantized-q5_k",
            SynapseSource::DequantizedQ6K => "dequantized-q6_k",
            SynapseSource::DequantizedIQ3M => "dequantized-iq3_m",
            SynapseSource::RoutingF32 => "routing-f32",
            SynapseSource::DequantizedInt4 => "dequantized-int4",
            SynapseSource::SyntheticFallback => "synthetic-fallback",
        }
    }
}

fn resolve_token_embedding_tensor(checkpoint: &MappedGgufCheckpoint, path: &str) -> Result<String> {
    if checkpoint.has_tensor("token_embd.weight") {
        Ok("token_embd.weight".to_owned())
    } else if checkpoint.has_tensor("tok_embeddings.weight") {
        Ok("tok_embeddings.weight".to_owned())
    } else {
        Err(HybridError::MissingTensor {
            name: "token_embd.weight".into(),
            path: path.to_owned(),
        })
    }
}

fn resolve_routing_tensor(
    checkpoint: &MappedGgufCheckpoint,
    num_experts: usize,
    path: &str,
) -> Result<String> {
    let routing_tensor = checkpoint
        .find_first_tensor_with_suffix("ffn_gate_inp.weight")
        .or_else(|| checkpoint.find_first_tensor_with_suffix("ffn_gate.weight"))
        .ok_or_else(|| HybridError::MissingTensor {
            name: "ffn_gate_inp.weight".into(),
            path: path.to_owned(),
        })?
        .to_owned();
    let routing_info = checkpoint.tensor_info(&routing_tensor, path)?;
    if routing_info.ggml_type != GGML_TYPE_F32 || routing_info.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 F32 in '{path}', got dims={:?} ggml_type={}",
            routing_info.dims, routing_info.ggml_type
        )));
    }
    let routing_experts = routing_info.dims[0].min(routing_info.dims[1]);
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }
    Ok(routing_tensor)
}

fn resolve_gguf_topology(
    metadata: &GgufMetadata,
    architecture: &str,
    path: &str,
) -> Result<(usize, usize, usize, usize)> {
    let hidden_size = metadata
        .numeric(&format!("{architecture}.embedding_length"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.embedding_length' in '{path}'"
            ))
        })?;
    let num_layers = metadata
        .numeric(&format!("{architecture}.block_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.block_count' in '{path}'"
            ))
        })?;
    let num_experts = metadata
        .numeric(&format!("{architecture}.expert_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.expert_count' in '{path}'"
            ))
        })?;
    let expert_used_count = metadata
        .numeric(&format!("{architecture}.expert_used_count"))
        .unwrap_or(1);
    Ok((hidden_size, num_layers, num_experts, expert_used_count))
}

pub(super) fn resolve_adapter(
    metadata: &GgufMetadata,
    checkpoint: &MappedGgufCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture().to_owned();
    let family = infer_family(&architecture, family_override, path)?;
    let (hidden_size, num_layers, num_experts, expert_used_count) =
        resolve_gguf_topology(metadata, &architecture, path)?;
    let token_embedding_tensor = resolve_token_embedding_tensor(checkpoint, path)?;

    let routing_tensor = resolve_routing_tensor(checkpoint, num_experts, path)?;

    let preferred_gpu_synapse_tensor = checkpoint
        .has_tensor("blk.0.attn_q.weight")
        .then(|| "blk.0.attn_q.weight".to_owned());
    let selection = select_gguf_synapse(
        preferred_gpu_synapse_tensor.as_deref(),
        &routing_tensor,
        checkpoint,
        metadata,
        hidden_size,
        path,
    );

    Ok(ModelAdapter {
        family,
        architecture,
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
        token_embedding_tensor,
        routing_tensor,
        preferred_gpu_synapse_tensor,
        synapse_source: selection.synapse_source,
        real_gpu_synapse_tensor: selection.real_gpu_synapse_tensor,
        dequant_q8_0_synapse_tensor: selection.dequant_q8_0_synapse_tensor,
        dequant_q5_k_synapse_tensor: selection.dequant_q5_k_synapse_tensor,
        dequant_q6_k_synapse_tensor: selection.dequant_q6_k_synapse_tensor,
        dequant_iq3_m_synapse_tensor: selection.dequant_iq3_m_synapse_tensor,
        routing_f32_synapse_tensor: selection.routing_f32_synapse_tensor,
        dequant_int4_synapse_tensor: None,
        quantization: metadata.quantization().to_owned(),
    })
}

/// Priority-ordered GGUF synapse source selection.
///
/// First matching candidate wins; all lower-priority Option fields stay `None`.
struct SynapseSelection {
    synapse_source: SynapseSource,
    real_gpu_synapse_tensor: Option<String>,
    dequant_q8_0_synapse_tensor: Option<String>,
    dequant_q5_k_synapse_tensor: Option<String>,
    dequant_q6_k_synapse_tensor: Option<String>,
    dequant_iq3_m_synapse_tensor: Option<String>,
    routing_f32_synapse_tensor: Option<String>,
}

fn empty_synapse_selection() -> SynapseSelection {
    SynapseSelection {
        synapse_source: SynapseSource::SyntheticFallback,
        real_gpu_synapse_tensor: None,
        dequant_q8_0_synapse_tensor: None,
        dequant_q5_k_synapse_tensor: None,
        dequant_q6_k_synapse_tensor: None,
        dequant_iq3_m_synapse_tensor: None,
        routing_f32_synapse_tensor: None,
    }
}

fn select_real_synapse(
    name: &str,
    info: &GgufTensorInfo,
    hidden_size: usize,
) -> Option<SynapseSelection> {
    (info.ggml_type == GGML_TYPE_F16 && info.dims == [hidden_size, hidden_size]).then(|| {
        SynapseSelection {
            synapse_source: SynapseSource::Real,
            real_gpu_synapse_tensor: Some(name.to_owned()),
            ..empty_synapse_selection()
        }
    })
}

fn is_quantization_iq3_m(quantization: &str) -> bool {
    quantization.contains("IQ3_M") || quantization.contains("iq3_m")
}

fn select_quantized_synapse(
    name: &str,
    info: &GgufTensorInfo,
    metadata: &GgufMetadata,
) -> Option<SynapseSelection> {
    if info.dims.len() != 2 {
        return None;
    }
    let d0 = info.dims[0];
    let is_iq3_m = is_quantization_iq3_m(metadata.quantization());
    match info.ggml_type {
        GGML_TYPE_Q8_0 if d0.is_multiple_of(32) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ8_0,
            dequant_q8_0_synapse_tensor: Some(name.to_owned()),
            ..empty_synapse_selection()
        }),
        GGML_TYPE_Q5_K if d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ5K,
            dequant_q5_k_synapse_tensor: Some(name.to_owned()),
            ..empty_synapse_selection()
        }),
        GGML_TYPE_Q6_K if d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ6K,
            dequant_q6_k_synapse_tensor: Some(name.to_owned()),
            ..empty_synapse_selection()
        }),
        _ if is_iq3_m && d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedIQ3M,
            dequant_iq3_m_synapse_tensor: Some(name.to_owned()),
            ..empty_synapse_selection()
        }),
        _ => None,
    }
}

fn select_preferred_synapse(
    name: &str,
    info: &GgufTensorInfo,
    hidden_size: usize,
    metadata: &GgufMetadata,
) -> Option<SynapseSelection> {
    select_real_synapse(name, info, hidden_size)
        .or_else(|| select_quantized_synapse(name, info, metadata))
}

fn select_gguf_synapse(
    preferred: Option<&str>,
    routing_tensor: &str,
    checkpoint: &MappedGgufCheckpoint,
    metadata: &GgufMetadata,
    hidden_size: usize,
    path: &str,
) -> SynapseSelection {
    if let Some(name) = preferred
        && let Ok(info) = checkpoint.tensor_info(name, path)
        && let Some(selection) = select_preferred_synapse(name, info, hidden_size, metadata)
    {
        return selection;
    }

    // RoutingF32 fallback: reached when no preferred tensor is present or none matched a dequant path.
    SynapseSelection {
        synapse_source: SynapseSource::RoutingF32,
        routing_f32_synapse_tensor: Some(routing_tensor.to_owned()),
        ..empty_synapse_selection()
    }
}

pub(super) fn resolve_safetensors_adapter(
    metadata: &SafetensorsMetadata,
    checkpoint: &MappedSafetensorsCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture.clone();
    let family = infer_family_safetensors(&architecture, family_override, path)?;
    let hidden_size = metadata.hidden_size;
    let num_layers = metadata.num_layers;
    let num_experts = metadata.num_experts;
    let expert_used_count = metadata.expert_used_count;

    let token_embedding_tensor = "model.embed_tokens.weight".to_owned();
    let routing_tensor = "model.layers.0.mlp.gate.weight".to_owned();

    // Validate routing tensor exists and is rank-2
    let routing_info =
        checkpoint
            .tensor_info(&routing_tensor)
            .ok_or_else(|| HybridError::MissingTensor {
                name: routing_tensor.clone(),
                path: path.to_owned(),
            })?;
    if routing_info.1.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 in '{path}', got dims={:?}",
            routing_info.1
        )));
    }
    let routing_experts = routing_info.1[0].min(routing_info.1[1]);
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }

    // Safetensors path: look for gate weight as synapse tensor;
    // detect Int4 quantized weights. Note: float dtypes (F16/BF16/F32)
    // remain synthetic fallback because the GPU synapse loader only
    // supports GGUF-registered tensors today.
    let preferred_gpu_synapse_tensor = checkpoint
        .tensor_info("model.layers.0.mlp.gate.weight")
        .map(|_| "model.layers.0.mlp.gate.weight".to_owned());
    let real_gpu_synapse_tensor = None;
    let dequant_int4_synapse_tensor = preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
        let info = checkpoint.tensor_info(name)?;
        (info.0 == "INT4" || info.0 == "I4" || info.0 == "U4").then(|| name.clone())
    });
    let synapse_source = if dequant_int4_synapse_tensor.is_some() {
        SynapseSource::DequantizedInt4
    } else {
        SynapseSource::SyntheticFallback
    };

    Ok(ModelAdapter {
        family,
        architecture,
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
        token_embedding_tensor,
        routing_tensor,
        preferred_gpu_synapse_tensor,
        synapse_source,
        real_gpu_synapse_tensor,
        dequant_q8_0_synapse_tensor: None,
        dequant_q5_k_synapse_tensor: None,
        dequant_q6_k_synapse_tensor: None,
        dequant_iq3_m_synapse_tensor: None,
        routing_f32_synapse_tensor: None,
        dequant_int4_synapse_tensor,
        quantization: "safetensors".into(),
    })
}

fn check_family_compatibility(expected: ModelFamily, inferred: ModelFamily) -> bool {
    expected == inferred
        || matches!(
            (expected, inferred),
            (ModelFamily::Moonlight16BA3B, ModelFamily::DeepSeek2)
                | (ModelFamily::DeepSeek2, ModelFamily::Moonlight16BA3B)
        )
}

/// Discriminator for architecture-string tables (GGUF vs Safetensors/HF names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchFormat {
    Gguf,
    Safetensors,
}

impl ArchFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::Safetensors => "Safetensors",
        }
    }
}

fn map_architecture(architecture: &str, format: ArchFormat) -> Option<ModelFamily> {
    match format {
        ArchFormat::Gguf => match architecture {
            "olmoe" => Some(ModelFamily::Olmoe),
            "qwen3moe" => Some(ModelFamily::Qwen3Moe),
            "gemma4" => Some(ModelFamily::Gemma4),
            "deepseek2" => Some(ModelFamily::DeepSeek2),
            "llama" => Some(ModelFamily::LlamaMoe),
            "moonlight" => Some(ModelFamily::Moonlight16BA3B),
            "granite" | "granitemoe" => Some(ModelFamily::Granite31A800M),
            "nemotronh" => Some(ModelFamily::Nemotron),
            "lfm2" | "lfm2moe" => Some(ModelFamily::Lfm2Moe),
            "phimoe" | "slimmoe" => Some(ModelFamily::SlimMoe),
            "zaya" => Some(ModelFamily::Zaya),
            "glm4" | "glm4moe" => Some(ModelFamily::Glm4),
            "gptoss" => Some(ModelFamily::GptOss),
            "step" | "step3" => Some(ModelFamily::Step),
            "minimax" => Some(ModelFamily::MiniMax),
            "cohere" => Some(ModelFamily::Cohere),
            "grin" | "grinmoe" => Some(ModelFamily::Grin),
            "skyworks" | "skyworkmoe" => Some(ModelFamily::Skyworks),
            "trinity" => Some(ModelFamily::Trinity),
            "grok" => Some(ModelFamily::Grok),
            _ => None,
        },
        ArchFormat::Safetensors => match architecture {
            "OlmoeForCausalLM" => Some(ModelFamily::Olmoe),
            "Qwen3MoeForCausalLM" => Some(ModelFamily::Qwen3Moe),
            "Gemma4ForCausalLM" => Some(ModelFamily::Gemma4),
            // DeepSeek-V2 class name; V3 is the HF tag used by Moonlight-16B-A3B
            // Safetensors packages (see configs/local_safetensors_lineup.template.toml).
            "DeepseekV2ForCausalLM" => Some(ModelFamily::DeepSeek2),
            "DeepseekV3ForCausalLM" => Some(ModelFamily::Moonlight16BA3B),
            "LlamaMoeForCausalLM" => Some(ModelFamily::LlamaMoe),
            "MoonlightForCausalLM" => Some(ModelFamily::Moonlight16BA3B),
            // Dense `GraniteForCausalLM` is intentionally not mapped — the MoE
            // A800M target reports `GraniteMoeForCausalLM` (Codex review on #125).
            "GraniteMoeForCausalLM" => Some(ModelFamily::Granite31A800M),
            "ZayaForCausalLM" => Some(ModelFamily::Zaya),
            "Glm4ForCausalLM" | "Glm4MoeForCausalLM" => Some(ModelFamily::Glm4),
            "NemotronHForCausalLM" => Some(ModelFamily::Nemotron),
            "Lfm2MoeForCausalLM" => Some(ModelFamily::Lfm2Moe),
            "PhiMoEForCausalLM" => Some(ModelFamily::SlimMoe),
            "GptOssForCausalLM" => Some(ModelFamily::GptOss),
            "Step3ForCausalLM" => Some(ModelFamily::Step),
            "MiniMaxForCausalLM" => Some(ModelFamily::MiniMax),
            "CohereForCausalLM" => Some(ModelFamily::Cohere),
            "GrinMoeForCausalLM" => Some(ModelFamily::Grin),
            "SkyworkMoeForCausalLM" => Some(ModelFamily::Skyworks),
            "TrinityForCausalLM" => Some(ModelFamily::Trinity),
            "GrokForCausalLM" => Some(ModelFamily::Grok),
            _ => None,
        },
    }
}

/// Unified family inference for GGUF and Safetensors architecture strings.
///
/// Format-specific arch tables live in [`map_architecture`]; override
/// compatibility and error formatting are shared here.
fn infer_family_for_format(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
    format: ArchFormat,
) -> Result<ModelFamily> {
    let inferred = map_architecture(architecture, format).ok_or_else(|| {
        HybridError::UnsupportedFormat(format!(
            "unsupported {} architecture '{architecture}' in '{path}'",
            format.label()
        ))
    })?;

    if let Some(expected) = family_override {
        if !check_family_compatibility(expected, inferred) {
            return Err(HybridError::InvalidConfig(format!(
                "model_family override {:?} does not match {} architecture '{architecture}'",
                expected,
                format.label()
            )));
        }
        Ok(expected)
    } else {
        Ok(inferred)
    }
}

/// GGUF entry point (kept for call sites and unit tests).
fn infer_family(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    infer_family_for_format(architecture, family_override, path, ArchFormat::Gguf)
}

/// Safetensors entry point (kept for call sites and unit tests).
fn infer_family_safetensors(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    infer_family_for_format(architecture, family_override, path, ArchFormat::Safetensors)
}

#[cfg(test)]
mod tests {
    use super::infer_family;
    use crate::types::ModelFamily;

    #[test]
    fn infer_family_supports_zaya_and_glm4_architectures() {
        assert_eq!(
            infer_family("zaya", None, "test.gguf").unwrap(),
            ModelFamily::Zaya
        );
        assert_eq!(
            infer_family("glm4", None, "test.gguf").unwrap(),
            ModelFamily::Glm4
        );
        assert_eq!(
            infer_family("glm4moe", None, "test.gguf").unwrap(),
            ModelFamily::Glm4
        );
    }

    #[test]
    fn infer_family_supports_granite_architecture() {
        assert_eq!(
            infer_family("granite", None, "test.gguf").unwrap(),
            ModelFamily::Granite31A800M
        );
    }

    #[test]
    fn infer_family_supports_moonlight_architecture() {
        assert_eq!(
            infer_family("moonlight", None, "test.gguf").unwrap(),
            ModelFamily::Moonlight16BA3B
        );
    }

    #[test]
    fn infer_family_supports_granitemoe_architecture() {
        assert_eq!(
            infer_family("granitemoe", None, "test.gguf").unwrap(),
            ModelFamily::Granite31A800M
        );
    }

    #[test]
    fn infer_family_supports_nemotron_architecture() {
        assert_eq!(
            infer_family("nemotronh", None, "test.gguf").unwrap(),
            ModelFamily::Nemotron
        );
    }

    #[test]
    fn infer_family_supports_new_cloud_architectures() {
        assert_eq!(
            infer_family("gptoss", None, "test.gguf").unwrap(),
            ModelFamily::GptOss
        );
        assert_eq!(
            infer_family("step3", None, "test.gguf").unwrap(),
            ModelFamily::Step
        );
        assert_eq!(
            infer_family("grinmoe", None, "test.gguf").unwrap(),
            ModelFamily::Grin
        );
        assert_eq!(
            infer_family("skyworks", None, "test.gguf").unwrap(),
            ModelFamily::Skyworks
        );
        assert_eq!(
            infer_family("trinity", None, "test.gguf").unwrap(),
            ModelFamily::Trinity
        );
        assert_eq!(
            infer_family("grok", None, "test.gguf").unwrap(),
            ModelFamily::Grok
        );
    }

    #[test]
    fn infer_family_safetensors_moonlight_and_granite_moe_tags() {
        assert_eq!(
            super::infer_family_safetensors("DeepseekV3ForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("MoonlightForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("GraniteMoeForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Granite31A800M
        );
        // Dense Granite is not the MoE A800M family.
        assert!(
            super::infer_family_safetensors("GraniteForCausalLM", None, "test.safetensors")
                .is_err()
        );
    }

    #[test]
    fn test_family_compatibility_overrides() {
        // Test Moonlight/DeepSeek2 compatibility checks for GGUF path
        assert_eq!(
            infer_family("moonlight", Some(ModelFamily::DeepSeek2), "test.gguf").unwrap(),
            ModelFamily::DeepSeek2
        );
        assert_eq!(
            infer_family("deepseek2", Some(ModelFamily::Moonlight16BA3B), "test.gguf").unwrap(),
            ModelFamily::Moonlight16BA3B
        );

        // Verify that incompatible overrides still error
        assert!(infer_family("olmoe", Some(ModelFamily::DeepSeek2), "test.gguf").is_err());

        // Test Moonlight/DeepSeek2 compatibility checks for Safetensors path
        assert_eq!(
            super::infer_family_safetensors(
                "DeepseekV2ForCausalLM",
                Some(ModelFamily::Moonlight16BA3B),
                "test.safetensors"
            )
            .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("DeepseekV2ForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::DeepSeek2
        );
        assert!(
            super::infer_family_safetensors(
                "OlmoeForCausalLM",
                Some(ModelFamily::DeepSeek2),
                "test.safetensors"
            )
            .is_err()
        );
    }
}
