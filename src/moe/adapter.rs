//! Model-family adapter resolution for the GGUF and Safetensors router host.

use super::checkpoint::{GgufMetadata, MappedGgufCheckpoint};
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
    DequantizedInt4,
    SyntheticFallback,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    pub(super) dequant_q6_k_synapse_tensor: Option<String>,
    pub(super) dequant_iq3_m_synapse_tensor: Option<String>,
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
            SynapseSource::DequantizedInt4 => "dequantized-int4",
            SynapseSource::SyntheticFallback => "synthetic-fallback",
        }
    }
}

pub(super) fn resolve_adapter(
    metadata: &GgufMetadata,
    checkpoint: &MappedGgufCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture().to_owned();
    let family = infer_family(&architecture, family_override, path)?;
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
    let token_embedding_tensor = if checkpoint.has_tensor("token_embd.weight") {
        "token_embd.weight".to_owned()
    } else if checkpoint.has_tensor("tok_embeddings.weight") {
        "tok_embeddings.weight".to_owned()
    } else {
        return Err(HybridError::MissingTensor {
            name: "token_embd.weight".into(),
            path: path.to_owned(),
        });
    };

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

    let preferred_gpu_synapse_tensor = checkpoint
        .has_tensor("blk.0.attn_q.weight")
        .then(|| "blk.0.attn_q.weight".to_owned());
    let real_gpu_synapse_tensor = preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
        let info = checkpoint.tensor_info(name, path).ok()?;
        (info.ggml_type == GGML_TYPE_F16 && info.dims == [hidden_size, hidden_size])
            .then(|| name.clone())
    });
    let dequant_q8_0_synapse_tensor = if real_gpu_synapse_tensor.is_none() {
        preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
            let info = checkpoint.tensor_info(name, path).ok()?;
            (info.ggml_type == GGML_TYPE_Q8_0 && info.dims.len() == 2 && info.dims[0] % 32 == 0)
                .then(|| name.clone())
        })
    } else {
        None
    };

    let dequant_q5_k_synapse_tensor = if real_gpu_synapse_tensor.is_none()
        && dequant_q8_0_synapse_tensor.is_none()
    {
        preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
            let info = checkpoint.tensor_info(name, path).ok()?;
            (info.ggml_type == GGML_TYPE_Q5_K && info.dims.len() == 2 && info.dims[0] % 256 == 0)
                .then(|| name.clone())
        })
    } else {
        None
    };

    let dequant_q6_k_synapse_tensor = if real_gpu_synapse_tensor.is_none()
        && dequant_q8_0_synapse_tensor.is_none()
        && dequant_q5_k_synapse_tensor.is_none()
    {
        preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
            let info = checkpoint.tensor_info(name, path).ok()?;
            (info.ggml_type == GGML_TYPE_Q6_K && info.dims.len() == 2 && info.dims[0] % 256 == 0)
                .then(|| name.clone())
        })
    } else {
        None
    };

    let dequant_iq3_m_synapse_tensor = if real_gpu_synapse_tensor.is_none()
        && dequant_q8_0_synapse_tensor.is_none()
        && dequant_q5_k_synapse_tensor.is_none()
        && dequant_q6_k_synapse_tensor.is_none()
    {
        preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
            let info = checkpoint.tensor_info(name, path).ok()?;
            // IQ3_M detection: check quantization metadata + tensor dimensions
            // (individual tensor ggml_type may vary in mixed-quant models)
            let is_iq3_m = metadata.quantization().contains("IQ3_M")
                || metadata.quantization().contains("iq3_m");
            (is_iq3_m && info.dims.len() == 2 && info.dims[0] % 256 == 0)
                .then(|| name.clone())
        })
    } else {
        None
    };

    let synapse_source = if real_gpu_synapse_tensor.is_some() {
        SynapseSource::Real
    } else if dequant_q8_0_synapse_tensor.is_some() {
        SynapseSource::DequantizedQ8_0
    } else if dequant_q5_k_synapse_tensor.is_some() {
        SynapseSource::DequantizedQ5K
    } else if dequant_q6_k_synapse_tensor.is_some() {
        SynapseSource::DequantizedQ6K
    } else if dequant_iq3_m_synapse_tensor.is_some() {
        SynapseSource::DequantizedIQ3M
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
        dequant_q8_0_synapse_tensor,
        dequant_q5_k_synapse_tensor,
        dequant_q6_k_synapse_tensor,
        dequant_iq3_m_synapse_tensor,
        dequant_int4_synapse_tensor: None,
        quantization: metadata.quantization().to_owned(),
    })
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
        dequant_int4_synapse_tensor,
        quantization: "safetensors".into(),
    })
}

fn infer_family_safetensors(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    let inferred = match architecture {
        "OlmoeForCausalLM" => ModelFamily::Olmoe,
        "Qwen3MoeForCausalLM" => ModelFamily::Qwen3Moe,
        "Gemma4ForCausalLM" => ModelFamily::Gemma4,
        "DeepseekV2ForCausalLM" => ModelFamily::DeepSeek2,
        "LlamaMoeForCausalLM" => ModelFamily::LlamaMoe,
        "NemotronHForCausalLM" => ModelFamily::Nemotron3Nano4B,
        "Lfm2MoeForCausalLM" => ModelFamily::Lfm2Moe,
        "PhiMoEForCausalLM" => ModelFamily::SlimMoe,
        other => {
            return Err(HybridError::UnsupportedFormat(format!(
                "unsupported Safetensors architecture '{other}' in '{path}'"
            )));
        }
    };

    if let Some(expected) = family_override
        && expected != inferred
    {
        return Err(HybridError::InvalidConfig(format!(
            "model_family override {:?} does not match Safetensors architecture '{architecture}'",
            expected
        )));
    }

    Ok(inferred)
}

fn infer_family(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    let inferred = match architecture {
        "olmoe" => ModelFamily::Olmoe,
        "qwen3moe" => ModelFamily::Qwen3Moe,
        "gemma4" => ModelFamily::Gemma4,
        "deepseek2" => ModelFamily::DeepSeek2,
        "llama" => ModelFamily::LlamaMoe,
        "moonlight" => ModelFamily::Moonlight16BA3B,
        "granite" | "granitemoe" => ModelFamily::Granite31A800M,
        "nemotronh" => ModelFamily::Nemotron3Nano4B,
        "lfm2" | "lfm2moe" => ModelFamily::Lfm2Moe,
        "phimoe" | "slimmoe" => ModelFamily::SlimMoe,
        "zaya" => ModelFamily::Zaya,
        "glm4" | "glm4moe" => ModelFamily::Glm4,
        other => {
            return Err(HybridError::UnsupportedFormat(format!(
                "unsupported GGUF architecture '{other}' in '{path}'"
            )));
        }
    };

    if let Some(expected) = family_override
        && expected != inferred
    {
        return Err(HybridError::InvalidConfig(format!(
            "model_family override {:?} does not match GGUF architecture '{architecture}'",
            expected
        )));
    }

    Ok(inferred)
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
}
