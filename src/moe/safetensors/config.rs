// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Hugging Face `config.json` parsing for Safetensors MoE adapter resolution.

use serde::de::Deserializer;
use serde::de::{self, SeqAccess, Visitor};

/// Nested language-model config used by multimodal HF packages (Gemma4,
/// Cohere2Vision, Llama4, …) where MoE fields live under `text_config`.
#[derive(Debug, serde::Deserialize)]
struct HfTextConfig {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    /// Skywork stores this as a single-element array (`[16]`); accept either form.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_usize_scalar_or_singleton_array"
    )]
    num_experts: Option<usize>,
    #[serde(default)]
    num_local_experts: Option<usize>,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    /// Gemma4 multimodal packages use `top_k_experts` instead of `num_experts_per_tok`.
    #[serde(default)]
    top_k_experts: Option<usize>,
    /// Step-3.5-Flash-style packs (also accepted under nested configs).
    #[serde(default)]
    moe_num_experts: Option<usize>,
    #[serde(default)]
    moe_top_k: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct HfConfig {
    pub(super) architectures: Vec<String>,
    /// Top-level fields are optional so multimodal configs that only nest
    /// them under `text_config` still parse.
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    /// Skywork stores this as a single-element array (`[16]`); accept either form.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_usize_scalar_or_singleton_array"
    )]
    num_experts: Option<usize>,
    /// Granite MoE and Llama4 Scout use this instead of `num_experts`.
    #[serde(default)]
    num_local_experts: Option<usize>,
    /// DeepSeek-V3 / Moonlight use this instead of `num_experts`.
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    /// Gemma4 and some MoE packs use this alias for expert fan-out (top-k).
    #[serde(default)]
    top_k_experts: Option<usize>,
    /// Step-3.5-Flash: `moe_num_experts` / `moe_top_k` instead of num_experts*.
    #[serde(default)]
    moe_num_experts: Option<usize>,
    #[serde(default)]
    moe_top_k: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    text_config: Option<HfTextConfig>,
}

impl HfConfig {
    pub(super) fn resolved_hidden_size(&self) -> Option<usize> {
        self.hidden_size
            .or_else(|| self.text_config.as_ref().and_then(|t| t.hidden_size))
    }

    pub(super) fn resolved_num_layers(&self) -> Option<usize> {
        self.num_hidden_layers
            .or_else(|| self.text_config.as_ref().and_then(|t| t.num_hidden_layers))
    }

    pub(super) fn resolved_num_experts(&self) -> usize {
        self.num_experts
            .or(self.num_local_experts)
            .or(self.n_routed_experts)
            .or(self.moe_num_experts)
            .or_else(|| self.text_config.as_ref().and_then(|t| t.num_experts))
            .or_else(|| self.text_config.as_ref().and_then(|t| t.num_local_experts))
            .or_else(|| self.text_config.as_ref().and_then(|t| t.n_routed_experts))
            .or_else(|| self.text_config.as_ref().and_then(|t| t.moe_num_experts))
            .unwrap_or(1)
    }

    pub(super) fn resolved_expert_used_count(&self) -> usize {
        self.num_experts_per_tok
            .or(self.top_k_experts)
            .or(self.moe_top_k)
            .or_else(|| {
                self.text_config
                    .as_ref()
                    .and_then(|t| t.num_experts_per_tok)
            })
            .or_else(|| self.text_config.as_ref().and_then(|t| t.top_k_experts))
            .or_else(|| self.text_config.as_ref().and_then(|t| t.moe_top_k))
            .unwrap_or(1)
    }

    pub(super) fn resolved_vocab_size(&self) -> Option<usize> {
        self.vocab_size
            .or_else(|| self.text_config.as_ref().and_then(|t| t.vocab_size))
    }
}

/// Accept `num_experts` as a scalar (`16`) or single-element array (`[16]`).
///
/// Skywork-MoE HF configs ship the latter; multi-element arrays are rejected so
/// heterogeneous layer expert counts stay explicit failures.
fn deserialize_optional_usize_scalar_or_singleton_array<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptUsizeVisitor;

    impl<'de> Visitor<'de> for OptUsizeVisitor {
        type Value = Option<usize>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a non-negative integer or a single-element integer array")
        }

        fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer.deserialize_any(OptUsizeVisitor)
        }

        fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            usize::try_from(value)
                .map(Some)
                .map_err(|_| E::custom(format!("num_experts {value} does not fit usize")))
        }

        fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            if value < 0 {
                return Err(E::custom(format!(
                    "num_experts must be non-negative, got {value}"
                )));
            }
            usize::try_from(value)
                .map(Some)
                .map_err(|_| E::custom(format!("num_experts {value} does not fit usize")))
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let first: Option<u64> = seq.next_element()?;
            let second: Option<u64> = seq.next_element()?;
            match (first, second) {
                (None, _) => Ok(None),
                (Some(v), None) => usize::try_from(v)
                    .map(Some)
                    .map_err(|_| de::Error::custom(format!("num_experts {v} does not fit usize"))),
                (Some(_), Some(_)) => Err(de::Error::custom(
                    "num_experts array must contain exactly one element (got more)",
                )),
            }
        }
    }

    deserializer.deserialize_any(OptUsizeVisitor)
}
