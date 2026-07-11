// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Compatibility re-export; GGUF implementation lives in [`super::gguf`].
//!
//! Callers under `moe` continue to import via `super::checkpoint::{...}`.
//! Implementation is split across private modules:
//! - `gguf/metadata.rs` — header/metadata parse, cursor
//! - `gguf/map.rs` — mmap checkpoint, tensor access, embeddings
//! - `gguf/dequant.rs` — row dequantization and f16 helpers
//! - `gguf/cuda_register.rs` — CUDA host registration (feature-gated)

// Full former `pub(super)` surface is re-exported for import stability even
// when a given name is only used by some call sites / cfg combinations.
#[allow(unused_imports)]
pub(super) use super::gguf::{
    GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint, ParsedCheckpointLayout,
    dequantize_row_iq3_m, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
    extract_named_token_embedding_from_checkpoint, f16_to_f32, parse_checkpoint_layout,
    probe_and_map_checkpoint, tensor_row_size,
};
