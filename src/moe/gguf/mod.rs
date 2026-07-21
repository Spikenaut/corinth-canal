// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Internal GGUF checkpoint implementation modules.
//!
//! Split from the former monolithic `checkpoint.rs` for reviewability.
//! Public (to `moe`) surface is re-exported here and mirrored by the
//! `checkpoint` façade so existing `super::checkpoint::{...}` imports keep
//! working.

mod cuda_register;
mod dequant;
mod map;
mod metadata;

pub(super) use dequant::{
    dequantize_row_iq3_m, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
    f16_to_f32, tensor_row_size,
};
pub(super) use map::{
    MappedGgufCheckpoint, extract_named_token_embedding_from_checkpoint, probe_and_map_checkpoint,
};
pub(super) use metadata::{
    GgufMetadata, GgufTensorInfo, ParsedCheckpointLayout, parse_checkpoint_layout,
};
