//! GGML type constants, labels, and diagnostic helpers.

pub(super) const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub(super) const GGUF_VERSION: u32 = 3;
pub(super) const GGML_TYPE_F32: u32 = 0;
pub(super) const GGML_TYPE_F16: u32 = 1;
pub(super) const GGML_TYPE_Q8_0: u32 = 8;
pub(super) const GGML_TYPE_Q5_K: u32 = 13;
pub(super) const GGML_TYPE_Q6_K: u32 = 14;
pub(super) const GGML_TYPE_IQ3_S: u32 = 21;
pub(super) const GGML_TYPE_IQ3_M: u32 = 31;
pub(super) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(super) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(super) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(super) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(super) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(super) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

/// Map a GGUF `ggml_type` u32 to a short human label.
pub fn ggml_type_label(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        GGML_TYPE_IQ3_M => "IQ3_M",
        _ => "unknown",
    }
}

/// Returns `true` iff the runtime can consume `ggml_type` as the source
/// for the GPU synapse tensor today.
pub fn synapse_dequant_path_supported(ggml_type: u32) -> bool {
    ggml_type == GGML_TYPE_F16
        || ggml_type == GGML_TYPE_Q8_0
        || ggml_type == GGML_TYPE_Q5_K
        || ggml_type == GGML_TYPE_Q6_K
        || ggml_type == GGML_TYPE_IQ3_M
}
