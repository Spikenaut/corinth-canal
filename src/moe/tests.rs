use super::*;
use std::io::Write;
use std::path::PathBuf;

fn write_temp_file(bytes: &[u8], label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "corinth_canal_{label}_{}.gguf",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    path
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    push_string(out, key);
    push_u32(out, GGUF_VALUE_TYPE_UINT32);
    push_u32(out, value);
}

fn push_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    push_string(out, key);
    push_u32(out, GGUF_VALUE_TYPE_STRING);
    push_string(out, value);
}

fn build_test_gguf(tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>, alignment: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, 7);
    push_kv_u32(&mut out, "general.alignment", alignment);
    push_kv_u32(&mut out, "general.file_type", 1);
    push_kv_string(&mut out, "general.architecture", "olmoe");
    push_kv_u32(&mut out, "olmoe.embedding_length", EMBEDDING_DIM as u32);
    push_kv_u32(&mut out, "olmoe.block_count", 16);
    push_kv_u32(&mut out, "olmoe.expert_count", 64);
    push_kv_u32(&mut out, "olmoe.expert_used_count", 8);

    let mut data_offset = 0usize;
    let mut tensor_payloads = Vec::new();
    for (name, dims, ggml_type, payload) in tensors {
        push_string(&mut out, name);
        push_u32(&mut out, dims.len() as u32);
        for dim in &dims {
            push_u64(&mut out, *dim as u64);
        }
        push_u32(&mut out, ggml_type);
        push_u64(&mut out, data_offset as u64);
        data_offset += payload.len();
        tensor_payloads.push(payload);
    }

    while out.len() % alignment as usize != 0 {
        out.push(0);
    }
    for payload in tensor_payloads {
        out.extend_from_slice(&payload);
    }

    out
}

fn build_real_size_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2];
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn build_quantized_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                Vec::new(),
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

/// Build a minimal valid Q8_0 payload for a tensor of `width * n_rows`
/// elements.  Each block uses `scale_bits` as the raw F16 scale and
/// `quant_val` for every quantized byte.
fn build_q8_0_payload(width: usize, n_rows: usize, scale_bits: u16, quant_val: i8) -> Vec<u8> {
    assert!(
        width.is_multiple_of(32),
        "Q8_0 width must be divisible by 32"
    );
    let blocks_per_row = width / 32;
    let row_bytes = blocks_per_row * 34;
    let mut out = vec![0u8; row_bytes * n_rows];
    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 34;
            let [lo, hi] = scale_bits.to_le_bytes();
            out[blk_start] = lo;
            out[blk_start + 1] = hi;
            for q in 0..32 {
                out[blk_start + 2 + q] = quant_val as u8;
            }
        }
    }
    out
}

fn build_q8_0_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    // Q8_0 payload: scale = 1.0 (f16 bits = 0x3c00), quant = 1
    let attn_q_payload = build_q8_0_payload(EMBEDDING_DIM, EMBEDDING_DIM, 0x3c00, 1);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q8_0,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

/// Build a minimal valid Q5_K payload for a tensor of `width * n_rows`
/// elements. Q5_K block layout (176 bytes per 256 elements):
/// - d (f16, 2 bytes): scale
/// - dmin (f16, 2 bytes): min scale
/// - scales (12 bytes): 6 pairs of (scale, min) for 32-element chunks
/// - qh (32 bytes): high 2 bits for each of 256 quant values
/// - ql (128 bytes): low 4 bits for each of 256 quant values
///
/// For simplicity, this creates a payload where all quant values are 1
/// and scales are set to produce output values of 1.0.
fn build_q5_k_payload(width: usize, n_rows: usize) -> Vec<u8> {
    assert!(
        width.is_multiple_of(256),
        "Q5_K width must be divisible by 256"
    );
    let blocks_per_row = width / 256;
    let row_bytes = blocks_per_row * 176;
    let mut out = vec![0u8; row_bytes * n_rows];

    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 176;

            // d = 1.0 (f16 bits = 0x3c00)
            out[blk_start] = 0x00;
            out[blk_start + 1] = 0x3c;

            // dmin = 0.0 (f16 bits = 0x0000)
            out[blk_start + 2] = 0x00;
            out[blk_start + 3] = 0x00;

            // scales: 6 pairs of (sc, m) for 32-element chunks
            // We want sc=1, m=0 for all chunks to get output = 1.0 * 1 - 0.0 = 1.0
            // scale_min_k4 encoding: lower 6 bits for scale, upper 2 bits contribute to min
            for i in 0..12 {
                out[blk_start + 4 + i] = 0x01;
            }

            // qh: high 2 bits for each quant value (all zeros for values 0-15)
            // We want quant values to be 1, so high bits are 0
            for i in 0..32 {
                out[blk_start + 16 + i] = 0x00;
            }

            // ql: each byte packs two 4-bit quant values (low nibble + high nibble).
            // 0x11 sets both nibbles to 1, so every quant value decodes as 1.
            for i in 0..128 {
                out[blk_start + 48 + i] = 0x11;
            }
        }
    }
    out
}

fn build_q6_k_payload(width: usize, n_rows: usize) -> Vec<u8> {
    assert!(
        width.is_multiple_of(256),
        "Q6_K width must be divisible by 256"
    );
    let blocks_per_row = width / 256;
    // Q6_K block: d(2) + scales(16) + ql(128) + qh(64) = 210 bytes
    let row_bytes = blocks_per_row * 210;
    let mut out = vec![0u8; row_bytes * n_rows];

    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 210;

            // d = 1.0 (f16 bits = 0x3c00)
            out[blk_start] = 0x00;
            out[blk_start + 1] = 0x3c;

            // scales: 16 bytes, all set to 1
            for i in 0..16 {
                out[blk_start + 2 + i] = 0x01;
            }

            // ql: each byte packs two 4-bit quant values.
            // 0x00 sets both nibbles to 0.
            for i in 0..128 {
                out[blk_start + 18 + i] = 0x00;
            }

            // qh: high 2 bits for each quant value (all zeros).
            // With ql=0x00 and qh=0x00, combined = 0, value = 0 - 32 = -32.
            // Output = d * scale * value = 1.0 * 1 * (-32) = -32.0 for every element.
            for i in 0..64 {
                out[blk_start + 146 + i] = 0x00;
            }
        }
    }
    out
}

fn build_q6_k_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = build_q6_k_payload(EMBEDDING_DIM, EMBEDDING_DIM);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q6_K,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn build_q5_k_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = build_q5_k_payload(EMBEDDING_DIM, EMBEDDING_DIM);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q5_K,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn stub() -> OlmoeRouter {
    OlmoeRouter::load_with_mode("", 8, 1, RoutingMode::StubUniform)
        .expect("stub load should succeed")
}

#[test]
fn test_stub_mode_loads() {
    let model = stub();
    assert!(!model.is_loaded());
    assert_eq!(model.quantization(), "stub");
}

#[test]
fn test_stub_forward_uniform_weights() {
    let mut model = stub();
    let out = model.forward(&vec![0.1; EMBEDDING_DIM]).unwrap();
    for weight in &out.expert_weights {
        assert!((*weight - 0.125).abs() < 1e-5);
    }
}

#[test]
fn test_dense_sim_uses_real_gate_weights() {
    let mut gate = vec![0.0f32; EMBEDDING_DIM * 64];
    for (expert, value) in gate.iter_mut().take(64).enumerate() {
        *value = if expert == 0 { 8.0 } else { -8.0 };
    }
    let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
    let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

    let mut model =
        OlmoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::DenseSim).unwrap();
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    embedding[0] = 1.0;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts[0], 0);
    assert_eq!(model.family(), ModelFamily::Olmoe);
    assert_eq!(model.routing_tensor_name(), "blk.0.ffn_gate_inp.weight");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_quantized_synapse_probe_uses_synthetic_fallback() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_quantized_synapse_checkpoint(gate_payload),
        "iq3-s-synapse",
    );

    let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "synthetic-fallback");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_quantized_attn_q_does_not_advertise_real_gpu_synapse_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * 4];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "quantized-attn-q");
    let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();

    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "synthetic-fallback");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_preferred_synapse_descriptor_iq3s_has_no_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_quantized_synapse_checkpoint(gate_payload),
        "iq3-s-descriptor",
    );

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for quantized attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_IQ3_S);
    assert_eq!(descriptor.ggml_type_label, "IQ3_S");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(!descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(model.synapse_source(), "synthetic-fallback");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_preferred_synapse_descriptor_f16_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_real_size_checkpoint(gate_payload), "f16-descriptor");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for F16 attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_F16);
    assert_eq!(descriptor.ggml_type_label, "F16");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(
        model.real_gpu_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "real");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_preferred_synapse_descriptor_q8_0_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_q8_0_synapse_checkpoint(gate_payload),
        "q8-0-descriptor",
    );

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for Q8_0 attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_Q8_0);
    assert_eq!(descriptor.ggml_type_label, "Q8_0");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(
        model.dequantized_q8_0_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "dequantized-q8_0");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_q8_0_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q8_0_synapse_checkpoint(gate_payload), "q8-0-probe");

    let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q8_0");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_synapse_tensor_row_major_shape_reports_q8_0_dims() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q8_0_synapse_checkpoint(gate_payload), "q8-0-shape");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();

    let shape = model
        .synapse_tensor_row_major_shape("blk.0.attn_q.weight")
        .expect("Q8_0 synapse tensor shape must be readable");
    assert_eq!(shape, (EMBEDDING_DIM, EMBEDDING_DIM));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_synapse_tensor_row_major_shape_uses_one_row_for_rank_one_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
            ("rank1.tensor", vec![7], GGML_TYPE_F16, vec![0u8; 7 * 2]),
        ],
        32,
    );
    let path = write_temp_file(&checkpoint, "rank-1-shape");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();

    let shape = model
        .synapse_tensor_row_major_shape("rank1.tensor")
        .expect("rank-1 tensor shape must be readable");
    assert_eq!(shape, (1, 7));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_synapse_tensor_row_major_shape_rejects_zero_dim_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
            ("zero-dim.tensor", vec![], GGML_TYPE_F16, Vec::new()),
        ],
        32,
    );
    let path = write_temp_file(&checkpoint, "zero-dim-shape");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();

    let err = model
        .synapse_tensor_row_major_shape("zero-dim.tensor")
        .expect_err("zero-dim tensor must be rejected");
    assert!(matches!(err, HybridError::UnsupportedFormat(_)));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_preferred_synapse_descriptor_q5_k_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_q5_k_synapse_checkpoint(gate_payload),
        "q5-k-descriptor",
    );

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for Q5_K attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_Q5_K);
    assert_eq!(descriptor.ggml_type_label, "Q5_K");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(
        model.dequantized_q5_k_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "dequantized-q5_k");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_q5_k_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q5_k_synapse_checkpoint(gate_payload), "q5-k-probe");

    let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q5_k");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_q6_k_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q6_k_synapse_checkpoint(gate_payload), "q6-k-probe");

    let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q6_k");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_q5_k_dequantize_full_tensor_succeeds() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q5_k_synapse_checkpoint(gate_payload), "q5-k-dequant");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();

    let weights = model
        .dequantized_q5_k_synapse_weights("blk.0.attn_q.weight")
        .expect("Q5_K dequantization must succeed");

    // Verify we get the expected number of elements
    assert_eq!(weights.len(), EMBEDDING_DIM * EMBEDDING_DIM);

    // Verify a few deterministic sample values from the synthetic
    // checkpoint payload so this test catches dequantization bugs such as
    // incorrect scale/min handling or nibble interpretation.
    let expected_samples = [
        (0usize, 1.0f32),
        (1usize, 1.0f32),
        (2usize, 1.0f32),
        (3usize, 1.0f32),
    ];
    for (idx, expected) in expected_samples {
        let actual = weights[idx];
        assert!(
            (actual - expected).abs() <= 1e-6,
            "unexpected dequantized value at index {idx}: expected {expected}, got {actual}"
        );
    }
    /*
    Also keep the broad sanity check that every produced value is finite.
    */
    for &v in &weights {
        assert!(v.is_finite(), "expected finite value, got {v}");
    }

    // Verify deterministic values from the known payload:
    // - The payload sets d=1.0, dmin=0.0, ql=0x11 (both nibbles = 1), qh=0x00.
    // - scale_min_k4 indices 0-3 have sc=1, so ql_chunks 0-1 (elements 0-127)
    //   produce d * 1 - 0 = 1.0.
    // - scale_min_k4 indices 4-7 have sc=0, so ql_chunks 2-3 (elements 128-255)
    //   produce 0.0 * 1 - 0 = 0.0.
    assert_eq!(weights[0], 1.0_f32, "element 0 should be 1.0");
    assert_eq!(weights[127], 1.0_f32, "element 127 should be 1.0");
    assert_eq!(weights[128], 1.0_f32, "element 128 should be 1.0");
    assert_eq!(weights[255], 1.0_f32, "element 255 should be 1.0");

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_q6_k_dequantize_full_tensor_succeeds() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q6_k_synapse_checkpoint(gate_payload), "q6-k-dequant");

    let model = OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
        .unwrap();

    let weights = model
        .dequantized_q6_k_synapse_weights("blk.0.attn_q.weight")
        .expect("Q6_K dequantization must succeed");

    // Verify we get the expected number of elements
    assert_eq!(weights.len(), EMBEDDING_DIM * EMBEDDING_DIM);

    // Verify all values are finite
    for &v in &weights {
        assert!(v.is_finite(), "expected finite value, got {v}");
    }

    // Verify deterministic values from the known payload:
    // - d=1.0, scales=1, ql=0x00, qh=0x00
    // - combined = 0, value = 0 - 32 = -32
    // - output = 1.0 * 1 * (-32) = -32.0 for every element
    assert!(
        (weights[0] - (-32.0_f32)).abs() < 1e-4,
        "element 0 should be -32.0, got {}",
        weights[0]
    );
    assert!(
        (weights[255] - (-32.0_f32)).abs() < 1e-4,
        "element 255 should be -32.0, got {}",
        weights[255]
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_synapse_tensor_row_major_shape_requires_checkpoint() {
    let model = stub();
    let err = model
        .synapse_tensor_row_major_shape("blk.0.attn_q.weight")
        .expect_err("stub router must not expose checkpoint-backed tensor shapes");
    assert!(matches!(err, HybridError::ModelLoad { .. }));
}

#[test]
fn test_ggml_type_label_covers_lineup_quants() {
    // Sanity: the labels we surface in synapse_diagnostic.json should
    // never read "unknown" for the SAAQ 1.5 lineup's known quant types.
    for &(ty, expected) in &[
        (0u32, "F32"),
        (1u32, "F16"),
        (8u32, "Q8_0"),
        (12u32, "Q4_K"),
        (13u32, "Q5_K"),
        (14u32, "Q6_K"),
        (20u32, "IQ4_NL"),
        (21u32, "IQ3_S"),
    ] {
        assert_eq!(ggml_type_label(ty), expected, "ggml_type={ty}");
    }
    assert_eq!(ggml_type_label(9999), "unknown");
    assert!(synapse_dequant_path_supported(GGML_TYPE_F16));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q8_0));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q5_K));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q6_K));
    for &ty in &[0u32, 12, 20, 21] {
        assert!(!synapse_dequant_path_supported(ty), "ggml_type={ty}");
    }
}

#[test]
fn test_spiking_sim_state_can_reset() {
    let mut model = OlmoeRouter::load_with_mode("", 8, 2, RoutingMode::SpikingSim).unwrap();
    let _ = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();
    assert!(model.has_state_activity());
    model.reset_state();
    assert!(!model.has_state_activity());
}

#[test]
fn test_real_checkpoint_probe_via_env() {
    let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
        return;
    };

    let metadata = OlmoeRouter::probe_model(&path, None).unwrap();
    assert!(!metadata.architecture.is_empty());
    assert!(metadata.hidden_size > 0);
    assert!(metadata.num_experts > 0);
    assert!(!metadata.routing_tensor_name.is_empty());
}

#[test]
fn test_ggml_type_label_covers_all_constants() {
    // Exercise every named constant through ggml_type_label
    let cases = [
        (GGML_TYPE_F32, "F32"),
        (GGML_TYPE_F16, "F16"),
        (GGML_TYPE_Q8_0, "Q8_0"),
        (GGML_TYPE_Q5_K, "Q5_K"),
        (GGML_TYPE_Q6_K, "Q6_K"),
        (GGML_TYPE_IQ3_S, "IQ3_S"),
    ];
    for (ty, expected) in cases {
        assert_eq!(ggml_type_label(ty), expected, "ggml_type={ty}");
    }
    // Unknown type
    assert_eq!(ggml_type_label(9999), "unknown");
}

#[test]
fn test_synapse_dequant_path_supported_exercises_all_named_types() {
    assert!(synapse_dequant_path_supported(GGML_TYPE_F16));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q8_0));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q5_K));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q6_K));
    assert!(!synapse_dequant_path_supported(GGML_TYPE_F32));
    assert!(!synapse_dequant_path_supported(GGML_TYPE_IQ3_S));
}

#[test]
fn test_ggml_type_label_exercises_all_match_arms() {
    // Exercise every branch in ggml_type_label for coverage
    let all_cases = [
        (0u32, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (15, "Q8_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (18, "IQ3_XXS"),
        (19, "IQ1_S"),
        (20, "IQ4_NL"),
        (21, "IQ3_S"),
        (22, "IQ2_S"),
        (23, "IQ4_XS"),
        (24, "I8"),
        (25, "I16"),
        (26, "I32"),
        (27, "I64"),
        (28, "F64"),
        (29, "IQ1_M"),
        (30, "BF16"),
    ];
    for (ty, expected) in all_cases {
        assert_eq!(ggml_type_label(ty), expected, "ggml_type={ty}");
    }
    assert_eq!(ggml_type_label(9999), "unknown");
}

#[test]
fn test_synapse_dequant_path_supported_comprehensive() {
    // All supported types
    assert!(synapse_dequant_path_supported(GGML_TYPE_F16));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q8_0));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q5_K));
    assert!(synapse_dequant_path_supported(GGML_TYPE_Q6_K));
    // Unsupported types
    assert!(!synapse_dequant_path_supported(GGML_TYPE_F32));
    assert!(!synapse_dequant_path_supported(GGML_TYPE_IQ3_S));
    assert!(!synapse_dequant_path_supported(2)); // Q4_0
    assert!(!synapse_dequant_path_supported(15)); // Q8_K
}
