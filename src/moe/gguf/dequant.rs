// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Row dequantization helpers and f16 conversion for GGUF tensors.

use super::super::ggml::{GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
use crate::error::{HybridError, Result};

pub(in crate::moe) fn tensor_row_size(ggml_type: u32, width: usize) -> Result<usize> {
    match ggml_type {
        GGML_TYPE_Q8_0 => {
            if !width.is_multiple_of(32) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q8_0 tensor width {width} is not divisible by 32"
                )));
            }
            Ok((width / 32) * (2 + 32))
        }
        GGML_TYPE_Q5_K => {
            if !width.is_multiple_of(256) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q5_K tensor width {width} is not divisible by 256"
                )));
            }
            Ok((width / 256) * (2 + 2 + 12 + 32 + 128))
        }
        GGML_TYPE_Q6_K => {
            if !width.is_multiple_of(256) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q6_K tensor width {width} is not divisible by 256"
                )));
            }
            // Q6_K block: d(2) + scales(16) + ql(128) + qh(64) = 210 bytes
            Ok((width / 256) * 210)
        }
        GGML_TYPE_IQ3_M_BLOCK => {
            if !width.is_multiple_of(256) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "IQ3_M block tensor width {width} is not divisible by 256"
                )));
            }
            // IQ3_M block: d(2) + hmask(32) + qs(64) + scales(12) + scales_h(1) = 111 bytes
            Ok((width / 256) * 111)
        }
        other => Err(HybridError::UnsupportedFormat(format!(
            "row-size lookup is not implemented for ggml_type={other}"
        ))),
    }
}

pub(in crate::moe) fn dequantize_row_q8_0(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(32) {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q8_0 width {width} is not divisible by 32"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &quant in &block[2..34] {
            out.push((quant as i8) as f32 * d);
        }
    }
    Ok(out)
}

pub(in crate::moe) fn dequantize_row_q5_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(256) {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q5_K width {width} is not divisible by 256"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        for ql_chunk in ql.chunks_exact(32) {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mn1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mn2 = dmin * m2 as f32;

            for (lane, &q) in ql_chunk.iter().enumerate() {
                let qh_byte = qh[lane];
                let hi1 = if qh_byte & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh_byte & u2 != 0 { 16 } else { 0 };
                out.push(d1 * ((q & 0x0F) + hi1) as f32 - mn1);
                out.push(d2 * ((q >> 4) + hi2) as f32 - mn2);
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

pub(in crate::moe) fn dequantize_row_q6_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(256) {
        return Err(HybridError::ModelLoad {
            path: "".into(),
            reason: format!("Q6_K width {width} is not divisible by 256"),
        });
    }
    // Q6_K block: d(2) + scales(16) + ql(128) + qh(64) = 210 bytes
    // Matches the ggml block_q6_K layout from llama.cpp.
    let expected_len = (width / 256) * 210;
    if row.len() != expected_len {
        return Err(HybridError::ModelLoad {
            path: "".into(),
            reason: format!(
                "Q6_K row length mismatch: expected {expected_len} bytes for width {width}, got {}",
                row.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales = &block[2..18];
        let ql = &block[18..146];
        let qh = &block[146..210];
        // Two passes of 128 values each (ql/qh advance by 64/32 per pass).
        // Each pass reads 8 of the 16 scale entries: pass 0 uses scales[0..8],
        // pass 1 uses scales[8..16]. Within each pass, the 4 output values per
        // iteration read different scale offsets (matching ggml's sc[is + offset]).
        for pass in 0..2u8 {
            let base = (pass as usize) * 64;
            let qh_base = (pass as usize) * 32;
            let sc_pass_base = (pass as usize) * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[base + l] & 0x0F) | ((qh[qh_base + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[base + 32 + l] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[base + 32 + l] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i8 - 32;
                out.push(d * (scales[sc_pass_base + is] as i8) as f32 * q1 as f32);
                out.push(d * (scales[sc_pass_base + is + 2] as i8) as f32 * q2 as f32);
                out.push(d * (scales[sc_pass_base + is + 4] as i8) as f32 * q3 as f32);
                out.push(d * (scales[sc_pass_base + is + 6] as i8) as f32 * q4 as f32);
            }
        }
    }
    Ok(out)
}

pub(in crate::moe) fn dequantize_row_iq3_m(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(256) {
        return Err(HybridError::UnsupportedFormat(format!(
            "IQ3_M width {width} is not divisible by 256"
        )));
    }
    let expected_len = (width / 256) * 111;
    if row.len() != expected_len {
        return Err(HybridError::ModelLoad {
            path: "".into(),
            reason: format!(
                "IQ3_M row length mismatch: expected {expected_len} bytes for width {width}, got {}",
                row.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(111) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let hmask = &block[2..34]; // 32 bytes
        let qs = &block[34..98]; // 64 bytes
        let scales = &block[98..110]; // 12 bytes
        let scales_h = block[110]; // 1 byte

        // Decode 6-bit scales from 12 bytes (16 scales total, 6 bits each)
        // 12 bytes * 8 bits = 96 bits; 96 / 6 = 16 scales
        let mut sc = [0u8; 16];
        let mut bit_pos = 0usize;
        for sc_val in sc.iter_mut() {
            let byte_idx = bit_pos / 8;
            let bit_shift = bit_pos % 8;
            let mut val = if byte_idx < 12 {
                (scales[byte_idx] >> bit_shift) & 0x3F
            } else {
                0
            };
            if bit_shift > 2 && byte_idx + 1 < 12 {
                let rem = 6 - (8 - bit_shift);
                val |= (scales[byte_idx + 1] & ((1 << rem) - 1)) << (8 - bit_shift);
            }
            *sc_val = val;
            bit_pos += 6;
        }

        // scales_h provides the high 2 bits for each of the 16 scales
        // bits 0-1 -> scale[0], bits 2-3 -> scale[1], etc.
        let scales_h_u32 = scales_h as u32;
        for (i, sc_val) in sc.iter_mut().enumerate() {
            let high_bits = ((scales_h_u32 >> (i * 2)) & 0x03) as u8;
            *sc_val |= high_bits << 6;
        }

        // Unpack 3-bit values from qs (64 bytes -> 256 values, 2 values per byte with hmask)
        for i in 0..256 {
            let qs_byte = qs[i / 4];
            let qs_shift = (i % 4) * 2;
            let low_2 = (qs_byte >> qs_shift) & 0x03;

            let hmask_byte = hmask[i / 8];
            let hmask_shift = i % 8;
            let high_bit = (hmask_byte >> hmask_shift) & 0x01;

            let q = low_2 | (high_bit << 2); // 3-bit value 0..7

            // Convert to signed: 0..7 -> -4..3 (centered at 3.5)
            let q_signed = q as i8 - 4;

            let scale_idx = i / 16;
            let scale = sc[scale_idx] as f32;
            out.push(d * scale * q_signed as f32);
        }
    }
    Ok(out)
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

pub(in crate::moe) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) & 0x7C00) >> 10;
    let mant = ((bits as u32) & 0x03FF) << 13;
    let val = if exp == 0 {
        mant
    } else if exp == 31 {
        0x7F800000 | mant
    } else {
        ((exp + 127 - 15) << 23) | mant
    };
    f32::from_bits(sign | val)
}
