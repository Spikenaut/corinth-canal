// SPDX-License-Identifier: Apache-2.0 OR MIT
//! GGUF header/metadata parsing and cursor helpers.

use super::super::ggml::{
    GGUF_MAGIC, GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32,
    GGUF_VALUE_TYPE_FLOAT64, GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32,
    GGUF_VALUE_TYPE_INT64, GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16,
    GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64, GGUF_VERSION,
};
use crate::error::{HybridError, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(in crate::moe) struct GgufTensorInfo {
    pub(in crate::moe) dims: Vec<usize>,
    pub(in crate::moe) ggml_type: u32,
    pub(in crate::moe) relative_offset: usize,
    pub(in crate::moe) absolute_offset: usize,
    pub(in crate::moe) n_elements: usize,
}

pub(in crate::moe) struct ParsedCheckpointLayout {
    pub(in crate::moe) metadata: GgufMetadata,
    pub(in crate::moe) tensors: HashMap<String, GgufTensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub(in crate::moe) struct GgufMetadata {
    architecture: String,
    quantization: String,
    numerics: HashMap<String, u64>,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl GgufMetadata {
    pub(in crate::moe) fn architecture(&self) -> &str {
        &self.architecture
    }

    pub(in crate::moe) fn quantization(&self) -> &str {
        &self.quantization
    }

    pub(in crate::moe) fn numeric(&self, key: &str) -> Option<usize> {
        self.numerics.get(key).copied().map(|v| v as usize)
    }
}

pub(in crate::moe) fn parse_checkpoint_layout(
    bytes: &[u8],
    path: &str,
) -> Result<ParsedCheckpointLayout> {
    let mut cursor = GgufCursor::new(bytes);

    let magic = cursor.read_exact(4, path)?;
    if magic != GGUF_MAGIC {
        return Err(HybridError::UnsupportedFormat(format!(
            "unrecognised model magic bytes: {magic:?}"
        )));
    }

    let version = cursor.read_u32(path)?;
    if version != GGUF_VERSION {
        return Err(HybridError::UnsupportedFormat(format!(
            "unsupported GGUF version {version}; expected {GGUF_VERSION}"
        )));
    }

    // Sanity-bound the header counts to prevent OOM allocation from malformed files.
    const MAX_TENSOR_COUNT: usize = 100_000;
    const MAX_KV_COUNT: usize = 100_000;
    const MAX_TENSOR_DIMS: usize = 8;

    let tensor_count_raw = cursor.read_u64(path)?;
    if tensor_count_raw > MAX_TENSOR_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor_count {tensor_count_raw} exceeds maximum allowed {MAX_TENSOR_COUNT}"
        )));
    }
    let tensor_count = tensor_count_raw as usize;

    let kv_count_raw = cursor.read_u64(path)?;
    if kv_count_raw > MAX_KV_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "kv_count {kv_count_raw} exceeds maximum allowed {MAX_KV_COUNT}"
        )));
    }
    let kv_count = kv_count_raw as usize;

    let mut alignment = 32usize;
    let mut file_type = None;
    let mut architecture = None;
    let mut numerics = HashMap::new();

    for _ in 0..kv_count {
        let key = cursor.read_string(path)?;
        let value_type = cursor.read_u32(path)?;
        match key.as_str() {
            "general.alignment" => alignment = cursor.read_numeric_as_usize(value_type, path)?,
            "general.file_type" => {
                let value = cursor.read_numeric_as_u32(value_type, path)?;
                file_type = Some(value);
                numerics.insert(key, value as u64);
            }
            "general.architecture" => {
                let value = cursor.read_string(path)?;
                architecture = Some(value);
            }
            _ => {
                if let Some(value) = cursor.read_numeric_value(value_type, path)? {
                    numerics.insert(key, value);
                } else if value_type == GGUF_VALUE_TYPE_STRING {
                    let _ = cursor.read_string(path)?;
                } else {
                    cursor.skip_value(value_type, path)?;
                }
            }
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string(path)?;
        let n_dims_raw = cursor.read_u32(path)? as usize;
        if n_dims_raw > MAX_TENSOR_DIMS {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has {n_dims_raw} dims, which exceeds maximum allowed {MAX_TENSOR_DIMS}"
            )));
        }
        let n_dims = n_dims_raw;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64(path)? as usize);
        }
        let ggml_type = cursor.read_u32(path)?;
        let relative_offset = cursor.read_u64(path)? as usize;
        let n_elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow"),
            })?;
        tensors.insert(
            name,
            GgufTensorInfo {
                dims,
                ggml_type,
                relative_offset,
                absolute_offset: 0,
                n_elements,
            },
        );
    }

    let tensor_data_offset = align_up(cursor.offset, alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }

    Ok(ParsedCheckpointLayout {
        metadata: GgufMetadata {
            architecture: architecture.unwrap_or_else(|| "unknown".into()),
            quantization: quantization_label(file_type),
            numerics,
        },
        tensors,
    })
}

fn quantization_label(file_type: Option<u32>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(other) => format!("GGUF({other})"),
        _ => "GGUF".into(),
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

impl<'a> GgufCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize, path: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "cursor overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "unexpected EOF while parsing GGUF".into(),
            });
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self, path: &str) -> Result<u8> {
        Ok(self.read_exact(1, path)?[0])
    }

    fn read_u16(&mut self, path: &str) -> Result<u16> {
        let bytes = self.read_exact(2, path)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self, path: &str) -> Result<u32> {
        let bytes = self.read_exact(4, path)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_u64(&mut self, path: &str) -> Result<u64> {
        let bytes = self.read_exact(8, path)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_i16(&mut self, path: &str) -> Result<i16> {
        Ok(self.read_u16(path)? as i16)
    }

    fn read_i32(&mut self, path: &str) -> Result<i32> {
        Ok(self.read_u32(path)? as i32)
    }

    fn read_i64(&mut self, path: &str) -> Result<i64> {
        Ok(self.read_u64(path)? as i64)
    }

    fn read_string(&mut self, path: &str) -> Result<String> {
        let len = self.read_u64(path)? as usize;
        let bytes = self.read_exact(len, path)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("invalid UTF-8 in GGUF string: {e}"),
        })
    }

    fn read_numeric_as_u32(&mut self, value_type: u32, path: &str) -> Result<u32> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(self.read_u8(path)? as u32),
            GGUF_VALUE_TYPE_INT8 => Ok(self.read_u8(path)? as i8 as i32 as u32),
            GGUF_VALUE_TYPE_UINT16 => Ok(self.read_u16(path)? as u32),
            GGUF_VALUE_TYPE_INT16 => Ok(self.read_i16(path)? as i32 as u32),
            GGUF_VALUE_TYPE_UINT32 => self.read_u32(path),
            GGUF_VALUE_TYPE_INT32 => Ok(self.read_i32(path)? as u32),
            GGUF_VALUE_TYPE_UINT64 => Ok(self.read_u64(path)? as u32),
            GGUF_VALUE_TYPE_INT64 => Ok(self.read_i64(path)? as u32),
            _ => Err(HybridError::UnsupportedFormat(format!(
                "GGUF numeric conversion from type {value_type} is not supported"
            ))),
        }
    }

    fn read_numeric_as_usize(&mut self, value_type: u32, path: &str) -> Result<usize> {
        Ok(self.read_numeric_as_u32(value_type, path)? as usize)
    }

    fn read_numeric_value(&mut self, value_type: u32, path: &str) -> Result<Option<u64>> {
        let value = match value_type {
            GGUF_VALUE_TYPE_UINT8 => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_INT8 => Some(self.read_u8(path)? as i8 as i64 as u64),
            GGUF_VALUE_TYPE_UINT16 => Some(self.read_u16(path)? as u64),
            GGUF_VALUE_TYPE_INT16 => Some(self.read_i16(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_INT32 => Some(self.read_i32(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_INT64 => Some(self.read_i64(path)? as u64),
            GGUF_VALUE_TYPE_BOOL => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_STRING | GGUF_VALUE_TYPE_ARRAY => None,
            other => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {other}"
                )));
            }
        };
        Ok(value)
    }

    fn skip_value(&mut self, value_type: u32, path: &str) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1, path)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2, path)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4, path)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8, path)?;
            }
            GGUF_VALUE_TYPE_STRING => {
                let _ = self.read_string(path)?;
            }
            GGUF_VALUE_TYPE_ARRAY => {
                let nested_type = self.read_u32(path)?;
                let len = self.read_u64(path)? as usize;
                for _ in 0..len {
                    self.skip_value(nested_type, path)?;
                }
            }
            _ => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {value_type}"
                )));
            }
        }
        Ok(())
    }
}
