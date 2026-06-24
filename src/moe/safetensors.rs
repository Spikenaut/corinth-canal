// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Safetensors checkpoint inspection, deterministic manifest generation, and
//! tensor loading for the router bridge.
//!
//! The `inspect_*` functions read only Safetensors headers and optional
//! Hugging Face shard index metadata.  The `MappedSafetensorsCheckpoint`
//! type memory-maps shards and extracts tensor payload bytes on demand.

use super::checkpoint::f16_to_f32;
use crate::error::{HybridError, Result};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use memmap2::Mmap;

const SAFETENSORS_EXTENSION: &str = "safetensors";
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const INDEX_UNREFERENCED_SHARDS_KEY: &str = "index:unreferenced_shards";
/// Unambiguous boundary between shard relative path and logical metadata key in
/// `shard:*` manifest keys (avoids ambiguity when the logical key contains `:`).
const SHARD_METADATA_KEY_SEP: char = '\u{001e}';

mod discovery;

pub use discovery::{
    SafetensorsCandidateSummary, SafetensorsExpertGroup, SafetensorsRouterCandidate,
};

use discovery::{classify_tensor, discover_candidates};

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsManifest {
    pub manifest_version: u32,
    pub format: &'static str,
    pub checkpoint: SafetensorsCheckpointSource,
    pub tensors: Vec<SafetensorsTensorRecord>,
    pub candidates: SafetensorsCandidateSummary,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsCheckpointSource {
    pub input_kind: String,
    pub index_file: Option<String>,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsTensorRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub source_shard: String,
    pub data_offsets: [usize; 2],
    pub labels: Vec<&'static str>,
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
struct ShardInspection {
    metadata: BTreeMap<String, String>,
    tensors: Vec<SafetensorsTensorRecord>,
}

struct NoDuplicateValue(Value);

/// Inspect a Safetensors file, sharded Safetensors index, or directory of
/// Safetensors shards and return a deterministic JSON-serializable manifest.
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let input = path.as_ref();
    let metadata = fs::metadata(input).map_err(|e| model_load(input, e.to_string()))?;

    if metadata.is_dir() {
        return inspect_directory(input);
    }

    if is_safetensors_index(input) {
        let root = parent_or_current(input);
        return inspect_index(root, input);
    }

    if input.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION) {
        return inspect_single_file(input);
    }

    Err(HybridError::UnsupportedFormat(format!(
        "expected .safetensors file, .safetensors.index.json file, or directory, got '{}'",
        input.display()
    )))
}

/// Write a deterministic pretty-printed Safetensors manifest to `output_path`.
pub fn write_safetensors_manifest(
    checkpoint_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<SafetensorsManifest> {
    let checkpoint_path = checkpoint_path.as_ref();
    let output_path = output_path.as_ref();
    let manifest = inspect_safetensors_checkpoint(checkpoint_path)?;
    reject_output_checkpoint_conflict(checkpoint_path, output_path, &manifest)?;
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| HybridError::ModelLoad {
        path: output_path.display().to_string(),
        reason: format!("serialize Safetensors manifest: {e}"),
    })?;
    fs::write(output_path, json).map_err(|e| HybridError::ModelLoad {
        path: output_path.display().to_string(),
        reason: e.to_string(),
    })?;
    Ok(manifest)
}

fn inspect_single_file(path: &Path) -> Result<SafetensorsManifest> {
    let root = parent_or_current(path);
    let shard = inspect_shard(path, root)?;
    Ok(build_manifest(
        "single_file",
        None,
        vec![path.to_path_buf()],
        shard.metadata,
        shard.tensors,
    ))
}

fn inspect_directory(root: &Path) -> Result<SafetensorsManifest> {
    if let Some(index_path) = find_index_file(root)? {
        return inspect_index(root, &index_path);
    }

    let shards = list_safetensors_files(root)?;
    if shards.is_empty() {
        return Err(HybridError::UnsupportedFormat(format!(
            "no .safetensors files found in '{}'",
            root.display()
        )));
    }
    inspect_shards("directory", root, None, shards)
}

fn inspect_index(root: &Path, index_path: &Path) -> Result<SafetensorsManifest> {
    let raw = read_index(index_path)?;
    let index_tensor_count = raw.weight_map.len();
    let mut expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for (tensor_name, relative) in raw.weight_map {
        let shard_path = index_shard_path(root, index_path, &relative)?;
        expected_by_shard
            .entry(shard_path)
            .or_default()
            .insert(tensor_name);
    }
    let shards = expected_by_shard.keys().cloned().collect::<Vec<_>>();

    let mut metadata = stringify_metadata("index", raw.metadata);
    metadata.remove(INDEX_UNREFERENCED_SHARDS_KEY);
    let indexed_shards = shards.iter().cloned().collect::<BTreeSet<_>>();
    let unreferenced_shards = list_safetensors_files(root)?
        .into_iter()
        .filter(|path| !indexed_shards.contains(path))
        .map(|path| relative_path(&path, root))
        .collect::<Vec<_>>();
    let unreferenced_shards_json = if unreferenced_shards.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&unreferenced_shards).map_err(|e| {
            model_load(
                index_path,
                format!("serialize {INDEX_UNREFERENCED_SHARDS_KEY} list: {e}"),
            )
        })?)
    };
    let index_file = Some(relative_path(index_path, root));
    inspect_index_shards(
        root,
        index_file,
        shards,
        expected_by_shard,
        metadata,
        index_tensor_count,
        unreferenced_shards_json,
    )
}

fn inspect_index_shards(
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>>,
    mut metadata: BTreeMap<String, String>,
    index_tensor_count: usize,
    unreferenced_shards_json: Option<String>,
) -> Result<SafetensorsManifest> {
    let mut tensors = Vec::new();
    for shard_path in &shards {
        let expected = expected_by_shard.get(shard_path).ok_or_else(|| {
            model_load(
                shard_path,
                "internal error: index shard has no expected tensor set".into(),
            )
        })?;
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );

        let found = shard
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        if let Some(missing) = expected.difference(&found).next() {
            return Err(model_load(
                shard_path,
                format!(
                    "index maps tensor '{missing}' to this shard, but the shard header does not contain it"
                ),
            ));
        }

        tensors.extend(
            shard
                .tensors
                .into_iter()
                .filter(|tensor| expected.contains(&tensor.name)),
        );
    }

    metadata.insert("index_tensor_count".into(), index_tensor_count.to_string());
    if let Some(encoded) = unreferenced_shards_json {
        metadata.insert(INDEX_UNREFERENCED_SHARDS_KEY.into(), encoded);
    }

    Ok(build_manifest(
        "hf_index", index_file, shards, metadata, tensors,
    ))
}

fn inspect_shards(
    input_kind: &str,
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
) -> Result<SafetensorsManifest> {
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for shard_path in &shards {
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );
        tensors.extend(shard.tensors);
    }

    Ok(build_manifest(
        input_kind, index_file, shards, metadata, tensors,
    ))
}

fn build_manifest(
    input_kind: &str,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    metadata: BTreeMap<String, String>,
    mut tensors: Vec<SafetensorsTensorRecord>,
) -> SafetensorsManifest {
    tensors.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.source_shard.cmp(&right.source_shard))
    });
    let candidates = discover_candidates(&tensors);

    SafetensorsManifest {
        manifest_version: 2,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: input_kind.to_string(),
            index_file,
            shard_count: shards.len(),
            tensor_count: tensors.len(),
            metadata,
        },
        tensors,
        candidates,
    }
}

fn inspect_shard(path: &Path, root: &Path) -> Result<ShardInspection> {
    let mut file = File::open(path).map_err(|e| model_load(path, e.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header length: {e}")))?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        model_load(
            path,
            format!("Safetensors header length {header_len_u64} does not fit in usize"),
        )
    })?;
    if header_len > MAX_HEADER_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
        ));
    }
    if 8u64
        .checked_add(header_len_u64)
        .is_none_or(|end| end > file_len)
    {
        return Err(model_load(
            path,
            "Safetensors header extends beyond file".into(),
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header: {e}")))?;
    let header = parse_json_rejecting_duplicate_keys(&header_bytes, path, "Safetensors header")?;
    parse_header(path, root, file_len, header_len_u64, header)
}

fn parse_header(
    path: &Path,
    root: &Path,
    file_len: u64,
    header_len: u64,
    header: Value,
) -> Result<ShardInspection> {
    let object = header
        .as_object()
        .ok_or_else(|| model_load(path, "Safetensors header must be a JSON object".to_string()))?;
    let data_len = file_len
        .checked_sub(8)
        .and_then(|len| len.checked_sub(header_len))
        .ok_or_else(|| model_load(path, "Safetensors data range underflow".into()))?;
    let source_shard = relative_path(path, root);
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for (name, value) in object {
        if name == "__metadata__" {
            metadata.extend(parse_metadata_object(path, value)?);
            continue;
        }

        let tensor = value.as_object().ok_or_else(|| {
            model_load(path, format!("tensor '{name}' metadata must be an object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing dtype")))?
            .to_string();
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing shape")))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets must have length 2"),
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid start offset")))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid end offset")))?;
        if start > end {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets are reversed"),
            ));
        }
        if end as u64 > data_len {
            return Err(model_load(
                path,
                format!("tensor '{name}' extends beyond Safetensors data section"),
            ));
        }
        let byte_size = end - start;
        let expected = expected_tensor_byte_size(&dtype, &shape, path, name)?;
        if expected != byte_size {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' byte size mismatch: shape/dtype imply {expected} bytes, data_offsets span {byte_size} bytes"
                ),
            ));
        }

        tensors.push(SafetensorsTensorRecord {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            byte_size,
            source_shard: source_shard.clone(),
            data_offsets: [start, end],
            labels: classify_tensor(name, &shape),
        });
    }

    reject_tensor_data_ranges(path, &tensors, data_len)?;

    Ok(ShardInspection { metadata, tensors })
}

fn read_index(path: &Path) -> Result<RawIndex> {
    let len = fs::metadata(path)
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    if len > MAX_INDEX_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors index is {len} bytes, exceeding limit {MAX_INDEX_BYTES}"),
        ));
    }
    let bytes = fs::read(path).map_err(|e| model_load(path, e.to_string()))?;
    let value = parse_json_rejecting_duplicate_keys(&bytes, path, "Safetensors index")?;
    serde_json::from_value(value).map_err(|e| model_load(path, format!("parse index JSON: {e}")))
}

fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = Vec::new();
    for path in read_dir_paths(root)? {
        if is_safetensors_index(&path) && is_regular_file(&path)? {
            validate_path_stays_under_root(root, &path)?;
            candidates.push(path);
        }
    }
    candidates.sort();
    if candidates.len() > 1 {
        let names = candidates
            .iter()
            .map(|path| relative_path(path, root))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(HybridError::UnsupportedFormat(format!(
            "multiple Safetensors index files found in '{}': {names}; pass the intended index file explicitly",
            root.display()
        )));
    }
    Ok(candidates.pop())
}

fn list_safetensors_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for path in read_dir_paths(root)? {
        if path.extension().and_then(|ext| ext.to_str()) != Some(SAFETENSORS_EXTENSION) {
            continue;
        }
        if !is_regular_file(&path)? {
            continue;
        }
        validate_path_stays_under_root(root, &path)?;
        shards.push(path);
    }
    shards.sort();
    Ok(shards)
}

fn is_regular_file(path: &Path) -> Result<bool> {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .map_err(|e| model_load(path, e.to_string()))
}

fn reject_tensor_data_ranges(
    path: &Path,
    tensors: &[SafetensorsTensorRecord],
    data_len: u64,
) -> Result<()> {
    let data_len = usize::try_from(data_len).map_err(|_| {
        model_load(
            path,
            format!("Safetensors data section length {data_len} does not fit in usize"),
        )
    })?;
    let mut ranges = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                tensor.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(start, end, name)| (*start, *end, *name));

    let mut expected_start = 0usize;
    let mut previous_name: Option<&str> = None;
    for (start, end, name) in ranges {
        if start < expected_start {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' data range overlaps tensor '{}'",
                    previous_name.unwrap_or("<unknown>")
                ),
            ));
        }
        if start > expected_start {
            return Err(model_load(
                path,
                format!("tensor '{name}' data range leaves a gap from {expected_start} to {start}"),
            ));
        }
        expected_start = end;
        previous_name = Some(name);
    }

    if expected_start != data_len {
        return Err(model_load(
            path,
            format!(
                "Safetensors tensor data ranges end at {expected_start}, but data section is {data_len} bytes"
            ),
        ));
    }

    Ok(())
}

fn read_dir_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| model_load(root, e.to_string()))? {
        let entry = entry.map_err(|e| model_load(root, format!("read directory entry: {e}")))?;
        paths.push(entry.path());
    }
    Ok(paths)
}

fn expected_tensor_byte_size(
    dtype: &str,
    shape: &[usize],
    path: &Path,
    tensor_name: &str,
) -> Result<usize> {
    let elements = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            model_load(
                path,
                format!("tensor '{tensor_name}' shape element count overflow"),
            )
        })
    })?;

    // Int4 is a packed format: 2 elements per byte
    if dtype == "INT4" || dtype == "I4" || dtype == "U4" {
        return elements
            .div_ceil(2)
            .checked_mul(1)
            .ok_or_else(|| model_load(path, format!("tensor '{tensor_name}' byte size overflow")));
    }

    let element_size = dtype_size_bytes(dtype).ok_or_else(|| {
        model_load(
            path,
            format!("tensor '{tensor_name}' has unsupported Safetensors dtype '{dtype}'"),
        )
    })?;
    elements
        .checked_mul(element_size)
        .ok_or_else(|| model_load(path, format!("tensor '{tensor_name}' byte size overflow")))
}

pub(super) fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "C128" => Some(16),
        "F64" | "I64" | "U64" | "C64" => Some(8),
        "F32" | "TF32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "I8" | "U8" | "BOOL" => Some(1),
        // Int4 is a packed format (2 elements per byte); callers must handle
        // packed semantics explicitly rather than using element_size * count.
        "INT4" | "I4" | "U4" => None,
        _ => None,
    }
}

fn shard_metadata_namespaced_key(shard_path: &str, key: &str) -> String {
    format!("shard:{shard_path}{SHARD_METADATA_KEY_SEP}{key}")
}

fn namespaced_shard_metadata_logical_key(namespaced: &str) -> Option<&str> {
    let rest = namespaced.strip_prefix("shard:")?;
    if let Some((_, logical_key)) = rest.split_once(SHARD_METADATA_KEY_SEP) {
        return Some(logical_key);
    }
    // Legacy keys used `shard:{path}:{key}` with a single `:` separator; this
    // remains ambiguous when `key` contains `:`, so prefer new keys above.
    rest.rsplit_once(':').map(|(_, logical_key)| logical_key)
}

fn merge_shard_metadata(
    metadata: &mut BTreeMap<String, String>,
    shard_path: &str,
    shard_metadata: BTreeMap<String, String>,
) {
    for (key, value) in shard_metadata {
        let namespaced_key = shard_metadata_namespaced_key(shard_path, &key);
        match metadata.get(&key) {
            None => {
                if !has_shard_metadata_key(metadata, &key) {
                    metadata.insert(key.clone(), value.clone());
                }
                metadata.insert(namespaced_key, value);
            }
            Some(existing) if existing == &value => {
                metadata.insert(namespaced_key, value);
            }
            Some(_) => {
                metadata.remove(&key);
                metadata.insert(namespaced_key, value);
            }
        }
    }
}

fn has_shard_metadata_key(metadata: &BTreeMap<String, String>, key: &str) -> bool {
    metadata
        .keys()
        .any(|existing| namespaced_shard_metadata_logical_key(existing).is_some_and(|k| k == key))
}

fn parse_unreferenced_shards_metadata(encoded: &str) -> Vec<String> {
    if let Ok(paths) = serde_json::from_str::<Vec<String>>(encoded) {
        return paths;
    }
    encoded
        .split(',')
        .filter(|shard| !shard.is_empty())
        .map(str::to_string)
        .collect()
}

fn reject_output_checkpoint_conflict(
    checkpoint_path: &Path,
    output_path: &Path,
    manifest: &SafetensorsManifest,
) -> Result<()> {
    let output = canonical_existing_or_parent(output_path)?;
    let input_metadata = fs::metadata(checkpoint_path)
        .map_err(|e| model_load(checkpoint_path, format!("stat checkpoint input: {e}")))?;
    let root = if input_metadata.is_dir() {
        checkpoint_path
    } else {
        parent_or_current(checkpoint_path)
    };

    let mut checkpoint_files = BTreeSet::new();
    if input_metadata.is_file() {
        checkpoint_files.insert(checkpoint_path.to_path_buf());
    }
    if let Some(index_file) = &manifest.checkpoint.index_file {
        checkpoint_files.insert(root.join(index_file));
    }
    for tensor in &manifest.tensors {
        checkpoint_files.insert(root.join(&tensor.source_shard));
    }
    if let Some(unreferenced_shards) = manifest
        .checkpoint
        .metadata
        .get(INDEX_UNREFERENCED_SHARDS_KEY)
    {
        for shard in parse_unreferenced_shards_metadata(unreferenced_shards) {
            checkpoint_files.insert(root.join(shard));
        }
    }

    for checkpoint_file in checkpoint_files {
        if paths_refer_to_same_file(&checkpoint_file, &output).map_err(|e| {
            model_load(
                &checkpoint_file,
                format!("compare checkpoint/output file identity: {e}"),
            )
        })? {
            return Err(model_load(
                output_path,
                "manifest output path must not overwrite a Safetensors checkpoint or index file"
                    .into(),
            ));
        }
    }

    Ok(())
}

fn paths_refer_to_same_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    if !left.exists() || !right.exists() {
        return Ok(false);
    }

    let left_meta = fs::metadata(left)?;
    let right_meta = fs::metadata(right)?;

    #[cfg(unix)]
    {
        Ok(left_meta.dev() == right_meta.dev() && left_meta.ino() == right_meta.ino())
    }

    #[cfg(windows)]
    {
        match (
            left_meta.volume_serial_number(),
            right_meta.volume_serial_number(),
            left_meta.file_index(),
            right_meta.file_index(),
        ) {
            (Some(vl), Some(vr), Some(il), Some(ir)) => Ok(vl == vr && il == ir),
            _ => Ok(fs::canonicalize(left)? == fs::canonicalize(right)?),
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        Ok(fs::canonicalize(left)? == fs::canonicalize(right)?)
    }
}

fn canonical_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()));
    }
    let parent = parent_or_current(path);
    let file_name = path.file_name().ok_or_else(|| {
        model_load(
            path,
            "manifest output path must include a file name".to_string(),
        )
    })?;
    let parent = fs::canonicalize(parent).map_err(|e| model_load(parent, e.to_string()))?;
    Ok(parent.join(file_name))
}

fn validate_path_stays_under_root(root: &Path, path: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(root).map_err(|e| model_load(root, e.to_string()))?;
    let canonical_path = fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(model_load(
            path,
            "Safetensors shard path must stay within the checkpoint directory".into(),
        ));
    }
    Ok(())
}

fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn is_safetensors_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

fn index_shard_path(root: &Path, index_path: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    let mut normalized = PathBuf::new();
    let escapes_root = relative_path.is_absolute()
        || relative_path.components().any(|component| match component {
            Component::Normal(part) => {
                normalized.push(part);
                false
            }
            Component::CurDir => false,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => true,
        });
    if escapes_root {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must stay within the checkpoint directory"),
        ));
    }
    if normalized.as_os_str().is_empty() {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must name a Safetensors shard"),
        ));
    }
    let candidate = root.join(normalized);
    validate_path_stays_under_root(root, &candidate)?;
    Ok(candidate)
}

fn stringify_metadata(prefix: &str, metadata: BTreeMap<String, Value>) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (format!("{prefix}:{key}"), stringify_json_value(&value)))
        .collect()
}

fn parse_json_rejecting_duplicate_keys(bytes: &[u8], path: &Path, context: &str) -> Result<Value> {
    serde_json::from_slice::<NoDuplicateValue>(bytes)
        .map(|value| value.0)
        .map_err(|e| model_load(path, format!("parse {context} JSON: {e}")))
}

fn parse_metadata_object(path: &Path, value: &Value) -> Result<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        model_load(
            path,
            "Safetensors __metadata__ must be a JSON object of strings".into(),
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                model_load(
                    path,
                    format!("Safetensors __metadata__ key '{key}' must be a string"),
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect()
}

fn stringify_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

impl<'de> Deserialize<'de> for NoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NoDuplicateValueVisitor)
    }
}

struct NoDuplicateValueVisitor;

impl<'de> Visitor<'de> for NoDuplicateValueVisitor {
    type Value = NoDuplicateValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("valid JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from_f64(value).ok_or_else(|| E::custom("invalid JSON number"))?;
        Ok(NoDuplicateValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateValue(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<NoDuplicateValue>()? {
            values.push(value.0);
        }
        Ok(NoDuplicateValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key '{key}'")));
            }
            let value = map.next_value::<NoDuplicateValue>()?;
            object.insert(key, value.0);
        }
        Ok(NoDuplicateValue(Value::Object(object)))
    }
}

fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn model_load(path: &Path, reason: String) -> HybridError {
    HybridError::ModelLoad {
        path: path.display().to_string(),
        reason,
    }
}

// ── Tensor loading backend ──────────────────────────────────────────────

/// Metadata extracted from a Hugging Face `config.json` for adapter
/// resolution.
#[derive(Debug, Clone)]
pub struct SafetensorsMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: usize,
    pub vocab_size: usize,
}

/// Memory-mapped Safetensors checkpoint with indexed tensor access.
#[derive(Debug)]
pub struct MappedSafetensorsCheckpoint {
    shards: Vec<MappedShard>,
    tensor_map: std::collections::BTreeMap<String, TensorLocation>,
    pub metadata: SafetensorsMetadata,
}

#[derive(Debug, Clone)]
struct TensorLocation {
    shard_idx: usize,
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug)]
struct MappedShard {
    // Retained for debug introspection; not read at runtime.
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Mmap,
    header_len: u64,
}

#[derive(Debug, serde::Deserialize)]
struct HfConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    vocab_size: usize,
}

fn build_indexed_shard_map(
    root: &Path,
    index_path: &Path,
) -> Result<(
    Vec<PathBuf>,
    std::collections::BTreeMap<String, TensorLocation>,
)> {
    let raw_index = read_index(index_path)?;
    let mut shard_paths: Vec<PathBuf> = Vec::new();
    let mut tensor_map: std::collections::BTreeMap<String, TensorLocation> =
        std::collections::BTreeMap::new();
    let mut shard_index_by_path: std::collections::BTreeMap<PathBuf, usize> =
        std::collections::BTreeMap::new();
    for (tensor_name, relative) in &raw_index.weight_map {
        let shard_path = index_shard_path(root, index_path, relative)?;
        let shard_idx = *shard_index_by_path
            .entry(shard_path.clone())
            .or_insert_with(|| {
                let idx = shard_paths.len();
                shard_paths.push(shard_path);
                idx
            });
        tensor_map.insert(
            tensor_name.clone(),
            TensorLocation {
                shard_idx,
                dtype: String::new(),
                shape: Vec::new(),
                data_offsets: [0, 0],
            },
        );
    }
    Ok((shard_paths, tensor_map))
}

fn build_unsharded_shard_map(
    shards: &[PathBuf],
) -> Result<(
    Vec<PathBuf>,
    std::collections::BTreeMap<String, TensorLocation>,
)> {
    let mut tensor_map: std::collections::BTreeMap<String, TensorLocation> =
        std::collections::BTreeMap::new();
    for (shard_idx, shard_path) in shards.iter().enumerate() {
        let file = File::open(shard_path).map_err(|e| model_load(shard_path, e.to_string()))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| model_load(shard_path, e.to_string()))?;
        let file_len = mmap.len() as u64;
        let mut len_bytes = [0u8; 8];
        let mut reader = &mmap[..];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| model_load(shard_path, format!("read header length: {e}")))?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_BYTES as u64 {
            return Err(model_load(
                shard_path,
                format!("header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
            ));
        }
        if 8u64
            .checked_add(header_len)
            .is_none_or(|end| end > file_len)
        {
            return Err(model_load(shard_path, "header extends beyond file".into()));
        }
        let header_bytes = &mmap[8..8 + header_len as usize];
        let header = parse_json_rejecting_duplicate_keys(header_bytes, shard_path, "header")?;
        let object = header.as_object().ok_or_else(|| {
            model_load(
                shard_path,
                "Safetensors header must be a JSON object".into(),
            )
        })?;
        for (name, _value) in object {
            if name == "__metadata__" {
                continue;
            }
            tensor_map.insert(
                name.clone(),
                TensorLocation {
                    shard_idx,
                    dtype: String::new(),
                    shape: Vec::new(),
                    data_offsets: [0, 0],
                },
            );
        }
    }
    Ok((shards.to_vec(), tensor_map))
}

impl MappedSafetensorsCheckpoint {
    /// Map a Safetensors directory containing `config.json` and either
    /// `model.safetensors.index.json` (sharded) or `model.safetensors`
    /// (single-file).
    pub fn from_directory(path: &str) -> Result<Self> {
        let root = Path::new(path);
        if !root.is_dir() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "path is not a directory".into(),
            });
        }

        let config = Self::read_config(root)?;

        // Try index-first (sharded) layout; fall back to single-file
        let (shard_paths, mut tensor_map) = if let Some(index_path) = find_index_file(root)? {
            build_indexed_shard_map(root, &index_path)?
        } else {
            let shards = list_safetensors_files(root)?;
            if shards.is_empty() {
                return Err(HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: "no .safetensors files found".into(),
                });
            }
            build_unsharded_shard_map(&shards)?
        };

        // Map each shard and parse its header to resolve tensor dtypes/offsets
        let mut shards: Vec<MappedShard> = Vec::new();
        for (shard_idx, shard_path) in shard_paths.iter().enumerate() {
            let file = File::open(shard_path).map_err(|e| model_load(shard_path, e.to_string()))?;
            let mmap =
                unsafe { Mmap::map(&file) }.map_err(|e| model_load(shard_path, e.to_string()))?;
            let file_len = mmap.len() as u64;
            let mut len_bytes = [0u8; 8];
            let mut reader = &mmap[..];
            reader
                .read_exact(&mut len_bytes)
                .map_err(|e| model_load(shard_path, format!("read header length: {e}")))?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len > MAX_HEADER_BYTES as u64 {
                return Err(model_load(
                    shard_path,
                    format!("header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
                ));
            }
            if 8u64
                .checked_add(header_len)
                .is_none_or(|end| end > file_len)
            {
                return Err(model_load(shard_path, "header extends beyond file".into()));
            }
            let header_bytes = &mmap[8..8 + header_len as usize];
            let header = parse_json_rejecting_duplicate_keys(header_bytes, shard_path, "header")?;
            let object = header.as_object().ok_or_else(|| {
                model_load(
                    shard_path,
                    "Safetensors header must be a JSON object".into(),
                )
            })?;

            let data_len = file_len
                .checked_sub(8)
                .and_then(|len| len.checked_sub(header_len))
                .ok_or_else(|| model_load(shard_path, "data range underflow".into()))?;

            for (name, value) in object {
                if name == "__metadata__" {
                    continue;
                }
                let Some(loc) = tensor_map.get_mut(name) else {
                    continue;
                };
                if loc.shard_idx != shard_idx {
                    continue;
                }
                let tensor = value.as_object().ok_or_else(|| {
                    model_load(
                        shard_path,
                        format!("tensor '{name}' metadata must be an object"),
                    )
                })?;
                let dtype = tensor
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing dtype"))
                    })?
                    .to_string();
                let shape = tensor
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing shape"))
                    })?
                    .iter()
                    .map(|dim| {
                        dim.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| {
                                model_load(shard_path, format!("tensor '{name}' invalid shape"))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let offsets = tensor
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing data_offsets"))
                    })?;
                if offsets.len() != 2 {
                    return Err(model_load(
                        shard_path,
                        format!("tensor '{name}' data_offsets must have length 2"),
                    ));
                }
                let start = offsets[0]
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' invalid start offset"))
                    })?;
                let end = offsets[1]
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' invalid end offset"))
                    })?;
                if start > end || end as u64 > data_len {
                    return Err(model_load(
                        shard_path,
                        format!("tensor '{name}' data range invalid"),
                    ));
                }
                let expected = expected_tensor_byte_size(&dtype, &shape, shard_path, name)?;
                if expected != end - start {
                    return Err(model_load(
                        shard_path,
                        format!(
                            "tensor '{name}' byte size mismatch: expected {expected}, got {}",
                            end - start
                        ),
                    ));
                }
                loc.dtype = dtype;
                loc.shape = shape;
                loc.data_offsets = [start, end];
            }

            shards.push(MappedShard {
                path: shard_path.clone(),
                mmap,
                header_len,
            });
        }

        // Validate that all tensors listed in weight_map were found in their declared shard
        for (name, loc) in &tensor_map {
            if loc.dtype.is_empty() {
                return Err(model_load(
                    &shard_paths[loc.shard_idx],
                    format!("tensor '{name}' listed in weight_map but missing from shard header"),
                ));
            }
        }

        let metadata = SafetensorsMetadata {
            architecture: config.architectures.first().cloned().unwrap_or_default(),
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_experts: config.num_experts.unwrap_or(1),
            expert_used_count: config.num_experts_per_tok.unwrap_or(1),
            vocab_size: config.vocab_size,
        };

        Ok(Self {
            shards,
            tensor_map,
            metadata,
        })
    }

    /// Extract a full tensor as `Vec<f32>`, converting BF16/F16 as needed.
    pub fn extract_tensor_f32(&self, name: &str, path: &str) -> Result<Vec<f32>> {
        let loc = self
            .tensor_map
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })?;
        let shard = &self.shards[loc.shard_idx];
        let data_start = 8 + shard.header_len as usize + loc.data_offsets[0];
        let data_end = 8 + shard.header_len as usize + loc.data_offsets[1];
        let bytes = &shard.mmap[data_start..data_end];

        match loc.dtype.as_str() {
            "F32" => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' F32 data is not 4-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            "F16" => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' F16 data is not 2-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            "BF16" => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' BF16 data is not 2-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            "INT4" | "I4" | "U4" => {
                // Int4: packed format, 2 elements per byte.
                // Compute exact expected element count from shape.
                let expected_elements = loc.shape.iter().product::<usize>();
                let mut out = Vec::with_capacity(expected_elements);
                let is_signed = loc.dtype == "I4";
                for &byte in bytes {
                    let low = byte & 0x0F;
                    let high = byte >> 4;
                    if is_signed {
                        // Sign-extend 4-bit to 8-bit, then convert to f32
                        out.push(((low as i8) << 4 >> 4) as f32);
                        if out.len() < expected_elements {
                            out.push(((high as i8) << 4 >> 4) as f32);
                        }
                    } else {
                        out.push(low as f32);
                        if out.len() < expected_elements {
                            out.push(high as f32);
                        }
                    }
                }
                Ok(out)
            }
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has unsupported Safetensors dtype '{other}'"
            ))),
        }
    }

    /// Extract a single row (token embedding) as `Vec<f32>`.
    pub fn extract_token_embedding(
        &self,
        name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let loc = self
            .tensor_map
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })?;
        if loc.shape.len() != 2 {
            return Err(HybridError::UnsupportedFormat(format!(
                "token embedding tensor '{name}' must be rank-2, got {:?}",
                loc.shape
            )));
        }
        let d0 = loc.shape[0];
        let d1 = loc.shape[1];
        if token_id >= d0 {
            return Err(HybridError::InputLengthMismatch {
                expected: d0,
                got: token_id,
            });
        }
        let row_elements = d1;
        let row_bytes = if loc.dtype == "INT4" || loc.dtype == "I4" || loc.dtype == "U4" {
            // Int4: packed format, 2 elements per byte
            row_elements.div_ceil(2)
        } else {
            let element_size = dtype_size_bytes(&loc.dtype).unwrap_or(2);
            row_elements * element_size
        };
        let shard = &self.shards[loc.shard_idx];
        let data_start = 8 + shard.header_len as usize + loc.data_offsets[0];
        let row_start = data_start + token_id * row_bytes;
        let row_end = row_start + row_bytes;
        let bytes = &shard.mmap[row_start..row_end];

        match loc.dtype.as_str() {
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            "INT4" | "I4" | "U4" => {
                let mut out = Vec::with_capacity(row_elements);
                let is_signed = loc.dtype == "I4";
                for &byte in bytes {
                    let low = byte & 0x0F;
                    let high = byte >> 4;
                    if is_signed {
                        out.push(((low as i8) << 4 >> 4) as f32);
                        if out.len() < row_elements {
                            out.push(((high as i8) << 4 >> 4) as f32);
                        }
                    } else {
                        out.push(low as f32);
                        if out.len() < row_elements {
                            out.push(high as f32);
                        }
                    }
                }
                Ok(out)
            }
            other => Err(HybridError::UnsupportedFormat(format!(
                "token embedding tensor '{name}' has unsupported dtype '{other}'"
            ))),
        }
    }

    pub fn tensor_info(&self, name: &str) -> Option<(&str, &[usize])> {
        self.tensor_map
            .get(name)
            .map(|loc| (loc.dtype.as_str(), loc.shape.as_slice()))
    }

    fn read_config(root: &Path) -> Result<HfConfig> {
        let config_path = root.join("config.json");
        let bytes = fs::read(&config_path).map_err(|e| model_load(&config_path, e.to_string()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| model_load(&config_path, format!("parse config.json: {e}")))
    }
}

/// Convert a BF16 bit pattern to `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests;
