// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Safetensors checkpoint inspection, deterministic manifest generation, and
//! tensor loading for the router bridge.
//!
//! Submodules:
//! - extractable (for #116): `discovery`, `json`, `paths`, `manifest`, `validate`
//! - corinth-specific: `config` (HF config), `map` (mmap load / token extract)
//!
//! The `inspect_*` functions read only Safetensors headers and optional
//! Hugging Face shard index metadata.  The `MappedSafetensorsCheckpoint`
//! type memory-maps shards and extracts tensor payload bytes on demand.

use crate::error::HybridError;
// Re-export for `tests.rs` (`use super::*`).
#[allow(unused_imports)]
use std::collections::BTreeMap;
#[allow(unused_imports)]
use std::fs::{self, File};
#[allow(unused_imports)]
use std::path::{Path, PathBuf};

mod config;
mod discovery;
mod json;
mod manifest;
mod map;
mod paths;
mod validate;

// Keep helpers in parent scope so `tests.rs` (`use super::*`) resolves them.
#[allow(unused_imports)]
use discovery::{classify_tensor, discover_candidates};
// Glob-import so tests keep resolving extracted helpers (`use super::*`).
#[allow(unused_imports)]
use config::*;
#[allow(unused_imports)]
use json::*;
#[allow(unused_imports)]
use manifest::*;
#[allow(unused_imports)]
use map::*;
#[allow(unused_imports)]
use paths::*;
#[allow(unused_imports)]
use validate::*;

pub use discovery::{
    SafetensorsCandidateSummary, SafetensorsExpertGroup, SafetensorsRouterCandidate,
};
pub use manifest::{
    SafetensorsCheckpointSource, SafetensorsManifest, SafetensorsTensorRecord,
    inspect_safetensors_checkpoint, write_safetensors_manifest,
};
pub use map::{MappedSafetensorsCheckpoint, SafetensorsMetadata, bf16_to_f32};
// Crate-private helper used by `moe/tests.rs` (and map); re-export for callers.
#[allow(unused_imports)] // used outside this module via `safetensors::dtype_size_bytes`
pub(crate) use validate::dtype_size_bytes;

pub(super) fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

pub(super) fn model_load(path: &Path, reason: String) -> HybridError {
    HybridError::ModelLoad {
        path: path.display().to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests;
