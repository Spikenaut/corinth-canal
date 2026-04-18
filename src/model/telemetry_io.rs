//! Shared CSV telemetry helpers for GPU routing outputs.

use super::core::GPU_ROUTING_TELEMETRY_HEADER;
use crate::gpu::{GpuError, GpuResult};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

pub(super) fn append_gpu_routing_telemetry_row(
    path: &Path,
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> GpuResult<()> {
    let needs_header = !path.exists()
        || std::fs::metadata(path)
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(true);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;

    if needs_header {
        writeln!(file, "{GPU_ROUTING_TELEMETRY_HEADER}")
            .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;
    }

    writeln!(
        file,
        "{token_idx},{best_score},{best_walker},{spike_count},{mean_adaptation:.4},{active_fraction:.4}"
    )
    .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;

    Ok(())
}
