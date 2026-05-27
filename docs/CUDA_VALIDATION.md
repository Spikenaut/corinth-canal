# CUDA Validation Ladder

This page records the focused CUDA validation path for SAAQ experiment runs.
Use it to separate host or GPU runtime defects from model, routing, and SAAQ
behavior before treating validation artifacts as meaningful.

Do not commit generated profiler artifacts, model weights, private telemetry, or
machine-local paths. Store local outputs under `artifacts/cuda-validation/` and
summarize only the pass/fail result, tool versions, and artifact filenames in PR
or issue comments.

## CUDA Entry Points

Current SAAQ and model-routing GPU paths use these repo entry points:

| Layer | Entry point | Purpose |
|-------|-------------|---------|
| SAAQ runner | `examples/saaq_latent_calibration.rs` | Creates `GpuAccelerator`, prepares GPU temporal state, ticks GPU temporal loop, downloads spikes and membrane state for latent calibration. |
| GPU temporal model API | `Model::prepare_gpu_temporal` | Allocates resident temporal buffers and uploads selected synapse weights. |
| GPU temporal model API | `Model::tick_gpu_temporal` | Runs one explicit-input GIF temporal tick and returns the on-device best walker. |
| GPU temporal model API | `Model::forward_gpu_temporal` | Projects telemetry on GPU, advances GIF state, builds a `ModelOutput`, and appends GPU routing telemetry. |
| Accelerator wrapper | `GpuAccelerator::project_snapshot_current` | Projects a `TelemetrySnapshot` into the resident GPU input buffer. |
| Accelerator wrapper | `GpuAccelerator::gif_step_weighted_tick` | Launches the weighted GIF temporal step. |
| Accelerator wrapper | `GpuAccelerator::saaq_find_best_walker` | Runs the SAAQ on-device reduction. |
| CUDA kernel | `project_snapshot_current` | GPU telemetry projection. |
| CUDA kernel | `gif_step_weighted` | F32 weighted GIF temporal step. |
| CUDA C ABI shim | `gif_step_weighted_f16` | F16 weighted GIF temporal step for GGUF-backed synapse sources. |
| CUDA C ABI shim | `saaq_find_best_walker` | Best-walker reduction used by the GPU smoke and SAAQ paths. |

The normal `saaq_latent_calibration` loop uses `Model::tick_gpu_temporal`, so it
does not append `snn_gpu_routing_telemetry.csv` by itself. `Model::forward_gpu_temporal`
is the GPU path that appends routing telemetry rows through `ModelConfig::gpu_routing_telemetry_path`.

## Tier 0: Host/GPU Sanity

Purpose: prove the machine has a visible GPU, usable NVIDIA driver, and basic
CUDA runtime/toolkit health before blaming `corinth-canal`.

Commands:

```bash
nvidia-smi

# Optional when CUDA samples are installed.
deviceQuery
bandwidthTest
```

Expected outcome:

- GPU is visible.
- Driver and CUDA runtime versions are visible.
- CUDA sample or sanity command can execute.
- Any host/toolkit failure is documented before repo-level checks.

## Tier 1: Repo-Native Validation

Purpose: prove the repo builds, CPU fallback tests remain safe, and the first
GPU execution gate can run against a real GGUF checkpoint.

Commands:

```bash
cargo check --all-targets
cargo test --no-default-features
cargo test --workspace
cargo check --example saaq_latent_calibration

# Full default smoke: 10,000 direct GPU ticks.
GGUF_CHECKPOINT_PATH=/path/to/model.gguf cargo run --release --example gpu_smoke_test

# Short deterministic smoke for fast validation or sanitizer setup.
GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  cargo run --release --example gpu_smoke_test
```

`examples/gpu_smoke_test.rs` validates the resident GPU temporal state on the
first tick, every 1,000 ticks, and the final tick. It checks that:

- `best_walker` is within the target-neuron range.
- downloaded spike output length equals the target-neuron count.
- downloaded spikes are binary `0` or `1`.
- downloaded membrane length equals the target-neuron count.
- downloaded membrane values are finite.

The raw membrane buffer is not required to be in `[0, 1]`; downstream projection
clamps GPU membrane values before using them as model potentials.

When CUDA is unavailable, GPU-specific paths fail fast with `GpuError::NoGpu`.
CPU-only validation remains covered by `cargo check --all-targets --no-default-features`
and `cargo test --no-default-features`.

## Tier 2: Compute Sanitizer Correctness

Purpose: catch memory access defects, synchronization hazards, and shared-memory
race hazards before interpreting SAAQ outputs.

Commands:

```bash
mkdir -p artifacts/cuda-validation

GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  compute-sanitizer --tool memcheck \
  --log-file artifacts/cuda-validation/memcheck.log \
  cargo run --release --example gpu_smoke_test

GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  compute-sanitizer --tool synccheck \
  --log-file artifacts/cuda-validation/synccheck.log \
  cargo run --release --example gpu_smoke_test

GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  compute-sanitizer --tool racecheck \
  --log-file artifacts/cuda-validation/racecheck.log \
  cargo run --release --example gpu_smoke_test
```

Expected outcome:

- Memcheck reports no invalid global/local/shared memory access.
- Synccheck reports no synchronization hazards.
- Racecheck reports no shared-memory race hazards for the smoke path.
- Any sanitizer finding becomes a focused follow-up issue for the specific kernel.

Reference: NVIDIA Compute Sanitizer documentation:
<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>

## Tier 3: Nsight Systems Timeline

Purpose: verify launch order, CPU/GPU gaps, synchronization points, memory copies,
and whether GPU work actually occurs during the SAAQ path.

Command:

```bash
mkdir -p artifacts/cuda-validation

GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  nsys profile -o artifacts/cuda-validation/nsys_gpu_smoke \
  cargo run --release --example gpu_smoke_test
```

Expected outcome:

- Timeline artifact is generated under `artifacts/cuda-validation/`.
- Kernel launches, host gaps, synchronization points, and copy behavior are visible.
- This tier is timeline validation only, not micro-optimization.

Reference: NVIDIA Nsight Systems: <https://developer.nvidia.com/nsight-systems>

## Tier 4: Nsight Compute Per-Kernel Profile

Purpose: profile individual kernels only after Tier 1 and Tier 2 are clean.

Command:

```bash
mkdir -p artifacts/cuda-validation

GPU_SMOKE_TICKS=1 GGUF_CHECKPOINT_PATH=/path/to/model.gguf \
  ncu --set full -o artifacts/cuda-validation/ncu_gpu_smoke \
  cargo run --release --example gpu_smoke_test
```

Expected outcome:

- Per-kernel report is generated under `artifacts/cuda-validation/`.
- Bottlenecks can be inspected for selected kernels.
- Optimization work is deferred to follow-up issues unless profiling exposes a correctness blocker.

Reference: NVIDIA Nsight Compute: <https://developer.nvidia.com/nsight-compute>

## Tier 5: DCGM Hardware/Cloud Diagnostics

Purpose: validate GPU health and hardware or cloud stability before attributing
weird SAAQ behavior to model or kernel logic.

Commands:

```bash
dcgmi diag -r 1

# Deeper validation when supported by the host/cloud image.
dcgmi diag -r 3
```

Expected outcome:

- Hardware diagnostics pass, or failures are documented.
- Cloud GPU instability, PCIe/device issues, memory errors, thermal issues, and
  power issues are separated from `corinth-canal` behavior.

Reference: NVIDIA DCGM diagnostics:
<https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html>

## PR/Issue Evidence Checklist

Record the following in PR or issue comments when running the ladder:

- Git SHA and branch.
- GPU model and driver/runtime versions from Tier 0.
- Exact Tier 1 command and pass/fail result.
- Compute Sanitizer tool names and log filenames for Tier 2.
- Nsight artifact filenames for Tier 3 and Tier 4 when collected.
- DCGM diagnostic level and pass/fail result for Tier 5 when collected.
- Follow-up issue links for any kernel-specific sanitizer or hardware finding.
