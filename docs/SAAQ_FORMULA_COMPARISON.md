# SAAQ 1.0 vs SAAQ 1.5 Formula Comparison

This document compares the two SAAQ (Spiking Adaptive Activity Quantization) update
rules implemented in `src/latent.rs`.

Formula provenance was cross-checked against the external
`rmems/Surrogate_Viz.jl` repository. That Julia workbench consumes
`corinth-canal` latent telemetry for SymbolicRegression.jl discovery and
paired-run validation; it does not define simulator runtime math.

## SAAQ 1.0 — `LegacyV1_0`

**Config name:** `saaq_v1_0` (env `SAAQ_RULE=legacy` or `SAAQ_RULE=saaq_v1_0`)

**Formula:**

```
saaq_delta_q_target = 0.52 * delta_q_prev
                    + 0.28 * activity_pressure
                    + 0.12 * membrane_pressure
                    + 0.20 * routing_entropy
                    - 0.18
```

Where:
- `activity_pressure = clamp(avg_pop_firing_rate_hz / 24.0, 0.0, 1.0)`
- `membrane_pressure = clamp(membrane_dv_dt / 12.0, -1.0, 1.0)`
- `routing_entropy = normalized_entropy(expert_weights)`
- `delta_q_prev` = the previous tick's `saaq_delta_q_target`

**Characteristics:**
- Multi-term linear blend (AR(1) with three perceptual pressure terms and an offset)
- Responds to both population firing rate and membrane derivative
- Uses routing entropy from expert weights as a pressure term
- Hardcoded coefficients: 0.52, 0.28, 0.12, 0.20, -0.18

## SAAQ 1.5 — `SaaqV1_5SqrtRate`

**Config name:** `saaq_v1_5` (env `SAAQ_RULE=saaq_v1_5`, or any unrecognized value; this is the **default**)

**Formula:**

```
saaq_delta_q_target = 0.0573 * sqrt(max(avg_pop_firing_rate_hz, 0.0))
                    + 0.496 * delta_q_prev
```

**Characteristics:**
- Simpler AR(1) with sqrt-rate drive and autoregressive decay
- Only two coefficients: 0.0573 (sqrt-rate drive) and 0.496 (autoregressive decay)
- No membrane pressure term
- No routing entropy term
- No constant offset
- Uses `sqrt(firing_rate)` which compresses high firing rates and prevents runaway

**Surrogate_Viz.jl reference:** `SAAQ_latent_discovery.jl` trains symbolic
regression over the columns `avg_pop_firing_rate_hz`, `membrane_dv_dt`,
`routing_entropy`, and `saaq_delta_q_prev`, targeting `saaq_delta_q_target`.
The checked-in `hall_of_fame.csv` includes the discovered expression:

```
(sqrt(avg_pop_firing_rate_hz) * 0.05727633160985141) - (saaq_delta_q_prev / -2.015263764843582)
```

This simplifies to approximately:

```
0.05727633160985141 * sqrt(avg_pop_firing_rate_hz)
+ 0.496212... * saaq_delta_q_prev
```

which matches the Rust `SaaqV1_5SqrtRate` implementation after coefficient
rounding (`0.0573`, `0.496`).

## Key differences

| Aspect | SAAQ 1.0 (`LegacyV1_0`) | SAAQ 1.5 (`SaaqV1_5SqrtRate`) |
|--------|------------------------|-------------------------------|
| Terms | 5 (AR + 3 pressures + offset) | 2 (sqrt-rate + AR) |
| Input signals | Firing rate, membrane ΔV/dt, routing entropy | Firing rate only |
| Firing rate transform | Linear (divide by 24) | Square root |
| Offset | Yes (-0.18) | No |
| AR coefficient | 0.52 | 0.496 |
| Default? | No | **Yes** |

## Run manifest labeling

When using the SAAQ calibration example (`saaq_latent_calibration`), the run
manifest (`run_manifest.json`) includes the `saaq_rule` field set to either
`"saaq_v1_0"` or `"saaq_v1_5"`, making it clear which formula produced the
results.

The dual calibrator (`SnnDualLatentCalibrator`) always emits **both** rule
trajectories in the CSV output, regardless of which rule is designated primary:

- `saaq_delta_q_legacy_prev` / `saaq_delta_q_legacy_target` — SAAQ 1.0
- `saaq_delta_q_v15_prev` / `saaq_delta_q_v15_target` — SAAQ 1.5

The primary rule's output also populates the legacy `saaq_delta_q_prev` /
`saaq_delta_q_target` columns for backward compatibility.

## Surrogate_Viz.jl consumers

The external `Surrogate_Viz.jl` repository uses these run labels and CSV columns
as follows:

- `SAAQ_latent_discovery.jl` expects `avg_pop_firing_rate_hz`,
  `membrane_dv_dt`, `routing_entropy`, `saaq_delta_q_prev`, and
  `saaq_delta_q_target` for symbolic-regression discovery.
- `src/Surrogate_Viz.jl` prefers `saaq_delta_q_v15_target` when selecting a
  SAAQ target column, then falls back to other `saaq` + `delta` + `target`
  columns with score penalties for `legacy` names.
- `compare_saaq15_baseline_pair.jl` filters imported runs with
  `rule = "SaaqV1_5SqrtRate"` and reports the detected SAAQ delta column.
- `compare_full_lineup_saaq15.jl` uses the same `SaaqV1_5SqrtRate` rule filter
  for full-lineup paired-run dashboards.
- `SAAQ_discovery.jl` is an older raw-telemetry discovery harness. Its
  `ideal_compression_y = (gpu_power_w + cpu_package_power_w) / 400.0` target is
  not one of the SAAQ 1.0 or SAAQ 1.5 latent update formulas.

## Caveats

- Neither formula has been fitted to empirical neural data; coefficients are
  hand-chosen heuristics.
- SAAQ 1.0's membrane pressure term depends on finite-difference ΔV/dt
  estimation, which can be noisy at small `dt`.
- SAAQ 1.5's `sqrt(rate)` compresses high rates but may under-respond to
  silence (near-zero firing), causing slow convergence from cold-start.
- The `Surrogate_Viz.jl` hall-of-fame equation is discovery evidence for the
  SAAQ 1.5 coefficients, not an independent runtime source of truth.
- The formulas are **not** changed by this issue; comparison is documentation
  only.
