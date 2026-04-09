# corinth-canal

Neuromorphic-ANN hybrid framework implemented in this repository.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` provides a hybrid pipeline with:
- telemetry-to-spike encoding
- spiking network simulation
- dense/spiking projection into embedding space
- frozen OLMoE-style routing simulation
- optional training signal publishing via ZMQ

## Architecture

```text
TelemetrySnapshot
       │
       ▼  Encoder
 [f32; 16] Poisson stimuli
       │
       ▼  SpikingNetwork × snn_steps
 spike_train + membrane_potentials
       │
       ▼  Projector
 dense embedding [2048]
       │
       ▼  OLMoE (frozen/simulated)
 expert_weights + selected_experts + hidden
       │
       ▼  Optional spine publish
 TrainSignal
```

## Quick start

```rust
use corinth_canal::{HybridConfig, HybridModel};
use spikenaut_telemetry::TelemetrySnapshot;

let cfg = HybridConfig::default();
let mut model = HybridModel::new(cfg)?;

let snap = TelemetrySnapshot::default();
let output = model.forward(&snap)?;

println!("Selected experts: {:?}", output.selected_experts);
```

## Features

| Feature | Effect |
|---------|--------|
| `gguf` | GGUF model header parsing support |
| `safetensors` | safetensors model loading support |
| `spine-zmq` | ZMQ publishing support for train signals |

## Run example

```bash
cargo run --example telemetry_bridge
```

With GGUF feature enabled:

```bash
OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf \
  cargo run --example telemetry_bridge --features gguf --release
```

## Project layout

| Path | Responsibility |
|------|----------------|
| `Cargo.toml` | crate config and dependencies |
| `src/lib.rs` | public API and re-exports |
| `src/types.rs` | `HybridConfig`, `HybridOutput`, enums/constants |
| `src/error.rs` | `HybridError`, `Result` |
| `src/tensor/mod.rs` | candle-free tensor utilities |
| `src/transformer/mod.rs` | transformer helpers |
| `src/hybrid/mod.rs` | hybrid module wiring |
| `src/hybrid/projector.rs` | 2-bit spiking projector logic |
| `src/hybrid/olmoe.rs` | GGUF loader and OLMoE simulation |
| `src/hybrid/hybrid.rs` | top-level orchestrator (`HybridModel`) |
| `examples/telemetry_bridge.rs` | end-to-end example |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
