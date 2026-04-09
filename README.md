# corinth-canal

**Neuromorphic-ANN hybrid framework** — the high-level orchestrator for the
SpikeLMo fusion between the Spikenaut event-driven SNN and
[OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct).

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

---

## Origin

This repository originated from the `spikenaut-hybrid` codebase and was reorganized into `corinth-canal` with a consolidated `src/hybrid`, `src/tensor`, and `src/transformer` structure.

## Architecture

```
TelemetrySnapshot (spikenaut-telemetry)
       │
       ▼  spikenaut-encoder
 NeuromodSensoryEncoder  →  [f32; 16] Poisson stimuli
       │
       ▼  neuromod
 SpikingNetwork × snn_steps  →  spike_train + membrane_potentials
       │
       ▼  corinth-canal :: Projector
 dense embedding  [DIM = 2048]
       │
       ▼  corinth-canal :: OLMoE  (frozen)
 OlmoeOutput { expert_weights, selected_experts, hidden }
       │
       ▼  spikenaut-spine  (optional, spine-zmq feature)
 TrainSignal  ──►  SpikenautDistill.jl  (E-prop on SNN only)
```

## Quick start

```rust
use corinth_canal::{HybridConfig, HybridModel};
use spikenaut_telemetry::TelemetrySnapshot;

// Stub mode (no GGUF download needed)
let cfg = HybridConfig::default();
let mut model = HybridModel::new(cfg)?;

let snap = TelemetrySnapshot::default();
let output = model.forward(&snap)?;

println!("Selected experts: {:?}", output.selected_experts);
```

With a real OLMoE checkpoint:

```rust
let cfg = HybridConfig {
    olmoe_model_path: "/models/OLMoE-1B-7B-Q5_K_M.gguf".into(),
    snn_steps: 20,
    num_experts: 8,
    top_k_experts: 1,
    ..Default::default()
};
let mut model = HybridModel::new(cfg)?;
```

## Feature flags

| Feature | What it enables |
|---------|----------------|
| `gguf` | GGUF Q5_K_M model parsing (pure Rust, no C++ llama.cpp) |
| `safetensors` | BF16 `.safetensors` shard loading |
| `spine-zmq` | ZMQ transport for Rust ↔ Julia training bridge |

Add to `Cargo.toml`:

```toml
[dependencies]
corinth-canal = { version = "0.1", features = ["gguf", "spine-zmq"] }
```

## Running the demo

```bash
# Stub mode — no checkpoint needed
cargo run --example telemetry_bridge

# With a real GGUF model
OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf \
  cargo run --example telemetry_bridge --features gguf --release
```

## Crate layout

| Source file | Responsibility |
|-------------|---------------|
| `src/lib.rs` | Public API, re-exports, crate-level docs |
| `src/types.rs` | `HybridConfig`, `HybridOutput`, `ProjectionMode`, `TrainSignal` |
| `src/error.rs` | `HybridError`, `Result` |
| `src/tensor/mod.rs` | Candle-free tensor math primitives |
| `src/transformer/mod.rs` | Core transformer helpers for OLMoE/Grok paths |
| `src/hybrid/mod.rs` | Hybrid switchboard + re-exports |
| `src/hybrid/projector.rs` | 2-bit spiking logic projector |
| `src/hybrid/olmoe.rs` | GGUF loader + SpikingSim logic |
| `src/hybrid/hybrid.rs` | Top-level orchestrator implementation |
| `examples/telemetry_bridge.rs` | End-to-end telemetry bridge demo |

## Pipeline data flow

```
HybridModel::forward(&TelemetrySnapshot)
   ├─ snapshot_to_stimuli()          → [f32; 8]  normalised channels
   ├─ NeuromodSensoryEncoder         → [f32; 16] bear/bull Poisson rates
   ├─ SpikingNetwork::step() × N     → Vec<Vec<usize>> spike train
   ├─ Projector::project()           → Vec<f32>  [2048] embedding
   └─ OLMoE::forward()               → OlmoeOutput
         ├─ expert_weights: Vec<f32> [8]
         ├─ selected_experts: Vec<usize>
         └─ hidden: Vec<f32> [2048]

HybridModel::train_step(&snap, &target)
   ├─ forward()                      (as above)
   ├─ MSE loss vs. target
   └─ spine publish → SpikenautDistill.jl (E-prop on SNN only)
```

## Ecosystem crates

| Crate | Role |
|-------|------|
| [`neuromod`](https://crates.io/crates/neuromod) | LIF + Izhikevich SNN with R-STDP |
| [`spikenaut-encoder`](https://github.com/Spikenaut/spikenaut-encoder) | Neuromodulator-driven Poisson encoding |
| [`spikenaut-telemetry`](https://github.com/Spikenaut/spikenaut-telemetry) | GPU/CPU/mining hardware telemetry |
| [`spikenaut-spine`](https://github.com/Spikenaut/spikenaut-spine) | ZMQ IPC Rust ↔ Julia bridge |
| `SpikenautDistill.jl` | Julia E-prop / OTTT SNN trainer (private) |

## Training loop

Only the SNN is updated. OLMoE stays **completely frozen**:

```
┌──────────────────────────────┐
│  Rust (spikenaut-hybrid)     │
│  1. forward(snap)            │
│  2. compute MSE loss         │
│  3. build TrainSignal        │──────ZMQ──────►┌───────────────────────┐
│  4. publish via spine        │                │  Julia (Distill.jl)   │
└──────────────────────────────┘                │  E-prop / OTTT        │
                                                │  Update SNN weights   │
                                                │  Send new W_proj back │
◄───────────────────────────────────────────────┤  via spine            │
 Projector::load_weights(W)                     └───────────────────────┘
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
