//! Public model API for the deterministic SNN -> projector -> MoE pipeline.
//!
//! Read this module first when you want the end-to-end runtime behavior.
//! Construction and validation live in [`core`], GPU temporal orchestration
//! lives in [`temporal`], and CSV helpers live in [`telemetry_io`].

mod core;
mod telemetry_io;
mod temporal;

pub use crate::types::{ModelConfig, ModelOutput};
pub use core::Model;
