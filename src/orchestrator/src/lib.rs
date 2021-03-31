#![deny(warnings)]
#![deny(missing_docs)]
#![warn(clippy::all)]

//! Orchestrator orchestrates inference requests for all loaded models.

pub(crate) mod batcher;
pub mod builder;
pub(crate) mod executor;
pub(crate) mod model;
pub mod orchestrator;

pub use crate::orchestrator::Command;
pub use model::Inference;
