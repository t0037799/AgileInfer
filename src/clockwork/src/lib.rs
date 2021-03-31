//! Clockwork model backend for AgileInfer
#![deny(missing_docs)]
#![deny(warnings)]

mod batched_model;
mod c_api;
mod model;
mod model_def;
mod pod_de;

pub use batched_model::BatchedModel as Model;
