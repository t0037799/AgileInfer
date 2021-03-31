//! This crate is the composite of API layer, Orchestrator, and CudaManager.
//! We provide a default scheduler for prioritized execution and concurrent inference.

#![deny(warnings)]

pub mod grpc;
mod scheduler;

pub use scheduler::Scheduler;
