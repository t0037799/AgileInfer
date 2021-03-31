#![deny(warnings)]
#![deny(missing_docs)]
#![warn(clippy::all)]

//! CudaManager is a Cuda runtime providing basic management for concurrent Cuda Streams execution

mod cuda_manager;
mod device_manager;
pub mod handle;
mod idmap;
pub mod utils;

pub use crate::cuda_manager::CudaManager;
pub use crate::handle::Handle;

#[derive(Debug, PartialEq)]
/// Cuda Executor report result through channel.
pub enum QueueResult {
    /// Cuda Executor successfully run all queued operations.
    Finish,
    /// OperationQueue is canceled.
    Cancel,
}

impl CudaKernel {
    /// Create CudaKernel
    pub fn new(function_id: usize, thread_axis: [u32; 6]) -> Self {
        CudaKernel {
            function_id,
            thread_axis,
        }
    }
}

/// CudaKernel's Argument
pub enum CudaArgument {
    /// Device Pointer
    Pointer(u64),
    /// Other type of argmuments
    Other(u64),
}

/// CudaOperation
pub enum CudaOperation {
    /// MemcpyHtoD(pointer, host_memory_id, copy_size)
    MemcpyHtoD(u64, usize, usize),
    /// MemcpyDtoH(pointer, host_memory_id, copy_size)
    MemcpyDtoH(u64, usize, usize),
    /// CudaKernel(kernel, argmuments)
    CudaKernel(CudaKernel, Vec<CudaArgument>),
}

#[derive(Hash, PartialEq, Eq)]
/// CudaKernel
pub struct CudaKernel {
    function_id: usize,
    thread_axis: [u32; 6],
}
