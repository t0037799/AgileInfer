//! Facade of CudaManager, caller use Handle as API instead of managing message passing
//! themselves.

use crate::{CudaManager, CudaOperation, QueueResult};
use crossbeam::channel;
use cuda_driver_sys::cudaError_enum;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

/// Handle for CudaManager
#[derive(Clone)]
pub struct Handle {
    backend: Arc<Mutex<CudaManager>>,
}

/// Errors during API calls.
#[derive(Debug)]
pub enum Error {
    /// Error from Cuda
    Cuda(cudaError_enum),
    /// Error during sending message
    Send,
    /// Error during receiving message
    Recv,
}

impl Handle {
    pub(crate) fn new(backend: CudaManager) -> Self {
        Handle {
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    /// Load Cuda Module into CudaManager and get module_id.
    pub fn load_module(&self, module_data: Vec<u8>) -> Result<usize, Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.load_module(&module_data).map_err(Error::Cuda)
    }

    /// Get Cuda Function's function_id from given module_id and function_name.
    pub fn get_function(&self, module_id: usize, function_name: &[u8]) -> Result<usize, Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager
            .get_function(module_id, function_name)
            .map_err(Error::Cuda)
    }

    /// Allocate GPU memory in all GPUs managed by CudaManager.
    pub fn allocate(&self, size: usize) -> Result<u64, Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.allocate(size).map_err(Error::Cuda)
    }

    /// Copy data to GPU in all GPUs managed by CudaManager.
    #[allow(clippy::clippy::rc_buffer)]
    pub fn copy_to_all_devices(&self, pointer: u64, source: Arc<Vec<u8>>) -> Result<(), Error> {
        let cuda_manager = self.backend.lock().unwrap();
        cuda_manager
            .copy_to_all_devices(pointer, &source)
            .map_err(Error::Cuda)
    }

    /// Allocate host memory to GPU for input or output.
    pub fn alloc_host_memory(&self, queue_id: usize, size: usize) -> *mut u8 {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.alloc_host_memory(queue_id, size)
    }

    /// Allocate host memory to GPU for input or output.
    pub fn free_host_memory(&self, queue_id: usize, ptr: *mut u8) {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.free_host_memory(queue_id, ptr)
    }

    /// Insert host memory to GPU for input or output.
    // TODO use typed variable and RAII
    pub fn insert_host_memory(&self, queue_id: usize, memory_id: usize, bytes: Vec<u8>) {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.insert_host_memory(queue_id, memory_id, bytes);
    }

    /// Create an OperationQueue for stream operations
    /// Set high_prior = true to use high priority Cuda Stream
    pub fn create_op_queue(&self, gpu_id: i32, high_prior: bool) -> Result<usize, Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager
            .create_op_queue(gpu_id, high_prior)
            .map_err(Error::Cuda)
    }

    /// Submit CudaOperation to OperationQueue.
    /// Submit is asynchronous
    pub fn submit_cuda_op(&self, queue_id: usize, op: CudaOperation) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.submit_cuda_op(queue_id, op);
        Ok(())
    }

    /// Start running OperationQueue.
    pub fn start_op_queue(
        &self,
        queue_id: usize,
        result_tx: channel::Sender<(QueueResult, HashMap<usize, Vec<u8>>)>,
        synchronize_interval: Duration,
        is_profile: bool,
    ) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.start_op_queue(queue_id, result_tx, synchronize_interval, is_profile);
        Ok(())
    }

    /// Stop OperationQueue
    pub fn stop_op_queue(&self, queue_id: usize) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.stop_op_queue(queue_id);
        Ok(())
    }

    /// Continue to run a started OperationQueue
    pub fn continue_op_queue(&self, queue_id: usize) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.continue_op_queue(queue_id);
        Ok(())
    }

    /// Cancel a started OperationQueue, only operations are removed.
    pub fn cancel_op_queue(&self, queue_id: usize) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.cancel_op_queue(queue_id);
        Ok(())
    }

    /// Delete a started OperationQueue, operations and queue are removed.
    pub fn delete_op_queue(&self, queue_id: usize) -> Result<(), Error> {
        let mut cuda_manager = self.backend.lock().unwrap();
        cuda_manager.delete_op_queue(queue_id);
        Ok(())
    }

    /// Get All GPUs' remaining workload
    pub fn get_workload(&self) -> Result<HashMap<usize, (i32, bool, Duration)>, Error> {
        let cuda_manager = self.backend.lock().unwrap();
        Ok(cuda_manager.get_workload())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{bytes_to_vec, vec_to_bytes};
    use crate::{CudaArgument::*, CudaKernel};
    use serial_test::serial;

    #[serial]
    #[test]
    fn test_through_apis() {
        let cuda_manager = crate::cuda_manager::CudaManager::from_gpu_ids(&[0, 1]).unwrap();
        let handle = cuda_manager.run();
        let mut ptx = include_bytes!("../resources/add.ptx").to_vec();
        ptx.push(0);
        let module_id = handle.load_module(ptx).unwrap();
        let function_id = handle.get_function(module_id, b"sum\0").unwrap();
        let x = handle.allocate(4).unwrap();
        let y = handle.allocate(4).unwrap();
        let z = handle.allocate(4).unwrap();
        let x_ = Arc::new(vec_to_bytes(vec![1.0f32]));
        let y_ = Arc::new(vec_to_bytes(vec![0.0f32]));
        let z_ = Arc::new(vec_to_bytes(vec![0.0f32]));
        handle.copy_to_all_devices(x, x_).unwrap();
        handle.copy_to_all_devices(y, y_).unwrap();
        handle.copy_to_all_devices(z, z_).unwrap();
        for &gpu_id in &[0, 1] {
            let handle = handle.clone();
            std::thread::spawn(move || {
                let queue_id = handle.create_op_queue(gpu_id, false).unwrap();
                let input = vec_to_bytes(vec![2.0f32 * gpu_id as f32]);
                let input_id = 0;
                handle.insert_host_memory(queue_id, input_id, input);
                let output = vec_to_bytes(vec![0.0f32]);
                let output_id = 1;
                handle.insert_host_memory(queue_id, output_id, output);
                handle
                    .submit_cuda_op(queue_id, CudaOperation::MemcpyHtoD(y, input_id, 4))
                    .unwrap();
                handle
                    .submit_cuda_op(
                        queue_id,
                        CudaOperation::CudaKernel(
                            CudaKernel::new(function_id, [1, 1, 1, 1, 1, 1]),
                            vec![Pointer(x), Pointer(y), Pointer(z), Other(1)],
                        ),
                    )
                    .unwrap();
                handle
                    .submit_cuda_op(queue_id, CudaOperation::MemcpyDtoH(z, output_id, 4))
                    .unwrap();
                let (tx, rx) = channel::unbounded();
                handle
                    .start_op_queue(queue_id, tx, Duration::from_secs(0), true)
                    .unwrap();
                let (result, mut host_memories) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
                assert_eq!(result, QueueResult::Finish);
                let output = bytes_to_vec::<f32>(host_memories.remove(&output_id).unwrap());
                assert_eq!(output[0], 1.0 + 2.0 * gpu_id as f32);
            });
        }
    }
}
