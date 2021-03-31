use crate::device_manager::DeviceManager;
use crate::handle::Handle;
use crate::idmap::IdMap;
use crate::utils::ToResult;
use crate::{CudaOperation, QueueResult};

use crossbeam::channel;
use cuda_driver_sys::{self as cuda, cudaError_enum};
use std::{collections::HashMap, time::Duration};

/// CudaManager
pub struct CudaManager {
    gpu_id_map: HashMap<i32, usize>,
    device_managers: Vec<DeviceManager>,
    common_modules: IdMap<Vec<usize>>,
    common_functions: IdMap<Vec<usize>>,
    // (inner_gpu_id, inner_queue_id)
    op_queues: IdMap<(usize, usize)>,
    queue_id_map: HashMap<(usize, usize), usize>,
}
unsafe impl Send for CudaManager {}

impl CudaManager {
    /// Create a CudaManager for all available GPUs.
    pub fn new() -> Result<Self, cudaError_enum> {
        unsafe {
            cuda::cuInit(0).to_result()?;
        }
        let mut gpu_id_map = HashMap::new();
        let mut device_managers = vec![];
        for (i, gpu_id) in (0..).enumerate() {
            match DeviceManager::new(gpu_id) {
                Ok(device_manager) => {
                    device_managers.push(device_manager);
                    gpu_id_map.insert(gpu_id, i);
                }
                Err(_) => {
                    break;
                }
            }
        }
        Ok(CudaManager {
            gpu_id_map,
            device_managers,
            common_modules: IdMap::new(),
            common_functions: IdMap::new(),
            op_queues: IdMap::new(),
            queue_id_map: HashMap::new(),
        })
    }

    /// Create a CudaManager from given GPU id
    pub fn from_gpu_ids(gpu_ids: &[i32]) -> Result<CudaManager, cudaError_enum> {
        unsafe {
            cuda::cuInit(0).to_result()?;
        }
        let mut gpu_id_map = HashMap::new();
        let mut device_managers = vec![];
        for (i, &gpu_id) in gpu_ids.iter().enumerate() {
            device_managers.push(DeviceManager::new(gpu_id)?);
            gpu_id_map.insert(gpu_id, i);
        }
        Ok(CudaManager {
            gpu_id_map,
            device_managers,
            common_modules: IdMap::new(),
            common_functions: IdMap::new(),
            op_queues: IdMap::new(),
            queue_id_map: HashMap::new(),
        })
    }

    /// CudaManager Run
    pub fn run(self) -> Handle {
        Handle::new(self)
    }

    /// Load Cuda Module
    pub fn load_module(&mut self, module_data: &[u8]) -> Result<usize, cudaError_enum> {
        let mut ids = vec![];
        for device_manager in &mut self.device_managers {
            let id = device_manager.load_module(module_data)?;
            ids.push(id);
        }
        Ok(self.common_modules.insert(ids))
    }

    /// Get Cuda Function
    pub fn get_function(
        &mut self,
        module_id: usize,
        function_name: &[u8],
    ) -> Result<usize, cudaError_enum> {
        let mut ids = vec![];
        for device_manager in &mut self.device_managers {
            let id = device_manager.get_function(module_id, function_name)?;
            ids.push(id);
        }
        Ok(self.common_functions.insert(ids))
    }

    /// Allocate GPU memory
    pub fn allocate(&mut self, size: usize) -> Result<u64, cudaError_enum> {
        let mut pointer = None;
        for device_manager in &mut self.device_managers {
            pointer = Some(device_manager.allocate(size)?);
        }
        pointer.ok_or(cudaError_enum::CUDA_ERROR_UNKNOWN)
    }

    /// Copy from Host to all GPUs
    pub fn copy_to_all_devices<T>(&self, pointer: u64, source: &[T]) -> Result<(), cudaError_enum> {
        for device_manager in &self.device_managers {
            device_manager.copy_to_device(pointer, source)?;
        }
        Ok(())
    }

    /// Allocate host memory
    pub fn alloc_host_memory(&mut self, queue_id: usize, size: usize) -> *mut u8 {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let gpu_id = op_queue.0;
        let device_manager = &mut self.device_managers[gpu_id];
        device_manager.alloc_host_memory(size)
    }

    /// Free host memory
    pub fn free_host_memory(&mut self, queue_id: usize, ptr: *mut u8) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let gpu_id = op_queue.0;
        let device_manager = &mut self.device_managers[gpu_id];
        device_manager.free_host_memory(ptr)
    }

    /// InsertHostMemory to one operations queue
    pub fn insert_host_memory(&mut self, queue_id: usize, memory_id: usize, bytes: Vec<u8>) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let gpu_id = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[gpu_id];
        device_manager.insert_host_memory(queue_id, memory_id, bytes);
    }

    /// Create a OperationQueue
    pub fn create_op_queue(
        &mut self,
        gpu_id: i32,
        high_prior: bool,
    ) -> Result<usize, cudaError_enum> {
        let index = self.gpu_id_map[&gpu_id];
        let device_manager = &mut self.device_managers[index];
        let inner_id = device_manager.create_op_queue(high_prior)?;
        let queue_id = self.op_queues.insert((index, inner_id));
        self.queue_id_map.insert((index, inner_id), queue_id);
        Ok(queue_id)
    }

    /// Submit op to a OperationQueue
    pub fn submit_cuda_op(&mut self, queue_id: usize, op: CudaOperation) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let gpu_id = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[gpu_id];
        device_manager.submit_cuda_op(queue_id, op);
    }

    /// Start a OperationQueue
    pub fn start_op_queue(
        &mut self,
        queue_id: usize,
        result_tx: channel::Sender<(QueueResult, HashMap<usize, Vec<u8>>)>,
        synchronize_interval: Duration,
        is_profile: bool,
    ) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let index = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[index];
        device_manager.start_op_queue(queue_id, result_tx, synchronize_interval, is_profile);
    }

    /// Stop a OperationQueue
    pub fn stop_op_queue(&mut self, queue_id: usize) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let index = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[index];
        device_manager.stop_op_queue(queue_id);
    }

    /// Continue a OperationQueue
    pub fn continue_op_queue(&mut self, queue_id: usize) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let index = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[index];
        device_manager.continue_op_queue(queue_id);
    }

    /// Cancel a OperationQueue
    pub fn cancel_op_queue(&mut self, queue_id: usize) {
        let op_queue = self.op_queues.get(queue_id).unwrap();
        let index = op_queue.0;
        let queue_id = op_queue.1;
        let device_manager = &mut self.device_managers[index];
        device_manager.cancel_op_queue(queue_id);
    }

    /// Delete a OperationQueue
    pub fn delete_op_queue(&mut self, queue_id: usize) {
        let op_queue = self.op_queues.remove(queue_id).unwrap();
        let gpu_id = op_queue.0;
        let queue_id = op_queue.1;
        self.queue_id_map.remove(&(gpu_id, queue_id));
        let device_manager = &mut self.device_managers[gpu_id];
        device_manager.delete_op_queue(queue_id);
    }

    /// Get Workload
    pub fn get_workload(&self) -> HashMap<usize, (i32, bool, Duration)> {
        let mut workloads = HashMap::new();
        for (&gpu_id, &index) in &self.gpu_id_map {
            let workload = self.device_managers[index].get_workload();
            for (queue_id, (is_running, remain_workload)) in workload {
                let queue_id = self.queue_id_map[&(index, queue_id)];
                workloads.insert(queue_id, (gpu_id, is_running, remain_workload));
            }
        }
        workloads
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
    fn test_multiple_gpus() {
        // common
        let mut cuda_manager = CudaManager::from_gpu_ids(&[0, 1]).unwrap();
        let mut ptx = include_bytes!("../resources/add.ptx").to_vec();
        ptx.push(0);
        let module_id = cuda_manager.load_module(&ptx).unwrap();
        let function_id = cuda_manager.get_function(module_id, b"sum\0").unwrap();
        let (x, y, z) = (
            cuda_manager.allocate(4).unwrap(),
            cuda_manager.allocate(4).unwrap(),
            cuda_manager.allocate(4).unwrap(),
        );
        cuda_manager.copy_to_all_devices(x, &vec![1.0f32]).unwrap();
        cuda_manager.copy_to_all_devices(y, &vec![0.0f32]).unwrap();
        cuda_manager.copy_to_all_devices(z, &vec![0.0f32]).unwrap();

        // per gpu
        for &gpu_id in &[0, 1] {
            let queue_id = cuda_manager.create_op_queue(gpu_id, false).unwrap();
            let input = vec_to_bytes(vec![2.0f32 * gpu_id as f32]);
            let input_id = 0;
            cuda_manager.insert_host_memory(queue_id, input_id, input);
            let output = vec_to_bytes(vec![0.0f32]);
            let output_id = 1;
            cuda_manager.insert_host_memory(queue_id, output_id, output);
            cuda_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyHtoD(y, input_id, 4));
            cuda_manager.submit_cuda_op(
                queue_id,
                CudaOperation::CudaKernel(
                    CudaKernel::new(function_id, [1; 6]),
                    vec![Pointer(x), Pointer(y), Pointer(z), Other(1)],
                ),
            );
            cuda_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyDtoH(z, output_id, 4));
            let (tx, rx) = channel::unbounded();
            cuda_manager.start_op_queue(queue_id, tx, Duration::from_secs(0), true);
            let (result, mut host_memories) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
            assert_eq!(result, QueueResult::Finish);
            let output = bytes_to_vec::<f32>(host_memories.remove(&output_id).unwrap());
            assert_eq!(output[0], 1.0 + 2.0 * gpu_id as f32);
        }
    }
}
