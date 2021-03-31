//! DeviceManager manage one GPU instance.
//! it owns handler to the device and create a context for stream execution

use crate::idmap::IdMap;
use crate::utils::ToResult;
use crate::{CudaArgument, CudaKernel, CudaOperation, QueueResult};
use crossbeam::channel;
use cuda_driver_sys::{self as cuda, cudaError_enum};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub(crate) struct DeviceManager {
    context: cuda::CUcontext,
    modules: IdMap<CudaModule>,
    functions: Arc<RwLock<IdMap<cuda::CUfunction>>>,
    op_queues: IdMap<OperationQueue>,
    executors: HashMap<usize, Executor>,
    profile: Arc<RwLock<HashMap<CudaKernel, Duration>>>,
    used_gpu_time: Arc<AtomicU64>,
    base_pointer: u64,
    bump: usize,
    managed: usize,
    cuda_stream_pool: HashMap<i32, Vec<cuda::CUstream>>,
    host_memory_pool: HashMap<usize, Vec<CudaHostPtr>>,
    host_memory_map: HashMap<*mut u8, CudaHostPtr>,
}

struct CudaHostPtr {
    inner: *mut u8,
    size: usize,
}

struct CudaModule(cuda::CUmodule);
impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            cuda::cuModuleUnload(self.0);
        }
    }
}

impl Drop for DeviceManager {
    fn drop(&mut self) {
        unsafe {
            cuda::cuMemFree_v2(self.base_pointer);
        }
    }
}

// SAFETY: CUDA related data structures only used internally and should be managed properly.
unsafe impl Send for IdMap<cuda::CUfunction> {}
unsafe impl Sync for IdMap<cuda::CUfunction> {}
unsafe impl Send for OperationQueue {}
unsafe impl Sync for OperationQueue {}

fn cuda_synchronize(stream: cuda::CUstream) -> Result<(), cuda::cudaError_enum> {
    unsafe { cuda::cuStreamSynchronize(stream).to_result() }
}

#[derive(Clone, Copy)]
struct CudaStream {
    inner: cuda::CUstream,
    priority: i32,
}

struct OperationQueue {
    stream: CudaStream,
    ops: VecDeque<CudaOperation>,
    host_memory: HashMap<usize, Vec<u8>>,
}

struct Executor {
    thread: JoinHandle<()>,
    state: Arc<AtomicU64>,
    estimate_gpu_time: Arc<AtomicU64>,
    stream: CudaStream,
}

impl DeviceManager {
    pub fn new(gpu_id: i32) -> Result<Self, cudaError_enum> {
        let context = unsafe {
            let mut device: cuda::CUdevice = 0;
            cuda::cuDeviceGet(&mut device as *mut _, gpu_id).to_result()?;
            let mut context = std::ptr::null_mut();
            cuda::cuCtxCreate_v2(&mut context as *mut _, 0x8, device).to_result()?;
            context
        };
        let (base_pointer, managed) = unsafe {
            let mut free = 0usize;
            let mut total = 0usize;
            let mut base_pointer = 0u64;
            cuda::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _).to_result()?;
            let managed = free * 9 / 10;
            cuda::cuMemAlloc_v2(&mut base_pointer as *mut _, managed).to_result()?;
            (base_pointer, managed)
        };
        let mut cuda_stream_pool = HashMap::new();
        for priority in &[-1, 0] {
            cuda_stream_pool.insert(*priority, vec![]);
        }
        Ok(DeviceManager {
            context,
            modules: IdMap::new(),
            functions: Arc::new(RwLock::new(IdMap::new())),
            op_queues: IdMap::new(),
            executors: HashMap::new(),
            profile: Arc::new(RwLock::new(HashMap::new())),
            used_gpu_time: Arc::new(AtomicU64::new(0)),
            base_pointer,
            bump: 0,
            managed,
            cuda_stream_pool,
            host_memory_pool: HashMap::new(),
            host_memory_map: HashMap::new(),
        })
    }

    fn set_context(&self) -> Result<(), cudaError_enum> {
        unsafe {
            cuda::cuCtxSetCurrent(self.context).to_result()?;
        }
        Ok(())
    }

    pub fn load_module(&mut self, module_data: &[u8]) -> Result<usize, cudaError_enum> {
        self.set_context()?;
        let module = unsafe {
            let mut module = std::ptr::null_mut();
            cuda::cuModuleLoadData(&mut module as *mut _, module_data.as_ptr() as *const _)
                .to_result()?;
            module
        };
        Ok(self.modules.insert(CudaModule(module)))
    }

    pub fn get_function(
        &mut self,
        module_id: usize,
        function_name: &[u8],
    ) -> Result<usize, cudaError_enum> {
        self.set_context()?;
        let module = self.modules.get(module_id).unwrap().0;
        let function = unsafe {
            let mut function = std::ptr::null_mut();
            cuda::cuModuleGetFunction(
                &mut function as *mut _,
                module,
                function_name.as_ptr() as *const _,
            )
            .to_result()?;
            function
        };
        Ok(self.functions.write().unwrap().insert(function))
    }

    // TODO don't use bump allocator
    pub fn allocate(&mut self, size: usize) -> Result<u64, cudaError_enum> {
        let ret = self.bump as u64;
        self.bump += size;
        if self.bump <= self.managed {
            Ok(ret)
        } else {
            Err(cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY)
        }
    }

    pub fn copy_to_device<T>(&self, pointer: u64, source: &[T]) -> Result<(), cudaError_enum> {
        self.set_context()?;
        let copy_size = source.len() * std::mem::size_of::<T>();
        unsafe {
            cuda::cuMemcpyHtoD_v2(
                pointer + self.base_pointer,
                source.as_ptr() as *const _,
                copy_size,
            )
            .to_result()?;
        }
        Ok(())
    }

    pub fn create_op_queue(&mut self, high_prior: bool) -> Result<usize, cudaError_enum> {
        self.set_context()?;
        let stream = {
            let priority = if high_prior { -1 } else { 0 };
            let pool = self.cuda_stream_pool.get_mut(&priority).unwrap();
            let stream = match pool.pop() {
                Some(stream) => stream,
                None => {
                    let mut stream = std::ptr::null_mut();
                    unsafe {
                        cuda::cuStreamCreateWithPriority(&mut stream as *mut _, 0x1, priority)
                            .to_result()?;
                    }
                    stream
                }
            };
            CudaStream {
                inner: stream,
                priority,
            }
        };
        let op_queue = OperationQueue {
            stream,
            ops: VecDeque::new(),
            host_memory: HashMap::new(),
        };
        Ok(self.op_queues.insert(op_queue))
    }

    pub fn submit_cuda_op(&mut self, queue_id: usize, op: CudaOperation) {
        let queue = self.op_queues.get_mut(queue_id).unwrap();
        queue.ops.push_back(op);
    }

    pub fn start_op_queue(
        &mut self,
        queue_id: usize,
        result_tx: channel::Sender<(QueueResult, HashMap<usize, Vec<u8>>)>,
        synchronize_interval: Duration,
        is_profile: bool,
    ) {
        let op_queue = self.op_queues.remove(queue_id).unwrap();
        let stream = op_queue.stream;
        let state = Arc::new(AtomicU64::new(1));
        let estimate_gpu_time = op_queue.ops.iter().fold(0, |acc, op| match op {
            CudaOperation::CudaKernel(kernel, _) => {
                self.profile
                    .read()
                    .unwrap()
                    .get(kernel)
                    .unwrap_or(&Duration::from_secs(0))
                    .as_nanos() as u64
                    + acc
            }
            _ => acc,
        });
        let estimate_gpu_time = Arc::new(AtomicU64::new(estimate_gpu_time));
        let functions = Arc::clone(&self.functions);
        let profile = Arc::clone(&self.profile);
        let base_pointer = self.base_pointer;
        let thread = {
            let estimate_gpu_time = Arc::clone(&estimate_gpu_time);
            let state = Arc::clone(&state);
            let used_gpu_time = Arc::clone(&self.used_gpu_time);
            thread::spawn(move || {
                if let Err(e) = cuda_executor(
                    result_tx,
                    state,
                    estimate_gpu_time,
                    base_pointer,
                    op_queue,
                    functions,
                    profile,
                    is_profile,
                    synchronize_interval,
                    queue_id,
                    used_gpu_time,
                ) {
                    log::warn!("Error in Cuda Executor {:?}", e)
                }
            })
        };
        self.executors.insert(
            queue_id,
            Executor {
                thread,
                state,
                estimate_gpu_time,
                stream,
            },
        );
    }

    pub fn continue_op_queue(&mut self, queue_id: usize) {
        if let Some(executor) = self.executors.get_mut(&queue_id) {
            executor.state.store(1, Ordering::SeqCst);
            executor.thread.thread().unpark();
        }
    }

    pub fn stop_op_queue(&mut self, queue_id: usize) {
        if let Some(executor) = self.executors.get_mut(&queue_id) {
            executor.state.store(0, Ordering::SeqCst);
        }
    }

    pub fn cancel_op_queue(&mut self, queue_id: usize) {
        if let Some(executor) = self.executors.get_mut(&queue_id) {
            executor.state.store(2, Ordering::SeqCst);
            executor.thread.thread().unpark();
        }
    }

    pub fn delete_op_queue(&mut self, queue_id: usize) {
        if let Some(executor) = self.executors.get(&queue_id) {
            executor.state.store(2, Ordering::SeqCst);
            executor.thread.thread().unpark();
            let executor = self.executors.remove(&queue_id).unwrap();
            let priority = executor.stream.priority;
            let pool = self.cuda_stream_pool.get_mut(&priority).unwrap();
            pool.push(executor.stream.inner);
        }
    }

    pub fn alloc_host_memory(&mut self, size: usize) -> *mut u8 {
        if let Some(cache) = self.host_memory_pool.get_mut(&size) {
            if let Some(ptr) = cache.pop() {
                let p = ptr.inner;
                self.host_memory_map.insert(ptr.inner, ptr);
                return p;
            }
        } else {
            self.host_memory_pool.insert(size, vec![]);
        }
        unsafe {
            self.set_context().unwrap();
            let mut ptr = std::ptr::null_mut();
            cuda::cuMemAllocHost_v2(&mut ptr as *mut _ as *mut _, size)
                .to_result()
                .unwrap();
            self.host_memory_map
                .insert(ptr, CudaHostPtr { inner: ptr, size });
            ptr
        }
    }

    pub fn free_host_memory(&mut self, ptr: *mut u8) {
        if let Some(ptr) = self.host_memory_map.remove(&ptr) {
            let cache = self.host_memory_pool.get_mut(&ptr.size).unwrap();
            cache.push(ptr);
        }
    }

    pub fn insert_host_memory(&mut self, queue_id: usize, memory_id: usize, bytes: Vec<u8>) {
        let queue = self.op_queues.get_mut(queue_id).unwrap();
        queue.host_memory.insert(memory_id, bytes);
    }

    pub fn get_workload(&self) -> HashMap<usize, (bool, Duration)> {
        let not_started = self
            .op_queues
            .iter()
            .map(|(&id, op_queue)| {
                let estimate_gpu_time = op_queue.ops.iter().fold(0, |acc, op| match op {
                    CudaOperation::CudaKernel(kernel, _) => {
                        self.profile
                            .read()
                            .unwrap()
                            .get(kernel)
                            .unwrap_or(&Duration::from_secs(0))
                            .as_nanos() as u64
                            + acc
                    }
                    _ => acc,
                });
                let estimate_gpu_time = Duration::from_nanos(estimate_gpu_time);
                (id, (false, estimate_gpu_time))
            })
            .collect::<HashMap<_, _>>();
        let mut started = self
            .executors
            .iter()
            .map(|(&id, executor)| {
                let is_running = matches!(executor.state.load(Ordering::SeqCst), 1);
                let estimate_gpu_time = executor.estimate_gpu_time.load(Ordering::SeqCst);
                let estimate_gpu_time = Duration::from_nanos(estimate_gpu_time);
                (id, (is_running, estimate_gpu_time))
            })
            .collect::<HashMap<_, _>>();
        started.extend(not_started);
        started
    }
}

#[allow(clippy::clippy::clippy::too_many_arguments)]
fn cuda_executor(
    result_tx: channel::Sender<(QueueResult, HashMap<usize, Vec<u8>>)>,
    state: Arc<AtomicU64>,
    estimate_gpu_time: Arc<AtomicU64>,
    base_pointer: u64,
    mut op_queue: OperationQueue,
    functions: Arc<RwLock<IdMap<cuda::CUfunction>>>,
    profile: Arc<RwLock<HashMap<CudaKernel, Duration>>>,
    is_profile: bool,
    synchronize_interval: Duration,
    queue_id: usize,
    used_gpu_time: Arc<AtomicU64>,
) -> Result<(), cudaError_enum> {
    let synchronize_globally = true;
    let mut accumulated_time = Duration::from_secs(0);
    let stream = op_queue.stream.inner;
    let mut last_sum_time = Duration::new(0, 0);
    loop {
        match state.load(Ordering::SeqCst) {
            0 => {
                thread::park();
                continue;
            }
            1 => {}
            2 => {
                result_tx
                    .send((QueueResult::Cancel, op_queue.host_memory))
                    .unwrap();
                return Ok(());
            }
            _ => {
                unreachable!("Invalid state")
            }
        }
        if let Some(op) = op_queue.ops.pop_front() {
            match op {
                CudaOperation::MemcpyHtoD(pointer, host_memory_id, size) => unsafe {
                    let now = Instant::now();
                    let host_src = op_queue.host_memory.get(&host_memory_id).unwrap();
                    if host_src.len() < size {
                        log::warn!(
                            "Host buffer size {} < Copied size {}. Skip this HtoD copy",
                            host_src.len(),
                            size
                        );
                        continue;
                    }
                    cuda::cuMemcpyHtoDAsync_v2(
                        pointer + base_pointer,
                        host_src.as_ptr() as *const _,
                        size,
                        stream,
                    )
                    .to_result()?;
                    log::debug!("HtoD {:?}", now.elapsed());
                },
                CudaOperation::MemcpyDtoH(pointer, host_memory_id, size) => unsafe {
                    let now = Instant::now();
                    let host_dest = op_queue.host_memory.get_mut(&host_memory_id).unwrap();
                    assert!(host_dest.len() >= size);
                    cuda::cuMemcpyDtoHAsync_v2(
                        host_dest.as_ptr() as *mut _,
                        pointer + base_pointer,
                        size,
                        stream,
                    )
                    .to_result()?;
                    log::debug!("DtoH {:?}", now.elapsed());
                },
                CudaOperation::CudaKernel(cuda_kernel, cuda_args) => {
                    let now = Instant::now();
                    let estimate = {
                        let profile = profile.read().unwrap();
                        *profile.get(&cuda_kernel).unwrap_or(&Duration::from_secs(0))
                    };
                    accumulated_time += estimate;
                    let functions = functions.read().unwrap();
                    let function = *functions.get(cuda_kernel.function_id).unwrap();
                    let mut cuda_args = cuda_args
                        .into_iter()
                        .map(|arg| match arg {
                            CudaArgument::Pointer(pointer) => pointer + base_pointer,
                            CudaArgument::Other(x) => x,
                        })
                        .collect::<Vec<_>>();
                    let mut kernel_params: Vec<*mut std::ffi::c_void> = cuda_args
                        .iter_mut()
                        .map(|arg| arg as *mut _ as *mut _)
                        .collect();
                    unsafe {
                        cuda::cuLaunchKernel(
                            function,
                            cuda_kernel.thread_axis[0],
                            cuda_kernel.thread_axis[1],
                            cuda_kernel.thread_axis[2],
                            cuda_kernel.thread_axis[3],
                            cuda_kernel.thread_axis[4],
                            cuda_kernel.thread_axis[5],
                            0,
                            stream,
                            kernel_params.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                        .to_result()?;
                    }
                    if is_profile {
                        unsafe {
                            cuda::cuStreamSynchronize(stream).to_result()?;
                        }
                        let elapsed = now.elapsed();
                        profile.write().unwrap().insert(cuda_kernel, elapsed);
                    } else {
                        let sum_time = used_gpu_time
                            .fetch_add(estimate.as_nanos() as u64, Ordering::SeqCst)
                            + estimate.as_nanos() as u64;
                        let sum_time = Duration::from_nanos(sum_time);
                        if sum_time - last_sum_time > synchronize_interval {
                            if synchronize_globally {
                                cuda_synchronize(stream)?;
                            }
                            last_sum_time = sum_time;
                        }
                        if accumulated_time > synchronize_interval {
                            if !synchronize_globally {
                                cuda_synchronize(stream)?;
                            }
                            accumulated_time = Duration::from_secs(0);
                        }
                    }
                    let t =
                        estimate_gpu_time.fetch_sub(estimate.as_nanos() as u64, Ordering::SeqCst);
                    log::debug!(
                        "Kernel {} {:?} {:?}",
                        queue_id,
                        now.elapsed(),
                        Duration::from_nanos(t)
                    );
                }
            }
        } else {
            cuda_synchronize(stream)?;
            result_tx
                .send((QueueResult::Finish, op_queue.host_memory))
                .unwrap();
            return Ok(());
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{bytes_to_vec, vec_to_bytes};
    use serial_test::serial;

    fn setup() -> (DeviceManager, usize, usize, (usize, usize), (u64, u64, u64)) {
        use CudaArgument::*;
        unsafe {
            cuda::cuInit(0).to_result().unwrap();
        }
        let mut device_manager = DeviceManager::new(0).unwrap();
        let mut ptx = include_bytes!("../resources/add.ptx").to_vec();
        ptx.push(0);
        let module_id = device_manager.load_module(&ptx).unwrap();
        let function_id = device_manager.get_function(module_id, b"sum\0").unwrap();
        let (x, y, z) = (
            device_manager.allocate(4).unwrap(),
            device_manager.allocate(4).unwrap(),
            device_manager.allocate(4).unwrap(),
        );
        let queue_id = device_manager.create_op_queue(false).unwrap();
        let input = vec_to_bytes(vec![2.0f32]);
        let input_id = 0;
        device_manager.insert_host_memory(queue_id, input_id, input);
        let output = vec_to_bytes(vec![0.0f32]);
        let output_id = 1;
        device_manager.insert_host_memory(queue_id, output_id, output);
        device_manager.copy_to_device(x, &vec![1.0f32]).unwrap();
        device_manager.copy_to_device(y, &vec![0.0f32]).unwrap();
        device_manager.copy_to_device(z, &vec![0.0f32]).unwrap();
        device_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyHtoD(y, input_id, 4));
        device_manager.submit_cuda_op(
            queue_id,
            CudaOperation::CudaKernel(
                CudaKernel::new(function_id, [1; 6]),
                vec![Pointer(x), Pointer(y), Pointer(z), Other(1)],
            ),
        );
        device_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyDtoH(z, output_id, 4));
        (
            device_manager,
            queue_id,
            function_id,
            (input_id, output_id),
            (x, y, z),
        )
    }

    #[serial]
    #[test]
    fn test_device_manager() {
        use CudaArgument::*;
        let (mut device_manager, queue_id, function_id, (input_id, output_id), (x, y, z)) = setup();
        let (tx, rx) = channel::unbounded();
        let workload = device_manager.get_workload();
        for &v in workload.values() {
            assert!(v.0 == false);
            assert!(v.1 == Duration::from_secs(0));
        }
        device_manager.start_op_queue(queue_id, tx, Duration::from_secs(0), true);
        let (result, mut output_memories) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(result, QueueResult::Finish);
        let output = bytes_to_vec::<f32>(output_memories.remove(&output_id).unwrap());
        assert_eq!(output, vec![3.0f32]);
        device_manager.delete_op_queue(queue_id);
        let queue_id = device_manager.create_op_queue(false).unwrap();
        device_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyHtoD(y, input_id, 4));
        device_manager.submit_cuda_op(
            queue_id,
            CudaOperation::CudaKernel(
                CudaKernel::new(function_id, [1; 6]),
                vec![Pointer(x), Pointer(y), Pointer(z), Other(1)],
            ),
        );
        device_manager.submit_cuda_op(queue_id, CudaOperation::MemcpyDtoH(z, output_id, 4));
        let workload = device_manager.get_workload();
        for &v in workload.values() {
            assert!(v.0 == false);
            assert!(v.1 > Duration::from_secs(0));
        }
    }

    #[serial]
    #[test]
    fn test_stop_op() {
        let (mut device_manager, queue_id, _, (_input_id, _output_id), _) = setup();
        let (tx, rx) = channel::unbounded();
        device_manager.start_op_queue(queue_id, tx, Duration::from_secs(0), true);
        device_manager.stop_op_queue(queue_id);
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_err());
        device_manager.continue_op_queue(queue_id);
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
    }

    #[serial]
    #[test]
    fn test_cancel_op() {
        let (mut device_manager, queue_id, _, (_input_id, _output_id), _) = setup();
        let (tx, rx) = channel::unbounded();
        device_manager.start_op_queue(queue_id, tx, Duration::from_secs(0), true);
        device_manager.cancel_op_queue(queue_id);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap().0,
            QueueResult::Cancel
        );
    }
}
