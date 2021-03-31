use crossbeam::channel;
use cuda_manager::Handle;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
/// All backend models managed by Orchestrator have Inference trait.
/// Orchestrator start model inference by it.
/// Model should be immutable after initialization, hence should be
/// Send + Sync
pub trait Inference: Send + Sync {
    /// model inference, the implementation should use CudaManager API
    /// to submit Cuda Operations.
    #[must_use]
    fn inference(
        &self,
        handle: Handle,
        batch_size: usize,
        queue_id: usize,
        input_id: usize,
    ) -> usize;
    /// Input size is in bytes.
    fn input_size(&self) -> usize;
}

pub struct Model {
    inner: Box<dyn Inference>,
    input_size: usize,
    profile: HashMap<i32, Vec<Duration>>,
}

impl Model {
    pub fn inference(
        &self,
        handle: Handle,
        batch_size: usize,
        queue_id: usize,
        input_id: usize,
    ) -> usize {
        self.inner.inference(handle, batch_size, queue_id, input_id)
    }

    pub fn profile(&self) -> &HashMap<i32, Vec<Duration>> {
        &self.profile
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

type BuildModelFn = dyn FnOnce(cuda_manager::Handle) -> Box<dyn Inference>;
pub struct Builder {
    warmup_times: usize,
    cuda_manager_handle: Option<cuda_manager::Handle>,
    build_function: Option<Box<BuildModelFn>>,
    gpu_ids: Vec<i32>,
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            warmup_times: 20,
            cuda_manager_handle: None,
            build_function: None,
            gpu_ids: vec![],
        }
    }
}

impl Builder {
    pub fn new() -> Self {
        Builder::default()
    }

    pub fn warmup_times(mut self, warmup_times: usize) -> Self {
        self.warmup_times = warmup_times;
        self
    }

    pub fn cuda_manager_handle(mut self, cuda_manager_handle: cuda_manager::Handle) -> Self {
        self.cuda_manager_handle = Some(cuda_manager_handle);
        self
    }

    pub fn gpu_ids(mut self, gpu_ids: &[i32]) -> Self {
        self.gpu_ids = gpu_ids.to_vec();
        self
    }

    pub fn build_function<F>(mut self, func: F) -> Self
    where
        F: 'static + FnOnce(cuda_manager::Handle) -> Box<dyn Inference>,
    {
        self.build_function = Some(Box::new(func));
        self
    }

    pub fn build(self) -> Model {
        assert!(self.cuda_manager_handle.is_some());
        assert!(self.build_function.is_some());
        let handle = self.cuda_manager_handle.unwrap();
        let build_function = self.build_function.unwrap();
        let inner = build_function(handle.clone());
        let input_size = inner.input_size();
        let mut profile = HashMap::new();
        for gpu_id in self.gpu_ids {
            profile.insert(gpu_id, vec![]);
            for &batch_size in &[1, 2, 4, 8, 16] {
                let input_size = inner.input_size() * batch_size;
                let bytes = vec![0; input_size];
                for _ in 0..self.warmup_times {
                    let queue_id = handle.create_op_queue(gpu_id, false).unwrap();
                    let input_id = 0;
                    handle.insert_host_memory(queue_id, input_id, bytes.clone());
                    let output_id = inner.inference(handle.clone(), batch_size, queue_id, input_id);
                    let (tx, rx) = channel::unbounded();
                    handle
                        .start_op_queue(queue_id, tx, Duration::from_secs(0), true)
                        .unwrap();
                    let (_, mut host_memories) = rx.recv().unwrap();
                    let output_buffer = host_memories.remove(&output_id).unwrap().leak();
                    handle.free_host_memory(queue_id, output_buffer.as_mut_ptr());
                }
                let queue_id = handle.create_op_queue(gpu_id, false).unwrap();
                let input_id = 0;
                handle.insert_host_memory(queue_id, input_id, bytes.clone());
                let now = Instant::now();
                let output_id = inner.inference(handle.clone(), batch_size, queue_id, input_id);
                let (tx, rx) = channel::unbounded();
                handle
                    .start_op_queue(queue_id, tx, Duration::from_secs(0), true)
                    .unwrap();
                let (_, mut host_memories) = rx.recv().unwrap();
                let output_buffer = host_memories.remove(&output_id).unwrap().leak();
                handle.free_host_memory(queue_id, output_buffer.as_mut_ptr());
                let elapsed = now.elapsed();
                profile.get_mut(&gpu_id).unwrap().push(elapsed);
            }
        }
        Model {
            inner,
            input_size,
            profile,
        }
    }
}
pub(crate) struct FakeModel;
impl Inference for FakeModel {
    fn inference(
        &self,
        handle: Handle,
        _batch_size: usize,
        queue_id: usize,
        _input_id: usize,
    ) -> usize {
        let output_size = 3;
        let mut output_buffer = unsafe {
            Vec::from_raw_parts(
                handle.alloc_host_memory(queue_id, output_size),
                output_size,
                output_size,
            )
        };
        output_buffer.iter_mut().for_each(|i| *i = 1);
        handle.insert_host_memory(queue_id, 1, output_buffer);
        1
    }

    fn input_size(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serial_test::serial;
    #[serial]
    #[test]
    fn test_build_model() {
        let handle = cuda_manager::CudaManager::from_gpu_ids(&[0]).unwrap().run();
        let model = Builder::new()
            .warmup_times(20)
            .gpu_ids(&[0])
            .cuda_manager_handle(handle.clone())
            .build_function(|_| Box::new(FakeModel))
            .build();
        assert_eq!(model.profile()[&0].len(), 5); // [1, 2, 4, 8, 16]
    }
}
