use crate::model::Model;
use cuda_manager::{CudaOperation, Handle};
use std::{fs::File, io::prelude::*, sync::Arc};
/// Clockwork Model
pub struct BatchedModel {
    models: Vec<Model>,
    pointer: u64,
}

impl BatchedModel {
    /// Create a model from path
    pub fn new(path: &str, handle: Handle) -> Self {
        let mut models = vec![];
        let mut params = vec![];
        let mut f = File::open(format!("{}/model.clockwork_params", path)).unwrap();
        f.read_to_end(&mut params).unwrap();
        for i in 0.. {
            let model_def = format!("{}/model.{}.clockwork", path, 1 << i);
            let model_lib = format!("{}/model.{}.so", path, 1 << i);
            if let Ok(model) = Model::new(&model_def, &model_lib, handle.clone()) {
                models.push(model);
            } else {
                break;
            }
        }
        let workspace_size = models.iter().fold(0, |max, m| {
            if max > m.workspace_size() {
                max
            } else {
                m.workspace_size()
            }
        });
        let input_size = models[models.len() - 1].input_size();
        let output_size = models[models.len() - 1].output_size();
        let weight_size = models[0].weight_size();
        let total_size = workspace_size + input_size + output_size + weight_size;
        let pointer = handle.allocate(total_size).unwrap();
        handle
            .copy_to_all_devices(pointer, Arc::new(params))
            .unwrap();
        BatchedModel { models, pointer }
    }
    /// Submit CudaOperations for model inference.
    pub fn inference(
        &self,
        handle: Handle,
        batch_size: usize,
        queue_id: usize,
        input_id: usize,
    ) -> usize {
        let index = batch_size.next_power_of_two().trailing_zeros() as usize;
        let input_size = self.models[0].input_size() * batch_size;
        let output_size = self.models[0].output_size() * batch_size;
        if let Some(model) = self.models.get(index) {
            //let output_buffer = vec![0; output_size];
            let output_buffer = unsafe {
                Vec::from_raw_parts(
                    handle.alloc_host_memory(queue_id, output_size),
                    output_size,
                    output_size,
                )
            };
            let output_id = 1;
            handle.insert_host_memory(queue_id, output_id, output_buffer);
            handle
                .submit_cuda_op(
                    queue_id,
                    CudaOperation::MemcpyHtoD(
                        self.pointer + model.input_offset() as u64,
                        input_id,
                        input_size,
                    ),
                )
                .unwrap();
            model.inference(handle.clone(), queue_id, self.pointer);
            handle
                .submit_cuda_op(
                    queue_id,
                    CudaOperation::MemcpyDtoH(
                        self.pointer + model.output_offset() as u64,
                        output_id,
                        output_size,
                    ),
                )
                .unwrap();
            return output_id;
        }
        panic!("Unsupported batch size {}", batch_size)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crossbeam::channel;
    use cuda_manager::*;
    use serial_test::serial;
    use std::time::Duration;

    #[serial]
    #[test]
    fn test_batched_model() {
        let gpu_id = 0;
        let cuda_manager = CudaManager::from_gpu_ids(&[gpu_id]).unwrap();
        let handle = cuda_manager.run();
        let model = BatchedModel::new("resources/resnet18_v2", handle.clone());
        for &batch_size in &[1, 2, 4, 7, 10] {
            let queue_id = handle.create_op_queue(gpu_id, false).unwrap();
            let input_buffer = vec![0; 602112 * batch_size];
            let input_id = 0;
            handle.insert_host_memory(queue_id, input_id, input_buffer);
            let output_id = model.inference(handle.clone(), batch_size, queue_id, input_id);
            let (tx, rx) = channel::unbounded();
            handle
                .start_op_queue(queue_id, tx, Duration::from_secs(0), true)
                .unwrap();
            let (result, mut host_memories) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
            assert_eq!(result, QueueResult::Finish);
            let output = host_memories.remove(&output_id).unwrap();
            assert_eq!(output.len(), batch_size * 4000);
            handle.free_host_memory(queue_id, output.leak().as_mut_ptr());
        }
    }
}
