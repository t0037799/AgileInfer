use crate::{
    batcher::{self, RequestInfo},
    model::Model,
    orchestrator,
};
use crossbeam::channel;
use cuda_manager::{Handle, QueueResult};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub(crate) struct Executor {
    arrive_time: Instant,
    batcher_id: usize,
}

impl Executor {
    pub fn arrive_time(&self) -> Instant {
        self.arrive_time
    }
    pub fn batcher_id(&self) -> usize {
        self.batcher_id
    }
    pub fn spawn(
        notify: channel::Sender<orchestrator::Command>,
        batcher_id: usize,
        queue_id: usize,
        synchronize_interval: Duration,
        handle: Handle,
        requests: Vec<RequestInfo>,
        model: Arc<Model>,
    ) -> Self {
        let arrive_time = requests[0].arrive_time();
        thread::spawn(move || {
            let batch_size = requests.len();
            let input_id = 0;
            let output_id = model.inference(handle.clone(), batch_size, queue_id, input_id);
            notify
                .send(orchestrator::Command::ExecutorStart(queue_id))
                .unwrap();
            let input_size = model.input_size();
            let total_size = input_size * batch_size;
            //let mut input_buffer = Vec::<u8>::with_capacity(total_size);
            let mut input_buffer = unsafe {
                Vec::from_raw_parts(
                    handle.alloc_host_memory(queue_id, total_size),
                    0,
                    total_size,
                )
            };
            requests.iter().for_each(|request| {
                let mut input = request.recv().unwrap();
                input_buffer.append(&mut input);
            });
            handle.insert_host_memory(queue_id, input_id, input_buffer);
            let (tx, rx) = channel::unbounded();
            handle
                .start_op_queue(queue_id, tx, synchronize_interval, false)
                .unwrap();
            let (result, mut host_memories) = rx.recv().unwrap();
            let mut output_buffer = host_memories.remove(&output_id).unwrap();
            let output_size = output_buffer.len() / batch_size;
            let output_pointer = output_buffer.as_mut_ptr();
            for (i, r) in requests.iter().enumerate() {
                // clone partly from output buffer.
                // it should leak and that output buffer to reclaim memory itself.
                let output = unsafe {
                    let tmp = Vec::from_raw_parts(
                        output_pointer.add(i * output_size),
                        output_size,
                        output_size,
                    );
                    let output = tmp.clone();
                    tmp.leak();
                    output
                };
                match result {
                    QueueResult::Finish => r.send(Ok(output)).unwrap(),
                    QueueResult::Cancel => r.send(Err(batcher::Error::Drop)).unwrap(),
                }
            }
            let input_buffer = host_memories.remove(&input_id).unwrap().leak();
            handle.free_host_memory(queue_id, input_buffer.as_mut_ptr());
            handle.free_host_memory(queue_id, output_buffer.leak().as_mut_ptr());
            let _ = notify.send(orchestrator::Command::ExecutorFinish(queue_id));
        });
        Executor {
            arrive_time,
            batcher_id,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::batcher::fake_request;
    use crate::model::{Builder, FakeModel};
    use serial_test::serial;
    use std::time::Duration;
    #[serial]
    #[test]
    fn test_executor() {
        let handle = cuda_manager::CudaManager::from_gpu_ids(&[0]).unwrap().run();
        let model = Builder::new()
            .warmup_times(20)
            .gpu_ids(&[0])
            .cuda_manager_handle(handle.clone())
            .build_function(|_| Box::new(FakeModel))
            .build();
        let (input_tx, input_rx) = channel::unbounded();
        let (output_tx, output_rx) = channel::unbounded();
        let requests = vec![fake_request(output_tx, input_rx)];
        let synchronize_interval = Duration::from_secs(0);
        let queue_id = handle.create_op_queue(0, false).unwrap();
        let (tx, rx) = channel::unbounded();
        Executor::spawn(
            tx,
            0,
            queue_id,
            synchronize_interval,
            handle.clone(),
            requests,
            Arc::new(model),
        );
        input_tx.send(vec![1]).unwrap();
        let result = output_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(result, vec![1; 3]);
    }
}
