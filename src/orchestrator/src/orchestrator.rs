//! Orchestrator
use crate::{
    batcher::{Batcher, Request},
    executor::Executor,
    model::Model,
};
use crossbeam::channel;
use std::{
    collections::HashMap,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

/// Command to Orchestrator
pub enum Command {
    /// NewRequest(result_tx(request_handle), request_id)
    NewRequest(channel::Sender<Option<Request>>, usize),
    /// ExecutorFinish(queue_id)
    ExecutorFinish(usize),
    /// Schedule
    Schedule,
    /// ExecutorStart(queue_id)
    ExecutorStart(usize),
}

/// Orchestrator
pub struct Orchestrator {
    tx: channel::Sender<Command>,
    rx: channel::Receiver<Command>,
    batchers: HashMap<usize, Batcher>,
    executors: HashMap<usize, Executor>,
    models: HashMap<usize, Arc<Model>>,
    cuda_manager_handle: cuda_manager::Handle,
    scheduler:
        Box<dyn Fn(&BatcherStat, &mut BatcherOps, &ExecutorStat, &mut ExecutorOps, &ModelProfile)>,
    model_profile: ModelProfile,
    synchronize_interval: Duration,
    unready_executors: usize,
}

unsafe impl Send for Orchestrator {}

/// BatcherStat
pub struct BatcherStat(pub HashMap<usize, Vec<Instant>>);
/// BatcherOps
pub struct BatcherOps(pub HashMap<usize, BatcherOp>);
/// ExecutorStat
pub struct ExecutorStat(pub HashMap<usize, (i32, usize, Duration, Instant)>);
/// ExecutorOps
pub struct ExecutorOps(pub HashMap<usize, ExecutorOp>);

/// Op
#[derive(Debug)]
pub enum BatcherOp {
    /// Handle Request (model_id, gpu_id, drop_size, batch_size, high_priority_stream)
    HandleRequest(usize, i32, usize, usize, bool),
}

#[derive(Debug)]
/// Op
pub enum ExecutorOp {
    /// Stop
    Stop,
    /// Run
    Run,
    /// Cancel
    Cancel,
}

/// ModelProfile
pub struct ModelProfile(pub HashMap<usize, HashMap<i32, Vec<Duration>>>);

impl Orchestrator {
    pub(crate) fn new(
        batchers: HashMap<usize, Batcher>,
        models: HashMap<usize, Arc<Model>>,
        cuda_manager_handle: cuda_manager::Handle,
        scheduler: Box<
            dyn Fn(&BatcherStat, &mut BatcherOps, &ExecutorStat, &mut ExecutorOps, &ModelProfile),
        >,
        synchronize_interval: Duration,
    ) -> Self {
        let model_profile = models
            .iter()
            .map(|(&i, m)| {
                let profile = m.profile().clone();
                (i, profile)
            })
            .collect();
        let model_profile = ModelProfile(model_profile);
        let (tx, rx) = channel::unbounded();
        Orchestrator {
            tx,
            rx,
            batchers,
            models,
            cuda_manager_handle,
            scheduler,
            synchronize_interval,
            model_profile,
            executors: HashMap::new(),
            unready_executors: 0,
        }
    }

    /// Run Orchestrator
    pub fn run(mut self) -> channel::Sender<Command> {
        use Command::*;
        let tx = self.tx.clone();
        {
            let tx = self.tx.clone();
            thread::spawn(move || loop {
                tx.send(Schedule).unwrap();
                thread::sleep(Duration::from_millis(1));
            });
        }
        thread::spawn(move || {
            let mut last_schedule = Instant::now();
            loop {
                if let Ok(cmd) = self.rx.try_recv() {
                    match cmd {
                        ExecutorFinish(queue_id) => {
                            self.cuda_manager_handle.delete_op_queue(queue_id).unwrap();
                            self.executors.remove(&queue_id);
                        }
                        NewRequest(result_tx, batcher_id) => {
                            let _ = result_tx.send(self.new_request(batcher_id));
                        }
                        Schedule => {}
                        ExecutorStart(_queue_id) => {
                            self.unready_executors -= 1;
                        }
                    }
                    if self.unready_executors == 0
                        && last_schedule.elapsed() > Duration::from_micros(100)
                    {
                        self.schedule();
                        last_schedule = Instant::now();
                    }
                }
            }
        });
        tx
    }

    /// Create a new request
    pub fn new_request(&mut self, batcher_id: usize) -> Option<Request> {
        let batcher = &mut self.batchers.get_mut(&batcher_id)?;
        Some(batcher.new_request())
    }

    fn schedule(&mut self) {
        let mut batcher_stat = HashMap::new();
        let mut batcher_ops = HashMap::new();
        for (&batcher_id, batcher) in self.batchers.iter() {
            let requests_buffer = batcher.requests_buffer();
            if !requests_buffer.is_empty() {
                batcher_stat.insert(batcher_id, requests_buffer);
                batcher_ops.insert(batcher_id, BatcherOp::HandleRequest(0, 0, 0, 0, false));
            }
        }
        let gpus_stat = self.cuda_manager_handle.get_workload().unwrap();
        let mut executor_stat = HashMap::new();
        let mut executor_ops = HashMap::new();
        for (&queue_id, executor) in &self.executors {
            if let Some(stat) = gpus_stat.get(&queue_id) {
                let gpu_id = stat.0;
                let remain_workoad = stat.2;
                let batcher_id = executor.batcher_id();
                executor_stat.insert(
                    queue_id,
                    (gpu_id, batcher_id, remain_workoad, executor.arrive_time()),
                );
                executor_ops.insert(queue_id, ExecutorOp::Stop);
            }
        }
        let batcher_stat = BatcherStat(batcher_stat);
        let mut batcher_ops = BatcherOps(batcher_ops);
        let executor_stat = ExecutorStat(executor_stat);
        let mut executor_ops = ExecutorOps(executor_ops);
        let scheduler = &self.scheduler;
        let profile = &self.model_profile;
        scheduler(
            &batcher_stat,
            &mut batcher_ops,
            &executor_stat,
            &mut executor_ops,
            &profile,
        );
        let handle = self.cuda_manager_handle.clone();
        for (queue_id, op) in executor_ops.0 {
            let is_running = gpus_stat[&queue_id].1;
            match op {
                ExecutorOp::Stop => {
                    if is_running {
                        log::debug!("Preempt {}", queue_id);
                    }
                    handle.stop_op_queue(queue_id).unwrap();
                }
                ExecutorOp::Run => {
                    if !is_running {
                        log::debug!("Resume {}", queue_id);
                    }
                    handle.continue_op_queue(queue_id).unwrap();
                }
                ExecutorOp::Cancel => {
                    log::debug!("Cancele {} is running? {}", queue_id, is_running);
                    handle.cancel_op_queue(queue_id).unwrap();
                }
            }
        }
        for (batcher_id, op) in batcher_ops.0 {
            match op {
                BatcherOp::HandleRequest(
                    model_id,
                    gpu_id,
                    drop_size,
                    batch_size,
                    high_priority_stream,
                ) => {
                    self.batchers
                        .get_mut(&batcher_id)
                        .unwrap()
                        .pop_requests(drop_size);
                    let requests = self
                        .batchers
                        .get_mut(&batcher_id)
                        .unwrap()
                        .pop_requests(batch_size);
                    if requests.is_empty() {
                        continue;
                    }
                    let queue_id = handle
                        .create_op_queue(gpu_id, high_priority_stream)
                        .unwrap();
                    let model = self.models[&model_id].clone();
                    let executor = Executor::spawn(
                        self.tx.clone(),
                        batcher_id,
                        queue_id,
                        self.synchronize_interval,
                        handle.clone(),
                        requests,
                        model,
                    );
                    self.executors.insert(queue_id, executor);
                    self.unready_executors += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::builder::*;
    use crate::model::FakeModel;
    use serial_test::serial;
    #[serial]
    #[test]
    fn test_orchestrator() {
        let mut orchestrator = Builder::new()
            .gpu_ids(vec![0])
            .register_model(ModelInfo::new(0, |_| Box::new(FakeModel)))
            .register_request(RequestInfo::new(0))
            .scheduler(|batcher_stat, _, _, _, _| {
                // orchestrator should only pass the higher priority ones to schedule_algorithm
                assert_eq!(batcher_stat.0.len(), 1);
                assert_eq!(batcher_stat.0[&0].len(), 2);
            })
            .build();
        orchestrator.new_request(0).unwrap();
        orchestrator.new_request(0).unwrap();
        orchestrator.schedule();
    }
}
