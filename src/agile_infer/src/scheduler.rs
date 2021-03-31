use orchestrator::orchestrator::*;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::{Duration, Instant},
};

/// The default scheduler provides prioritized execution and concurrent execution.
/// You can also set minimum batch size for batch jobs.
pub struct Scheduler {
    concurrency: usize,
    high_prio_stream: bool,
    batcher_to_priority: HashMap<usize, usize>,
    slos: Vec<Duration>,
    min_batch_size: usize,
}

impl Scheduler {
    /// Create a new scheduler with configurations.
    pub fn new(
        concurrency: usize,
        high_prio_stream: bool,
        priorities: Vec<usize>,
        slos: Vec<Duration>,
        min_batch_size: usize,
    ) -> Self {
        let mut priority_to_batchers = BTreeMap::new();
        let mut batcher_to_priority = HashMap::new();
        for (batcher_id, priority) in priorities.into_iter().enumerate() {
            match priority_to_batchers.get_mut(&priority) {
                None => {
                    priority_to_batchers.insert(priority, vec![batcher_id]);
                }
                Some(set) => set.push(batcher_id),
            }
            batcher_to_priority.insert(batcher_id, priority);
        }
        Scheduler {
            concurrency,
            high_prio_stream,
            batcher_to_priority,
            slos,
            min_batch_size,
        }
    }

    /// Orchestrator will call this to get decisions.
    pub fn schedule(
        &self,
        batcher_stat: &BatcherStat,
        batcher_ops: &mut BatcherOps,
        executor_stat: &ExecutorStat,
        executor_ops: &mut ExecutorOps,
        model_profile: &ModelProfile,
    ) {
        let mut next_run = 0;
        let mut already_started = HashSet::new();
        let highest_prio = {
            let mut highest_prio = 0;
            for (_, batcher_id, _, _) in executor_stat.0.values() {
                highest_prio = highest_prio.max(self.batcher_to_priority[batcher_id]);
            }
            for batcher_id in batcher_stat.0.keys() {
                highest_prio = highest_prio.max(self.batcher_to_priority[batcher_id]);
            }
            highest_prio
        };
        let mut remain_gpu_time = {
            let mut running = vec![];
            for (queue_id, (_gpu_id, batcher_id, remain, arrive_time)) in &executor_stat.0 {
                let slo = self.slos[*batcher_id];
                let deadline = *arrive_time + slo;
                let now = Instant::now();
                let priority = self.batcher_to_priority[batcher_id];
                let diff = deadline.saturating_duration_since(now);
                already_started.insert(batcher_id);
                if priority == highest_prio {
                    running.push((diff, *remain, *queue_id));
                }
            }
            running.sort();
            let mut workload = Duration::new(0, 1);
            let mut remain_gpu_time = Duration::new(0, 0);
            for &(diff, remain, queue_id) in &running {
                let op = executor_ops.0.get_mut(&queue_id).unwrap();
                if diff == Duration::new(0, 0) {
                    // already miss deadline cancel it.
                    log::error!("Executor Missed Deadline, {}", queue_id);
                    *op = ExecutorOp::Cancel;
                    continue;
                }
                if next_run == self.concurrency {
                    break;
                }
                if diff > workload + remain {
                    if remain_gpu_time == Duration::new(0, 0) {
                        remain_gpu_time = diff - remain;
                    } else if remain_gpu_time > remain {
                        remain_gpu_time -= remain;
                    } else {
                        // GPU budget used up, no further check. Stop it and wait for future
                        // schedule.
                        break;
                    }

                    *op = ExecutorOp::Run;
                    next_run += 1;
                    workload += remain;
                } else {
                    // impossible to meet deadline
                    log::error!("Executor Predict Missed Deadline {}", queue_id);
                    *op = ExecutorOp::Cancel;
                }
            }
            remain_gpu_time
        };
        let mut deadlines = vec![];
        for (batcher_id, arrive_times) in &batcher_stat.0 {
            for arrive_time in arrive_times {
                let slo = self.slos[*batcher_id];
                deadlines.push((*arrive_time + slo, batcher_id));
            }
        }
        deadlines.sort();
        let now = Instant::now();
        for (deadline, batcher_id) in deadlines {
            let priority = self.batcher_to_priority[batcher_id];
            if priority != highest_prio {
                continue;
            }
            let op = batcher_ops.0.get_mut(batcher_id).unwrap();
            let diff = deadline.saturating_duration_since(now);
            match op {
                BatcherOp::HandleRequest(
                    model_id,
                    gpu_id,
                    drop_size,
                    batch_size,
                    _high_prio_stream,
                ) => {
                    if *batch_size == 16 {
                        continue;
                    }
                    *model_id = *batcher_id;
                    let last_estimate_gpu_time = {
                        if *batch_size == 0 {
                            Duration::new(0, 0)
                        } else {
                            let batch_order =
                                (*batch_size).next_power_of_two().trailing_zeros() as usize;
                            model_profile.0[batcher_id][gpu_id][batch_order]
                        }
                    };
                    let estimate_gpu_time = {
                        let batch_order =
                            (*batch_size + 1).next_power_of_two().trailing_zeros() as usize;
                        if batch_order >= 5 {
                            println!("{}", batch_size);
                        }
                        model_profile.0[batcher_id][gpu_id][batch_order]
                    };
                    if diff == Duration::new(0, 0) {
                        // already miss deadline
                        log::error!("Batcher {} Missed Deadline", batcher_id);
                        *drop_size += 1;
                        continue;
                    }
                    if diff < estimate_gpu_time && *batch_size == 0 {
                        // impossible to meet deadline
                        // if batch_size already greater than 0, then leave the decision to next
                        // schedule
                        log::error!("Batcher {} Predict Missed Deadline", batcher_id);
                        *drop_size += 1;
                        continue;
                    }
                    if already_started.contains(batcher_id) {
                        *batch_size = 0;
                        continue;
                    }
                    if remain_gpu_time == Duration::new(0, 0) {
                        // no GPU task is running, set remain_gpu_time.
                        remain_gpu_time = diff - estimate_gpu_time;
                        *batch_size = 1;
                        continue;
                    }
                    let extra_gpu_time = estimate_gpu_time - last_estimate_gpu_time;
                    if remain_gpu_time < extra_gpu_time {
                        // GPU budget used up, leave decision to next schedule
                        break;
                    } else {
                        *batch_size += 1;
                        remain_gpu_time -= extra_gpu_time;
                    }
                }
            }
        }
        for op in batcher_ops.0.values_mut() {
            match op {
                BatcherOp::HandleRequest(
                    model_id,
                    gpu_id,
                    _drop_size,
                    batch_size,
                    high_prio_stream,
                ) => {
                    // always use GPU0
                    *gpu_id = 0;
                    // closed loop request's batch_size should not be less than configured batch
                    // size
                    if *model_id >= 1 && *batch_size < self.min_batch_size {
                        *batch_size = 0;
                    }
                    *high_prio_stream =
                        self.high_prio_stream && self.batcher_to_priority[model_id] > 0;
                }
            }
        }
    }
}
