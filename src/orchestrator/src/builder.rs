//! Builder for Orchestrator, caller should provide Model and Scheduler
use crate::{
    batcher::Batcher,
    model::{self, Inference},
    orchestrator::{
        BatcherOps, BatcherStat, ExecutorOps, ExecutorStat, ModelProfile, Orchestrator,
    },
};
use std::{sync::Arc, time::Duration};

type SchedulerArg =
    dyn Fn(&BatcherStat, &mut BatcherOps, &ExecutorStat, &mut ExecutorOps, &ModelProfile);

/// Builder for the juggler service
pub struct Builder {
    gpu_ids: Vec<i32>,
    warmup_times: usize,
    synchronize_interval: Duration,
    scheduler: Option<Box<SchedulerArg>>,
    model_infos: Vec<ModelInfo>,
    request_infos: Vec<RequestInfo>,
}

macro_rules! config {
    ($name:ident, $t: ty, $comment: literal) => {
        #[doc=$comment]
        pub fn $name(mut self, $name: $t) -> Self {
            self.$name = $name;
            self
        }
    };
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            gpu_ids: vec![],
            warmup_times: 10,
            synchronize_interval: Duration::from_millis(1),
            model_infos: vec![],
            request_infos: vec![],
            scheduler: None,
        }
    }
}

impl Builder {
    /// Create  a new builder with default configuration
    pub fn new() -> Self {
        Builder::default()
    }

    config!(gpu_ids, Vec<i32>, "GPU device ids");
    config!(synchronize_interval, Duration, "synchronize interval");
    config!(warmup_times, usize, "Warm up time before actual profile");

    /// model register
    pub fn register_model(mut self, info: ModelInfo) -> Self {
        self.model_infos.push(info);
        self
    }

    /// batcher register
    pub fn register_request(mut self, info: RequestInfo) -> Self {
        self.request_infos.push(info);
        self
    }

    /// scheduling algorithm
    pub fn scheduler<F>(mut self, scheduler: F) -> Self
    where
        F: 'static
            + Fn(&BatcherStat, &mut BatcherOps, &ExecutorStat, &mut ExecutorOps, &ModelProfile),
    {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    /// Build the service
    pub fn build(self) -> Orchestrator {
        let gpu_ids = self.gpu_ids;
        let model_infos = self.model_infos;
        let request_infos = self.request_infos;
        let warmup_times = self.warmup_times;
        let handle = cuda_manager::CudaManager::from_gpu_ids(&gpu_ids)
            .unwrap()
            .run();
        let models = model_infos
            .into_iter()
            .map(|info| {
                let build_function = info.model_builder;
                let model = model::Builder::new()
                    .warmup_times(warmup_times)
                    .gpu_ids(&gpu_ids)
                    .cuda_manager_handle(handle.clone())
                    .build_function(build_function)
                    .build();
                let profile = model.profile();
                let mut log = format!("Model {} Registered", info.id);
                log = format!("{}\nProfile:", log);
                for (gpu_id, profile) in profile {
                    log = format!("{}\nGPU[{}]: ", log, gpu_id);
                    for (i, t) in profile.iter().enumerate() {
                        log = format!("{}\n{} : {:?}", log, 1 << i, t);
                    }
                }
                log::info!("{}", log);
                (info.id, Arc::new(model))
            })
            .collect();
        let batchers = request_infos
            .iter()
            .map(|info| (info.id, Batcher::new()))
            .collect();
        let scheduler = self.scheduler.unwrap();
        Orchestrator::new(
            batchers,
            models,
            handle,
            scheduler,
            self.synchronize_interval,
        )
    }
}

/// register model info
pub struct ModelInfo {
    model_builder: Box<dyn FnOnce(cuda_manager::Handle) -> Box<dyn Inference>>,
    id: usize,
}

impl ModelInfo {
    /// create ModelInfo
    pub fn new<F>(id: usize, model_builder: F) -> Self
    where
        F: 'static + FnOnce(cuda_manager::Handle) -> Box<dyn Inference>,
    {
        let model_builder = Box::new(model_builder);
        ModelInfo { id, model_builder }
    }
}

/// register request info
pub struct RequestInfo {
    id: usize,
}

impl RequestInfo {
    /// create BatcherInfo
    pub fn new(id: usize) -> Self {
        RequestInfo { id }
    }
}
