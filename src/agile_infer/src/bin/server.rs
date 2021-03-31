#![deny(warnings)]
use agile_infer::{
    grpc::{self, infer_server::InferServer},
    Scheduler,
};
use clap::{App, Arg};
use orchestrator::builder::*;
use std::time::Duration;
use tonic::transport::Server;

struct Model(clockwork::Model, usize);

impl Model {
    fn new(path: &str, handle: cuda_manager::Handle) -> Self {
        let inner = clockwork::Model::new(path, handle);
        Model(inner, 602112)
    }
}

impl orchestrator::Inference for Model {
    fn inference(
        &self,
        handle: cuda_manager::Handle,
        batch_size: usize,
        queue_id: usize,
        input_id: usize,
    ) -> usize {
        self.0.inference(handle, batch_size, queue_id, input_id)
    }

    fn input_size(&self) -> usize {
        self.1
    }
}

#[tokio::main]
async fn main() {
    let matches = App::new("Agile Infer Server")
        .about("An example server for evaluating Agile Infer")
        .arg(
            Arg::with_name("synchronize_interval")
                .short("t")
                .help("GPU synchronize its execution after the interval elapsed, use value[s|ms|us|ns], e.g. 1ns")
                .default_value("1ms")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("concurrency")
                .short("c")
                .help("How many inferences scheduler can concurrently with the same priority")
                .default_value("4")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("model_copies")
                .short("m")
                .help("How many copies of model used in closed loop request")
                .default_value("4")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("closed_loop_model")
                .short("C")
                .help("Model path for closed loop request")
                .default_value("resources/resnet50_v2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("trace_model")
                .short("T")
                .help("Model path for trace request")
                .default_value("resources/resnet18_v2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("closed_loop_batch_size")
                .short("b")
                .help("Minimum batch size for inference closed loop request")
                .default_value("16")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("slo_ms")
                .short("s")
                .help("Service Level Objectives for trace request in milliseconds")
                .default_value("10")
                .takes_value(true),
        )
        .get_matches();

    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp_micros().init();
    let high_prio_stream = false;
    let concurrency = matches.value_of("concurrency").unwrap().parse().unwrap();
    let copies = matches.value_of("model_copies").unwrap().parse().unwrap();
    let synchronize_interval = {
        let s = matches.value_of("synchronize_interval").unwrap();
        if let Some(idx) = s.find("ns") {
            Duration::from_nanos(s[..idx].parse().unwrap())
        } else if let Some(idx) = s.find("us") {
            Duration::from_micros(s[..idx].parse().unwrap())
        } else if let Some(idx) = s.find("ms") {
            Duration::from_millis(s[..idx].parse().unwrap())
        } else if let Some(idx) = s.find('s') {
            Duration::from_secs(s[..idx].parse().unwrap())
        } else {
            println!("Can't parse {} to duration, use default interval 1ms", s);
            Duration::from_millis(1)
        }
    };
    let mut priorities = vec![1];
    priorities.append(&mut vec![0; copies]);
    let mut slos = vec![Duration::from_millis(
        matches.value_of("slo_ms").unwrap().parse().unwrap(),
    )];
    slos.append(&mut vec![Duration::from_secs(1000); copies]);
    let closed_loop_batch_size = matches
        .value_of("closed_loop_batch_size")
        .unwrap()
        .parse()
        .unwrap();
    let scheduler = Scheduler::new(
        concurrency,
        high_prio_stream,
        priorities,
        slos,
        closed_loop_batch_size,
    );
    let mut builder = Builder::new()
        .gpu_ids(vec![0])
        .synchronize_interval(synchronize_interval)
        .warmup_times(1)
        .scheduler(
            move |batcher_stat, batcher_ops, executor_stat, executor_ops, model_profile| {
                scheduler.schedule(
                    batcher_stat,
                    batcher_ops,
                    executor_stat,
                    executor_ops,
                    model_profile,
                )
            },
        );
    let trace_model = matches.value_of("trace_model").unwrap();
    for i in 0..1 {
        let path = trace_model.to_string();
        builder = builder.register_model(ModelInfo::new(i, move |handle| {
            Box::new(Model::new(&path, handle))
        }));
        builder = builder.register_request(RequestInfo::new(i));
    }
    let closed_loop_model = matches.value_of("closed_loop_model").unwrap();
    for i in 1..copies + 1 {
        let path = closed_loop_model.to_string();
        builder = builder.register_model(ModelInfo::new(i, move |handle| {
            Box::new(Model::new(&path, handle))
        }));
        builder = builder.register_request(RequestInfo::new(i));
    }
    let orchestrator = builder.build();
    let tx = orchestrator.run();
    let grpc_service = grpc::Service::new(tx);
    println!("service is ready");
    Server::builder()
        .add_service(InferServer::new(grpc_service))
        .serve("127.0.0.1:8080".parse().unwrap())
        .await
        .unwrap();
}
