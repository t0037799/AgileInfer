//! An example gRPC front API for AgileInfer.
//! You can try this at the beginning.

tonic::include_proto!("api");

use crossbeam::channel;
use orchestrator::Command;
use std::thread;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

/// AgileInfer's gRPC service
pub struct Service {
    tx: channel::Sender<Command>,
}

impl Service {
    /// Create a Service
    pub fn new(tx: channel::Sender<Command>) -> Self {
        Service { tx }
    }
}

#[tonic::async_trait]
impl infer_server::Infer for Service {
    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let request = request.into_inner();
        let compressed_input = request.compressed_input;
        let batcher_id = request.batcher_id as usize;
        let (tx, rx) = channel::bounded(1);
        self.tx.send(Command::NewRequest(tx, batcher_id)).unwrap();
        let request = rx.recv().unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        thread::spawn(move || {
            let input = lz4_flex::decompress_size_prepended(&compressed_input).unwrap();
            request.send(input).unwrap();
            let response = request.recv();
            let _ = tx.send(response);
        });
        let response = rx.await.unwrap();
        match response {
            Ok(raw_output) => Ok(Response::new(InferResponse {
                status_code: 0,
                raw_output,
            })),
            Err(_) => Ok(Response::new(InferResponse {
                status_code: 1,
                raw_output: vec![],
            })),
        }
    }
}
