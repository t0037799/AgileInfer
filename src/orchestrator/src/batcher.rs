use crossbeam::channel;
use std::{collections::VecDeque, time::Instant};

pub(crate) struct Batcher {
    requests_buffer: VecDeque<RequestInfo>,
}

impl Batcher {
    pub fn new() -> Self {
        Batcher {
            requests_buffer: VecDeque::new(),
        }
    }
    pub fn new_request(&mut self) -> Request {
        let (compressed_input_tx, compressed_input_rx) = channel::unbounded();
        let (response_tx, response_rx) = channel::unbounded();
        self.requests_buffer.push_back(RequestInfo {
            arrive_time: Instant::now(),
            compressed_input_rx,
            response_tx,
        });
        Request {
            compressed_input_tx,
            response_rx,
        }
    }

    pub fn requests_buffer(&self) -> Vec<Instant> {
        self.requests_buffer
            .iter()
            .map(|info| info.arrive_time)
            .collect()
    }

    pub fn pop_requests(&mut self, count: usize) -> Vec<RequestInfo> {
        let mut requests = vec![];
        for _ in 0..count {
            match self.requests_buffer.pop_front() {
                Some(request) => requests.push(request),
                None => break,
            }
        }
        requests
    }
}

/// RequestInfo is used by Scheduler and Executor.
pub struct RequestInfo {
    // in micro seconds, an instant since service_start_time
    arrive_time: Instant,
    compressed_input_rx: channel::Receiver<Vec<u8>>,
    response_tx: channel::Sender<Response>,
}

impl RequestInfo {
    /// send input to the Executor
    pub fn send(&self, response: Response) -> Result<(), channel::SendError<Response>> {
        self.response_tx.send(response)
    }

    /// receive output from the Executor
    pub fn recv(&self) -> Result<Vec<u8>, channel::RecvError> {
        self.compressed_input_rx.recv()
    }

    pub fn arrive_time(&self) -> Instant {
        self.arrive_time
    }
}

// for testing
#[allow(dead_code)]
pub(crate) fn fake_request(
    response_tx: channel::Sender<Response>,
    compressed_input_rx: channel::Receiver<Vec<u8>>,
) -> RequestInfo {
    RequestInfo {
        arrive_time: Instant::now(),
        compressed_input_rx,
        response_tx,
    }
}

/// Request is used by Front end service.
pub struct Request {
    compressed_input_tx: channel::Sender<Vec<u8>>,
    response_rx: channel::Receiver<Response>,
}

impl Request {
    /// send input to the Executor
    pub fn send(&self, compressed_input: Vec<u8>) -> Result<(), channel::SendError<Vec<u8>>> {
        self.compressed_input_tx.send(compressed_input)
    }

    /// receive output from the Executor
    pub fn recv(&self) -> Response {
        match self.response_rx.recv() {
            Ok(response) => response,
            Err(_) => Err(Error::Recv),
        }
    }
}

type Response = Result<Vec<u8>, Error>;

/// InferError
#[derive(Debug)]
pub enum Error {
    /// Request is dropped.
    Drop,
    /// Recv Error from Executor
    Recv,
}

#[cfg(test)]
mod test {
    use super::*;
    use std::thread;
    #[test]
    fn test_batcher() {
        let mut batcher = Batcher {
            requests_buffer: VecDeque::new(),
        };
        let (tx, rx) = channel::unbounded();
        let request = batcher.new_request();
        tx.send(request).unwrap();
        thread::spawn(move || {
            let request = rx.recv().unwrap();
            request.send(vec![0]).unwrap();
            let response = request.recv().unwrap();
            assert_eq!(response, vec![1]);
        });
        let requests = batcher.pop_requests(3);
        for request in requests {
            let input = request.recv().unwrap();
            assert_eq!(input, vec![0]);
            request.send(Ok(vec![1])).unwrap();
        }
    }
}
