use agile_infer::grpc::infer_client::InferClient;
use agile_infer::grpc::InferRequest;
use clap::{App, Arg};
use lz4_flex::compress_prepend_size;
use std::time::{Duration, Instant};
use std::{
    fs::File,
    io::prelude::*,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub fn bytes_to_vec<T>(orig: Vec<u8>) -> Vec<T> {
    let slice = orig.leak();
    let size = slice.len() / std::mem::size_of::<T>();
    let ptr = slice.as_mut_ptr();
    unsafe { Vec::from_raw_parts(ptr as *mut T, size, size) }
}

fn read_trace(trace_str: &str, now: Instant) -> Vec<(usize, Instant)> {
    trace_str
        .lines()
        .map(|line| {
            let mut split = line.split(' ');
            let batcher_id = split.next().unwrap().parse::<usize>().unwrap();
            let duration = split.next().unwrap().parse::<u64>().unwrap();
            (batcher_id, now + Duration::from_nanos(duration))
        })
        .collect()
}

async fn run_trace(input: &[u8], trace_str: &str) {
    let client = InferClient::connect("http://127.0.0.1:8080").await.unwrap();
    let trace = read_trace(trace_str, Instant::now());
    let input = input.to_vec();
    let done = Arc::new(AtomicUsize::new(0));
    let drop = Arc::new(AtomicUsize::new(0));
    let compressed_input = compress_prepend_size(&input);
    for (batcher_id, next) in trace {
        tokio::time::sleep_until(tokio::time::Instant::from_std(next)).await;
        let mut client = client.clone();
        let done = done.clone();
        let drop = drop.clone();
        let compressed_input = compressed_input.clone();
        tokio::spawn(async move {
            let request = tonic::Request::new(InferRequest {
                batcher_id: batcher_id as u64,
                compressed_input,
            });
            let s = Instant::now();
            let fut = tokio::time::timeout(Duration::from_millis(200), client.infer(request));
            match fut.await {
                Err(_) => {
                    println!("Trace miss deadline by 200 ms")
                }
                Ok(res) => {
                    let resp = res.unwrap().into_inner();
                    if resp.status_code == 0 {
                        println!("done {:?}", s.elapsed());
                        done.fetch_add(1, Ordering::SeqCst);
                    } else {
                        println!("drop");
                        drop.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
    }
    println!(
        "End of Trace done/drop {}/{}",
        done.load(Ordering::SeqCst),
        drop.load(Ordering::SeqCst)
    );
}

async fn closed_loop(
    count: Arc<AtomicUsize>,
    input: &[u8],
    model_copies: usize,
    concurrency: usize,
) {
    let client = InferClient::connect("http://127.0.0.1:8080").await.unwrap();
    let input = input.to_vec();
    for batcher_id in 1..model_copies + 1 {
        for _ in 0..concurrency {
            let mut client = client.clone();
            let count = count.clone();
            let input = input.clone();
            tokio::spawn(async move {
                let compressed_input = compress_prepend_size(&input);
                loop {
                    let request = tonic::Request::new(InferRequest {
                        batcher_id: batcher_id as u64,
                        compressed_input: compressed_input.clone(),
                    });

                    let fut = tokio::time::timeout(Duration::from_secs(10), client.infer(request));
                    match fut.await {
                        Err(_) => {
                            println!("Closed loop miss deadline by 10 sec")
                        }
                        Ok(res) => {
                            let resp = res.unwrap().into_inner();
                            if resp.status_code == 0 {
                                count.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                }
            });
        }
    }
}

async fn telemetry(count: Arc<AtomicUsize>) {
    let mut last = 0;
    loop {
        let this = count.load(Ordering::SeqCst);
        println!("throughput {}/s", (this - last) as f64 / 5.0);
        last = this;
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[tokio::main]
async fn main() {
    let matches = App::new("Agile Infer Client")
        .about("An example client for evaluating Agile Infer")
        .arg(
            Arg::with_name("model_copies")
                .short("m")
                .help(
                    "How many copies of model used in closed loop request, \
                      this should correspond to server's setting",
                )
                .default_value("4")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("concurrency")
                .short("c")
                .help(
                    "How many requests concurrently send to the same closed_loop model copy, \
                      this should be greater or equal to server's closed_loop batch size",
                )
                .default_value("16")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("trace")
                .short("t")
                .help("Path to trace file for trace request, set \"disable\" to disable trace")
                .default_value("resources/trace_file.txt")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("input")
                .short("i")
                .help("Path to model's input")
                .default_value("resources/n01768244_309.JPEG")
                .takes_value(true),
        )
        .get_matches();
    let count = Arc::new(AtomicUsize::new(0));
    tokio::spawn(telemetry(count.clone()));
    let input = {
        let mut buf = vec![];
        let mut f = File::open(matches.value_of("input").unwrap()).unwrap();
        f.read_to_end(&mut buf).unwrap();
        buf
    };
    let trace = {
        let mut s = String::new();
        let path = matches.value_of("trace").unwrap();
        if path != "disable" {
            let mut f = File::open(matches.value_of("trace").unwrap())
                .expect("Should be a valid trace text file, or disable");
            f.read_to_string(&mut s).unwrap();
        }
        s
    };
    let move_input = input.clone();
    tokio::spawn(async move {
        run_trace(&move_input, &trace).await;
    });
    let copies = matches.value_of("model_copies").unwrap().parse().unwrap();
    let concurrency = matches.value_of("concurrency").unwrap().parse().unwrap();
    tokio::spawn(async move {
        closed_loop(count, &input, copies, concurrency).await;
    });
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1000)).await;
    }
}
