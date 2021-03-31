# AgileInfer

![badge](https://github.com/t0037799/AgileInfer/actions/workflows/ci.yaml/badge.svg)

A DNN serving system research prototype that uses CUDA stream synchronization to achieve higher throughput and lower preemption latency on NVIDIA GPUs.

## Get Started

### Build

#### From source

**prerequsite:** CUDA library, cargo

```
cargo build --release
```

### Run

```
./target/release/server
# wait until server print "service is ready"
# open  another terminal
./target/release/client
```


## Supported Model

AgileInfer supports model format used in another DNN serving system, [Clockwork](https://gitlab.mpi-sws.org/cld/ml/clockwork).
Models are converted from [TVM](https://github.com/apache/tvm) compiled models.

Because of Github's upload limit, we don't include all model's weight.
You can refer to [Clockwork's modelzoo](https://gitlab.mpi-sws.org/cld/ml/clockwork-modelzoo-volta), if you want to try other models.
