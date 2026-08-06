# Code Lab

## What this code proves

The report explains the communication problem. These programs make it visible:

| Program | What it demonstrates |
|---|---|
| `collectives_demo.py` | Broadcast, Reduce, AllReduce, AllGather, and ReduceScatter |
| `ddp_training_demo.py` | DDP automatically synchronizes gradients during `backward()` |
| `allreduce_benchmark.py` | Communication time increases with tensor size |
| `ring_allreduce_simulation.py` | Conceptual ReduceScatter + AllGather ring behavior |

## Environment limitation

NCCL only runs with CUDA-capable NVIDIA GPUs. A Mac cannot execute the NCCL backend because Apple Silicon does not provide CUDA/NCCL.

The same scripts support `gloo` so that the collective semantics and DDP behavior can still be studied on a Mac. This verifies the distributed algorithm, but it does not measure NVIDIA GPU communication performance.

## Installation

```bash
cd Assignment-06
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## Run everything on a Mac or CPU machine

```bash
bash code/run_all.sh 4 gloo
```

## Run on four NVIDIA GPUs with NCCL

```bash
bash code/run_all.sh 4 nccl
```

## Run one experiment at a time

### Collective operations

```bash
torchrun --standalone --nproc-per-node=4 \
  code/collectives_demo.py --backend gloo
```

On an NVIDIA GPU machine:

```bash
torchrun --standalone --nproc-per-node=4 \
  code/collectives_demo.py --backend nccl
```

Expected values with four ranks:

```text
Broadcast:     [100] on every rank
Reduce sum:   [10] on rank 0
AllReduce:    [10] on every rank
AllGather:    [0, 1, 2, 3] on every rank
ReduceScatter:
  rank 0 receives 0+10+20+30 = 60
  rank 1 receives 1+11+21+31 = 64
  rank 2 receives 2+12+22+32 = 68
  rank 3 receives 3+13+23+33 = 72
```

### DDP training

```bash
torchrun --standalone --nproc-per-node=4 \
  code/ddp_training_demo.py --backend gloo
```

Every rank generates a different mini-batch. After each optimizer step, the
printed parameter checksums should remain identical. This proves that DDP
combined the gradients before the update.

### AllReduce benchmark

Use smaller sizes on a laptop:

```bash
torchrun --standalone --nproc-per-node=4 \
  code/allreduce_benchmark.py \
  --backend gloo \
  --sizes-mb 1 4 16
```

Use larger sizes on a GPU server:

```bash
torchrun --standalone --nproc-per-node=4 \
  code/allreduce_benchmark.py \
  --backend nccl \
  --sizes-mb 1 16 64 256
```

### Ring simulation

```bash
python code/ring_allreduce_simulation.py
```

## What to record in the assignment

After running the code, add these observations to `reflection.md`:

1. AllReduce gives every rank the same sum.
2. AllGather concatenates distinct shards instead of reducing them.
3. ReduceScatter reduces values but leaves only one result shard per rank.
4. DDP model checksums stay identical even though local mini-batches differ.
5. Larger tensors take longer to synchronize.
6. CPU/Gloo results demonstrate semantics; only GPU/NCCL results measure NCCL.
