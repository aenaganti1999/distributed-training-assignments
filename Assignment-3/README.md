# Assignment 03 — The GPU Communication Mystery (NCCL)

## Scenario

Our data-parallel training job is correct, but adding GPUs gives disappointing speedup. Each GPU finishes its local forward and backward computation, then waits while gradients are synchronized. The investigation shows that the bottleneck is no longer only matrix multiplication. It is moving large tensors between GPUs quickly enough to keep every GPU working.

This report explains why GPU communication is necessary, what collective communication operations do, how Ring AllReduce works, how much traffic a 7-billion-parameter model can create, and where NCCL fits beneath PyTorch Distributed, FSDP, and DeepSpeed.

---

# Executive Summary

In data parallelism, every GPU owns a full model replica but processes a different mini-batch. Because the mini-batches are different, the gradients calculated by the GPUs are also different. Before the optimizer updates the model, those gradients must be combined so that every model replica applies the same update.

Without synchronization, the replicas stop representing one shared model. GPU 0 follows the gradient from its mini-batch, GPU 1 follows another gradient, and their parameters diverge after the first optimizer step.

NCCL is necessary because this synchronization moves very large tensors and must happen repeatedly. NCCL provides high-performance implementations of collective communication operations such as AllReduce, AllGather, ReduceScatter, Broadcast, and Reduce. It chooses communication paths and algorithms suited to NVIDIA GPU systems and transports data directly between GPUs over links such as NVLink, NVSwitch, PCIe, and network fabrics.

NCCL does not make communication free. It makes unavoidable communication substantially more efficient and exposes it through collective operations that distributed training frameworks can schedule, overlap, bucket, and compose.

---

# Part 1 — The Problem

## 1. Why do GPUs need to communicate?

Consider synchronous data-parallel training with four GPUs:

```text
GPU0: model copy + batch A
GPU1: model copy + batch B
GPU2: model copy + batch C
GPU3: model copy + batch D
```

All four model replicas begin an iteration with identical parameters. Each GPU performs a forward pass on different samples, calculates a local loss, and runs backpropagation. The resulting local gradients answer slightly different questions because each GPU saw different data.

For parameter `w`, the GPUs may calculate:

```text
GPU0: grad(w) =  0.20
GPU1: grad(w) = -0.05
GPU2: grad(w) =  0.15
GPU3: grad(w) =  0.10
```

The distributed job should behave approximately like one larger batch. It therefore combines the local gradients:

```text
global gradient = (0.20 - 0.05 + 0.15 + 0.10) / 4
                = 0.10
```

Every GPU must use that same global gradient when updating `w`.

Communication also becomes necessary in model-sharded systems. If weights, gradients, optimizer states, activations, or tensor dimensions are partitioned across GPUs, a GPU may need data owned by another GPU before it can continue computation.

## 2. When during training do GPUs communicate?

The exact points depend on the parallelism strategy.

### Distributed Data Parallel

The important communication occurs during the backward pass. As gradients for parameter groups become ready, they can be grouped into buckets and synchronized. Frameworks often overlap communication for earlier buckets with backward computation for later layers.

Conceptual sequence:

```text
forward pass
    ↓
loss
    ↓
backward pass
    ↓
gradient bucket becomes ready
    ↓
AllReduce bucket
    ↓
next buckets become ready and are reduced
    ↓
optimizer step using synchronized gradients
```

### Fully Sharded Data Parallel and ZeRO-style training

Communication may occur several times:

```text
before a layer computes:
    AllGather required parameter shards

after gradients are produced:
    ReduceScatter gradients back to their owners

during checkpointing or state reconstruction:
    AllGather selected state
```

### Tensor parallelism

Communication may occur inside the forward and backward execution of an individual Transformer layer because one matrix operation is split across GPUs.

## 3. What information is exchanged?

Depending on the training strategy, GPUs exchange:

- gradients;
- parameter or weight shards;
- optimizer-state shards;
- activations;
- partial matrix-operation outputs;
- scalar metadata or synchronization signals;
- occasionally model states for initialization or checkpoint reconstruction.

For ordinary synchronous data parallelism, gradients are the primary large payload.

## 4. What happens if GPUs never communicate?

In data parallelism, each GPU becomes an independent training run after the first update.

```text
Initial state:
W0 = W1 = W2 = W3

After independent updates:
W0' ≠ W1' ≠ W2' ≠ W3'
```

Consequences:

1. There is no longer one global model.
2. The effective global batch is lost.
3. Checkpoint selection becomes ambiguous.
4. Predictions differ depending on which GPU's replica is used.
5. Later synchronization becomes mathematically inconsistent unless a deliberate local-training algorithm is being used.

Communication is therefore part of the training algorithm, not merely an infrastructure convenience.

---

# Part 2 — Collective Communication

A **collective** is an operation in which a defined group of ranks participates. A rank is usually one process associated with one GPU.

## Broadcast

### Input

One root GPU owns the source tensor. Other GPUs provide receive space.

```text
GPU0: W
GPU1: empty
GPU2: empty
GPU3: empty
```

### Output

Every GPU receives the root's tensor.

```text
GPU0: W
GPU1: W
GPU2: W
GPU3: W
```

### Real training example

At initialization, rank 0 can broadcast model parameters or configuration state so that all workers start from identical values. Broadcast can also distribute a control value or tensor generated by one designated rank.

### Important distinction

Broadcast copies one rank's value to everyone. It does not combine different values.

---

## Reduce

### Input

Every GPU owns a tensor of the same shape.

```text
GPU0: G0
GPU1: G1
GPU2: G2
GPU3: G3
```

### Output

The selected root receives the reduced result. Other ranks do not necessarily receive it.

```text
GPU0: G0 + G1 + G2 + G3
GPU1: no global result
GPU2: no global result
GPU3: no global result
```

### Real training example

Workers may reduce validation-loss totals or sample counts to rank 0 for centralized logging. A reduce could combine gradients at one root, although ordinary replicated data parallelism then still needs a way to return the result to all workers.

### Important distinction

Reduce performs the combination but places the final value only at the root.

---

## AllReduce

### Input

Every GPU owns a tensor of the same shape.

```text
GPU0: G0
GPU1: G1
GPU2: G2
GPU3: G3
```

### Output

Every GPU receives the same reduced tensor.

```text
GPU0: G0 + G1 + G2 + G3
GPU1: G0 + G1 + G2 + G3
GPU2: G0 + G1 + G2 + G3
GPU3: G0 + G1 + G2 + G3
```

The framework may divide by the world size to produce the mean gradient, or account for the averaging elsewhere.

### Real training example

Distributed Data Parallel synchronizes gradient buckets with AllReduce so every replicated model applies an equivalent optimizer update.

### Important distinction

AllReduce is logically similar to:

```text
Reduce to one rank
        +
Broadcast result to all ranks
```

An optimized implementation does not need to perform those two operations in that naïve form.

---

## AllGather

### Input

Each GPU owns a different shard.

```text
GPU0: P0
GPU1: P1
GPU2: P2
GPU3: P3
```

### Output

Every GPU receives the concatenation or collection of all shards.

```text
GPU0: [P0, P1, P2, P3]
GPU1: [P0, P1, P2, P3]
GPU2: [P0, P1, P2, P3]
GPU3: [P0, P1, P2, P3]
```

### Real training example

FSDP or ZeRO Stage 3 may AllGather parameter shards before executing a layer. Each GPU stores only a portion of the parameters at rest, but temporarily reconstructs the full parameter group needed for computation.

### Important distinction

AllGather collects distinct pieces. It does not sum them.

---

## ReduceScatter

### Input

Every GPU owns a full-sized tensor, logically divided into shards.

```text
GPU0: [G00, G01, G02, G03]
GPU1: [G10, G11, G12, G13]
GPU2: [G20, G21, G22, G23]
GPU3: [G30, G31, G32, G33]
```

### Output

Corresponding shards are reduced, and each GPU keeps only one reduced shard.

```text
GPU0: G00 + G10 + G20 + G30
GPU1: G01 + G11 + G21 + G31
GPU2: G02 + G12 + G22 + G32
GPU3: G03 + G13 + G23 + G33
```

### Real training example

FSDP and ZeRO-style systems can ReduceScatter gradients after backward computation. The gradients are summed across workers, but each worker retains only the shard for which it is responsible. This both synchronizes the gradients and preserves sharded memory ownership.

### Important distinction

ReduceScatter combines data and partitions the result. It can be viewed as the first half of Ring AllReduce.

---

# Collective Selection Guide

| Desired result | Suitable collective |
|---|---|
| Copy one rank's tensor to everyone | Broadcast |
| Combine values and keep result only at root | Reduce |
| Combine values and give result to everyone | AllReduce |
| Collect different shards on every rank | AllGather |
| Combine full inputs and leave one result shard per rank | ReduceScatter |

---

# Part 3 — Ring AllReduce

## Ring topology

```text
          ┌──────────────────────────────┐
          │                              │
          ▼                              │
       GPU0 ─────▶ GPU1 ─────▶ GPU2 ─────▶ GPU3
          ▲                              │
          └──────────────────────────────┘
```

Each rank sends to its next neighbor and receives from its previous neighbor:

```text
GPU0 sends to GPU1; receives from GPU3
GPU1 sends to GPU2; receives from GPU0
GPU2 sends to GPU3; receives from GPU1
GPU3 sends to GPU0; receives from GPU2
```

## Why Ring AllReduce exists

A naïve central-parameter-server design creates a hotspot:

```text
GPU1 ─┐
GPU2 ─┼──▶ central GPU/server ───▶ everybody
GPU3 ─┤
GPU4 ─┘
```

The central participant must receive and transmit data for all workers. Its links become the bottleneck.

Ring AllReduce distributes the work. Every GPU repeatedly sends one chunk and receives one chunk. No single rank must carry all traffic for the group.

## Step 1 — Split the tensor

With four GPUs, divide each gradient tensor into four chunks:

```text
GPU0 gradient: [A0, B0, C0, D0]
GPU1 gradient: [A1, B1, C1, D1]
GPU2 gradient: [A2, B2, C2, D2]
GPU3 gradient: [A3, B3, C3, D3]
```

The letters identify chunk positions. The numbers identify source ranks.

## Phase A — ReduceScatter

For `p` GPUs, ReduceScatter takes `p - 1` communication steps.

During each step, every GPU:

1. sends one chunk to the next GPU;
2. receives one chunk from the previous GPU;
3. adds the received values into the corresponding partial reduction;
4. forwards an appropriate partial result in the next step.

After three steps for four GPUs, each GPU owns one fully reduced chunk:

```text
GPU0 owns: A0 + A1 + A2 + A3
GPU1 owns: B0 + B1 + B2 + B3
GPU2 owns: C0 + C1 + C2 + C3
GPU3 owns: D0 + D1 + D2 + D3
```

The exact chunk-to-rank assignment can vary; the invariant is that every reduced chunk has one owner.

## Phase B — AllGather

Now the reduced chunks circulate around the ring. Again, this takes `p - 1` steps.

After three more steps, every GPU has all reduced chunks:

```text
GPU0: [AΣ, BΣ, CΣ, DΣ]
GPU1: [AΣ, BΣ, CΣ, DΣ]
GPU2: [AΣ, BΣ, CΣ, DΣ]
GPU3: [AΣ, BΣ, CΣ, DΣ]
```

where, for example:

```text
AΣ = A0 + A1 + A2 + A3
```

## What each GPU sends and receives

At every ring step, each GPU sends approximately one `1/p` tensor chunk and receives one `1/p` chunk.

It does **not** send the full tensor to every other GPU.

## Number of communication rounds

For `p` GPUs:

```text
ReduceScatter: p - 1 steps
AllGather:     p - 1 steps
Total:         2(p - 1) steps
```

For four GPUs:

```text
2(4 - 1) = 6 steps
```

For eight GPUs:

```text
2(8 - 1) = 14 steps
```

## Why it scales better

The important bandwidth property is that each rank transfers:

```text
2 × (p - 1) / p × tensor_size
```

As `p` grows, this approaches approximately:

```text
2 × tensor_size
```

Thus, per-GPU data volume does not grow linearly with the number of GPUs. Ring AllReduce can also keep all links active concurrently.

However, latency still matters because there are more communication steps as the number of ranks increases. Ring is especially effective for large tensors, where bandwidth dominates. Other algorithms may be preferable for small messages or different network topologies.

---

# Part 4 — Communication Cost

## Given

```text
Model parameters: 7 billion
Gradient size:     28 GB
GPU count:          8
```

The 28 GB value corresponds conceptually to 7 billion gradient values stored at four bytes each:

```text
7 billion × 4 bytes ≈ 28 GB
```

## Ring AllReduce estimate

For eight GPUs:

```text
chunk size = 28 GB / 8
           = 3.5 GB
```

Each GPU participates in:

```text
7 ReduceScatter steps
7 AllGather steps
14 total steps
```

During every step, each GPU sends one 3.5 GB chunk:

```text
per-GPU sent payload = 14 × 3.5 GB
                     = 49 GB
```

It also receives approximately 49 GB.

Using the standard per-rank formula:

```text
2 × (8 - 1) / 8 × 28 GB
= 49 GB transferred per GPU
```

Across eight GPUs, the sum of sent payloads is approximately:

```text
8 × 49 GB = 392 GB
```

This 392 GB is an aggregate network-traffic accounting number. It does not mean each GPU receives a 392 GB object. Every GPU begins with a 28 GB gradient tensor and finishes with a 28 GB reduced gradient tensor; the larger number reflects the repeated movement of chunks through the ring.

Real traffic can be higher due to protocol headers, alignment, padding, retries, topology effects, and multiple communication channels.

## Time intuition

If a GPU effectively sustains 25 GB/s for this operation:

```text
49 GB / 25 GB/s ≈ 1.96 seconds
```

If the backward computation itself takes one second, the iteration cannot finish in one second unless a large portion of communication is hidden behind computation.

At 200 GB/s effective bandwidth:

```text
49 GB / 200 GB/s ≈ 0.245 seconds
```

The exact values depend heavily on the hardware and topology, but the reasoning is the point: large gradient payloads make link bandwidth part of the training-time equation.

## Why communication eventually dominates computation

### 1. Local computation shrinks as work is divided

Adding GPUs usually reduces the amount of data processed by each GPU per iteration. Its local forward and backward work may fall.

The gradient tensor, however, is tied primarily to model size. A 7B model still produces approximately one gradient value per trainable parameter.

### 2. Synchronization lies on the critical path

The optimizer cannot safely update replicated parameters until the required gradient synchronization is complete. Slow ranks or slow links delay the whole group.

### 3. More participants add coordination and latency

Even when per-rank bandwidth cost stays efficient, larger groups create more communication phases, more opportunities for stragglers, and often more traffic across slower inter-node links.

### 4. Inter-node links are usually slower than on-device memory

Modern GPUs can execute tensor operations and access high-bandwidth memory extremely quickly. Moving data across PCIe or a cluster network is slower than reading local GPU memory, especially when crossing machines.

### 5. Small batches produce a poor compute-to-communication ratio

If each GPU performs little compute before synchronizing a large gradient tensor, communication occupies a large percentage of iteration time.

## Why adding GPUs gives diminishing returns

A simplified iteration model is:

```text
iteration time =
    local computation
  + exposed communication
  + synchronization delay
  + framework overhead
```

Adding GPUs reduces the local-computation term but does not eliminate the other terms. Eventually, the saved compute time is smaller than the extra communication and coordination cost.

---

# Part 5 — Real Systems

## NCCL's position in the stack

```text
Training script
      ↓
PyTorch DDP / FSDP / DeepSpeed
      ↓
distributed process-group and collective abstractions
      ↓
NCCL
      ↓
NVLink / NVSwitch / PCIe / network transport
      ↓
NVIDIA GPUs
```

NCCL is the communication engine, not the full distributed-training system.

## PyTorch Distributed

PyTorch Distributed exposes process groups and collective operations. For CUDA tensors on NVIDIA GPUs, NCCL is commonly used as the backend.

### DDP behavior

Distributed Data Parallel:

1. replicates the model across ranks;
2. registers mechanisms that detect when gradients become ready;
3. groups gradients into buckets;
4. launches gradient synchronization, commonly using AllReduce;
5. attempts to overlap bucket communication with the rest of backpropagation;
6. leaves each rank with equivalent synchronized gradients.

The user works with the DDP abstraction. The process-group backend performs the actual collective communication.

## FSDP

Fully Sharded Data Parallel divides model states across ranks rather than keeping every state fully replicated.

A simplified per-layer pattern is:

```text
parameter shards at rest
        ↓
AllGather parameters needed for computation
        ↓
forward/backward computation
        ↓
ReduceScatter gradients
        ↓
each rank retains its gradient shard
```

Depending on configuration and execution, FSDP may reshard parameters after use and prefetch upcoming parameter groups. Its performance depends not only on the speed of a single collective but also on scheduling collectives so they overlap with useful computation.

NCCL provides the underlying high-performance GPU collectives. FSDP decides **what** should be sharded and **when** communication should occur.

## DeepSpeed

DeepSpeed provides distributed training optimizations including ZeRO.

Conceptually:

- ZeRO Stage 1 partitions optimizer states.
- ZeRO Stage 2 also partitions gradients.
- ZeRO Stage 3 also partitions parameters.

As partitioning becomes more aggressive, communication patterns include operations such as parameter AllGather and gradient ReduceScatter. DeepSpeed orchestrates sharding, offload, memory management, scheduling, and optimizer integration while relying on lower-level distributed communication backends for GPU data exchange.

NCCL solves fast communication between NVIDIA GPUs. DeepSpeed solves a broader systems problem: how to train models efficiently by coordinating computation, memory, parallelism, and communication.

## Why users rarely call NCCL directly

Most ML engineers need semantic operations such as:

```text
"keep these replicated gradients synchronized"
"reconstruct this parameter group before the layer runs"
"return reduced gradient shards to their owners"
```

They generally should not need to manually manage:

- rank topology;
- communicators;
- stream ordering;
- exact chunk schedules;
- transport selection;
- link utilization;
- error propagation;
- collective launch coordination;
- gradient bucket construction;
- overlap with autograd;
- sharding lifecycle.

Frameworks turn model-level intent into collectives and call the communication backend. This provides safer integration with autograd, optimizers, checkpointing, mixed precision, failure handling, and model state management.

Direct NCCL programming is more appropriate for framework authors, communication researchers, and developers implementing custom distributed runtimes.

---

# What NCCL Solves—and What It Does Not

## NCCL solves

- efficient GPU collective communication;
- topology-aware use of available NVIDIA GPU interconnects;
- coordinated movement and reduction of large tensors;
- reusable primitives for distributed-training frameworks;
- asynchronous collective launches that frameworks can overlap with compute.

## NCCL does not solve by itself

- deciding the parallelism strategy;
- determining how to shard a model;
- choosing gradient bucket sizes;
- fixing an undersized per-GPU batch;
- eliminating network limits;
- handling every training failure;
- guaranteeing linear scaling;
- making all collectives equally efficient for every message size and topology.

---

# Communication Bottleneck Investigation Checklist

When multi-GPU scaling is poor, investigate:

1. **Compute-to-communication ratio**  
   Is each GPU doing enough computation between synchronizations?

2. **Collective duration**  
   How much iteration time is spent in AllReduce, AllGather, or ReduceScatter?

3. **Overlap**  
   Does communication overlap backward computation, or is the GPU waiting?

4. **Topology**  
   Are communicating GPUs connected through NVLink/NVSwitch, PCIe, or an inter-node network?

5. **Bucket sizes**  
   Are operations too small and latency-bound, or too large to overlap effectively?

6. **Stragglers**  
   Does one rank reach collectives later than the others?

7. **Cross-node traffic**  
   Did performance fall sharply when training expanded beyond one machine?

8. **Precision**  
   Can communication payloads safely use a lower-precision representation?

9. **Sharding strategy**  
   Is the job trading memory savings for excessive AllGather or ReduceScatter traffic?

10. **Batch size**  
    Can each GPU process more samples per synchronization without harming convergence?

---

# Final Answer: Why Is NCCL Necessary?

NCCL is necessary because distributed GPU training repeatedly requires many GPUs to exchange and combine very large tensors while remaining synchronized as one training job. A correct data-parallel algorithm needs globally consistent gradients, and sharded algorithms need weights, gradients, or activations to move between their owners and the GPUs performing computation.

General-purpose or poorly scheduled communication can leave expensive GPUs idle. NCCL supplies optimized collective communication for NVIDIA GPU systems so frameworks can move and reduce tensors using available high-bandwidth paths without routing the core payload through ordinary application logic. It does not remove communication cost; it makes that mandatory cost low enough, predictable enough, and composable enough for large-scale training frameworks to build on.

---

# Sources Consulted

This report was written as an engineering explanation rather than copied from documentation. The following official references were used to verify current framework behavior and collective definitions:

1. NVIDIA NCCL User Guide — Overview and Collective Operations.
2. PyTorch documentation — `torch.distributed`, DistributedDataParallel, and FullyShardedDataParallel.
3. DeepSpeed documentation — Training overview and ZeRO tutorial.

Accessed August 6, 2026.
