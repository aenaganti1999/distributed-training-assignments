# Reflection — Assignment 06

## What I understood before this assignment

Before this investigation, I understood data parallelism mainly as:

```text
copy the model
split the batch
run training on multiple GPUs
average the gradients
```

That description was mathematically correct but incomplete. It did not explain how expensive gradient averaging becomes, how the tensors physically move, or why adding GPUs can stop improving throughput.

## The most important realization

The gradient synchronization is not an optional reporting step. It is part of the training algorithm.

Every GPU sees a different mini-batch, so every GPU calculates a different local gradient. If those gradients are not combined before the optimizer update, the replicas diverge and the system stops training one shared model.

## What NCCL actually contributes

NCCL does not decide how the model is trained. It does not implement DDP, FSDP, ZeRO, or the optimizer. It supplies efficient GPU communication primitives that those systems use.

The distinction I now understand is:

```text
DDP/FSDP/DeepSpeed:
decide what data must move and when

NCCL:
moves and combines that data efficiently
```

## What surprised me

A 28 GB gradient tensor does not mean only 28 GB of communication. With eight GPUs using Ring AllReduce, each GPU sends approximately 49 GB and receives approximately 49 GB during one gradient synchronization. Across the cluster, the aggregate sent payload is about 392 GB per iteration.

The model still ends with one 28 GB reduced gradient tensor on each GPU. The extra traffic comes from forwarding chunks through multiple communication steps.

## Ring AllReduce in my own words

Ring AllReduce first divides a tensor into chunks. During ReduceScatter, partial sums move around the ring until every chunk has been fully reduced and assigned to one GPU. During AllGather, those completed chunks circulate again until every GPU has the complete reduced tensor.

For `p` GPUs, there are:

```text
p - 1 ReduceScatter steps
p - 1 AllGather steps
2(p - 1) total steps
```

The design avoids one central communication bottleneck and keeps every rank participating.

## Why scaling becomes difficult

Adding GPUs reduces local computation per GPU, but the model's gradient size does not shrink merely because more data-parallel replicas were added. As computation becomes shorter, synchronization forms a larger fraction of the iteration.

The practical goal is therefore not simply “use more GPUs.” It is:

```text
perform enough useful computation per GPU
while hiding or minimizing unavoidable communication
```

## Difference between the main collectives

- Broadcast: one rank's tensor goes to all ranks.
- Reduce: all ranks' tensors are combined at one root.
- AllReduce: all ranks' tensors are combined and returned to every rank.
- AllGather: different shards are collected on every rank.
- ReduceScatter: full inputs are reduced, but each rank keeps only one output shard.

## How this connects to later topics

This assignment prepares me for FSDP and DeepSpeed because their memory savings are created through sharding, and sharding creates communication requirements.

FSDP and ZeRO do not simply reduce memory for free. They trade replication for operations such as:

```text
AllGather parameters
ReduceScatter gradients
```

I should evaluate a distributed strategy using both:

```text
memory saved
communication introduced
```

## Final one-sentence answer

NCCL is necessary because distributed GPU training cannot remain one coherent training process unless large tensors are exchanged and combined efficiently enough that communication does not leave the GPUs waiting most of the time.
