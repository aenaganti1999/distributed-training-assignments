# GPU Memory Investigation

## Overview

This project investigates whether a **single 80 GB GPU** can train different Large Language Models (LLMs) ranging from **7 billion** to **175 billion** parameters.

Instead of relying on published hardware requirements or benchmark tables, this assignment estimates memory requirements **from first principles**. The objective is to understand **why large models require multiple GPUs** and what actually occupies GPU memory during training.

---

# Problem Statement

Given an **80 GB GPU**, determine whether it can train the following models:

* 7B
* 13B
* 34B
* 70B
* 175B

For each model, explain:

1. Whether the model weights fit into GPU memory.
2. Whether the model can actually be trained.
3. Why or why not.

---

# Learning Objectives

After completing this assignment, you should understand:

* Why parameter count alone is not enough to estimate GPU memory.
* The difference between loading a model and training a model.
* Why optimizer states and gradients consume significant memory.
* Why large language models require distributed training techniques.

---

# First-Principles Approach

Rather than memorizing hardware requirements, we estimate memory usage mathematically.

We begin with the assumption that model weights are stored in **FP16** or **BF16** precision.

### FP16/BF16

Each parameter occupies

```
2 bytes
```

Therefore,

```
Weight Memory
=
Number of Parameters × 2 bytes
```

Since

```
1 Billion Parameters
≈
1 Billion Bytes
≈
1 GB
```

The calculation becomes

```
Weight Memory (GB)

≈

Parameters (Billions) × 2
```

Example

```
7B

7 × 2

=

14 GB
```

---

# Why Weight Memory Is Not Enough

A common misconception is that if the model weights fit into GPU memory, then the model can be trained.

This is incorrect.

During training, the GPU stores much more than the model weights.

The major memory components include:

* Model weights
* Gradients
* FP32 master weights
* Adam optimizer states
* Activations
* Temporary CUDA buffers

---

# Memory Breakdown

For standard mixed-precision Adam training:

| Component           | Bytes per Parameter |
| ------------------- | ------------------: |
| FP16/BF16 weights   |                   2 |
| Gradients           |                   2 |
| FP32 master weights |                   4 |
| Adam first moment   |                   4 |
| Adam second moment  |                   4 |
| **Total**           |        **16 bytes** |

Therefore,

```
Training Memory

≈

Parameters × 16 bytes
```

This estimate **does not include activation memory**, making it a lower bound for actual training memory.

---

# Memory Calculations

## Weight Memory

| Model | Weight Memory |
| ----- | ------------: |
| 7B    |         14 GB |
| 13B   |         26 GB |
| 34B   |         68 GB |
| 70B   |        140 GB |
| 175B  |        350 GB |

Only considering model weights:

* 7B fits.
* 13B fits.
* 34B fits.
* 70B does not fit.
* 175B does not fit.

However, this does **not** mean the models can be trained.

---

## Estimated Training Memory

Using

```
Parameters × 16 bytes
```

| Model | Estimated Training Memory |
| ----- | ------------------------: |
| 7B    |                    112 GB |
| 13B   |                    208 GB |
| 34B   |                    544 GB |
| 70B   |         1120 GB (1.12 TB) |
| 175B  |          2800 GB (2.8 TB) |

---

# Final Results

| Model | Weights Fit? | Full Training Possible? |
| ----- | ------------ | ----------------------- |
| 7B    | Yes          | No                      |
| 13B   | Yes          | No                      |
| 34B   | Yes          | No                      |
| 70B   | No           | No                      |
| 175B  | No           | No                      |

---

# Why Training Requires More Memory

During the forward pass, every transformer layer produces intermediate outputs called **activations**.

These activations must be stored because the backward pass needs them to compute gradients.

Activation memory depends on several factors:

* Batch size
* Sequence length
* Hidden dimension
* Number of transformer layers
* Precision
* Gradient checkpointing

As these values increase, activation memory can become extremely large.

Therefore, even if the fixed training-state memory fits, the model may still run out of memory during training.

---

# Difference Between Loading and Training

Loading a model into memory only requires storing the model weights.

Training additionally requires:

* Computing gradients
* Updating optimizer states
* Saving activations
* Allocating temporary computation buffers

This is why training requires significantly more GPU memory than inference.

---

# Why Large Models Use Multiple GPUs

Large language models are typically trained using distributed techniques such as:

* Data Parallelism
* Tensor Parallelism
* Pipeline Parallelism
* Fully Sharded Data Parallel (FSDP)
* DeepSpeed ZeRO

These techniques divide model parameters, optimizer states, or computations across multiple GPUs so that no single GPU has to store the entire training state.

---

# Project Structure

```
gpu-memory-investigation/
│
├── gpu_memory_investigation.py
├── README.md
```

---

# Running the Project

Clone the repository and execute

```bash
python gpu_memory_investigation.py
```

The program prints:

* Weight memory for FP32, FP16, and BF16
* Estimated mixed-precision Adam training memory
* Whether each model fits within an 80 GB GPU
* A detailed explanation for every model
* A final summary

---

# Key Takeaways

* GPU memory requirements are much larger than model weights alone.
* Mixed-precision Adam training typically requires around **16 bytes per parameter** before activation memory.
* A 7B model's weights fit within 80 GB, but full training does not.
* Larger models require distributed training because one GPU cannot store all training states.
* Understanding memory from first principles is more valuable than memorizing published hardware requirements.

---

# Conclusion

This assignment demonstrates that **fitting model weights into GPU memory is not sufficient for training**.

While smaller models such as **7B**, **13B**, and **34B** can load their weights onto an 80 GB GPU, the additional memory required for gradients, optimizer states, master weights, activations, and temporary buffers exceeds the available memory.

Consequently, modern large language models rely on distributed training strategies and memory optimization techniques to make training feasible.

# Memory Optimization Techniques Used in Practice

The memory calculations in this assignment assume standard mixed-precision Adam training without any optimizations. In practice, modern LLM training uses several techniques to reduce GPU memory consumption and enable training of large models.

## Gradient Checkpointing

Instead of storing all intermediate activations during the forward pass, only selected activations are saved. The missing activations are recomputed during backpropagation, reducing activation memory at the cost of additional computation.

## 8-bit Optimizers

Optimizer states, such as Adam's first and second moment estimates, are normally stored in FP32. Using 8-bit optimizers significantly reduces the memory required for optimizer states while maintaining similar model quality.

## FlashAttention

FlashAttention implements a more memory-efficient attention algorithm by avoiding the storage of large attention matrices in GPU memory. This reduces memory usage and often improves training speed.

## CPU Offloading

Some training states, such as optimizer states or gradients, can be moved from GPU memory to CPU memory when they are not immediately needed. This allows larger models to be trained on limited GPU memory, although data transfers introduce some performance overhead.

## ZeRO (Zero Redundancy Optimizer)

ZeRO partitions optimizer states, gradients, and model parameters across multiple GPUs instead of storing complete copies on every device. Different ZeRO stages progressively reduce memory usage by distributing more components.

## FSDP (Fully Sharded Data Parallel)

FSDP extends the idea of parameter sharding by distributing model parameters, gradients, and optimizer states across all participating GPUs. Each GPU only stores a small portion of the model, enabling training of much larger networks.

## LoRA and QLoRA

Rather than updating every model parameter, LoRA trains only small low-rank adapter matrices while keeping the original model weights frozen. QLoRA further reduces memory by storing the pretrained model in 4-bit precision during fine-tuning. These approaches make it possible to fine-tune very large language models on a single high-memory GPU.
