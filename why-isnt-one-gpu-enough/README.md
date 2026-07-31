# Assignment 1: Why Isn't One GPU Enough?

## Objective

The goal of this assignment is to estimate the GPU memory required to store and train a large language model such as Llama-70B.

The assignment answers the following questions:

1. How much memory do the model parameters require?
2. What else occupies GPU memory during training?
3. Why can inference run on fewer GPUs than training?
4. Why can't GPU memory simply be increased forever?

## Parameter Memory Calculation

The memory needed for model parameters is calculated using:

\[
\text{Memory} =
\text{Number of Parameters}
\times
\text{Bytes per Parameter}
\]

For Llama-70B:

### FP32

\[
70 \text{ billion} \times 4 \text{ bytes}
=
280 \text{ GB}
\]

### FP16

\[
70 \text{ billion} \times 2 \text{ bytes}
=
140 \text{ GB}
\]

### BF16

\[
70 \text{ billion} \times 2 \text{ bytes}
=
140 \text{ GB}
\]

Therefore, even FP16 or BF16 parameters alone require approximately 140 GB of memory.
This is already larger than the memory available on many individual GPUs.

## Memory Components During Training

### 1. Parameters

Parameters are the weights learned by the model.

Llama-70B contains approximately 70 billion parameters. These weights must remain in memory during both training and inference.

### 2. Gradients

During backpropagation, the model calculates the derivative of the loss with respect to every trainable parameter.

For every parameter \(w\), training computes:

\[
\frac{\partial L}{\partial w}
\]

These gradients must be stored before the optimizer updates the parameters.

### 3. Optimizer States

Optimizers such as Adam store additional information for every parameter.

Adam stores:

- First moment estimate
- Second moment estimate

These are commonly stored in FP32.

Therefore, Adam requires approximately:

\[
4 + 4 = 8 \text{ bytes per parameter}
\]

For 70 billion parameters:

\[
70B \times 8 = 560 \text{ GB}
\]

### 4. Activations

Activations are the intermediate outputs produced by each neural-network layer during the forward pass.

They must often be retained because backpropagation needs them to calculate gradients.

Activation memory depends on:

- Batch size
- Sequence length
- Hidden dimension
- Number of layers
- Numerical precision
- Attention implementation
- Gradient checkpointing

### Important Limitation of the Activation Estimate

The activation calculation above is intentionally simplified.

It estimates only one hidden-state tensor per layer. A real transformer training run may also store:

- Attention query, key, and value tensors
- Attention scores
- Softmax outputs
- MLP intermediate tensors
- Normalization results
- Dropout masks
- Temporary CUDA buffers

Therefore, actual activation memory can be much larger.

The correct conclusion is not that activations always use one fixed number. The correct conclusion is that activation memory changes based on the training configuration.

**Plot**

The chart shows that activation memory increases approximately linearly with batch size.

A larger batch processes more sequences at the same time, so more intermediate values must remain in GPU memory.

**How Squence length effects**

Increasing the sequence length increases the number of token representations stored by every transformer layer.

The simplified hidden-state estimate grows linearly with sequence length.

However, standard attention may also create attention-score matrices whose memory grows approximately with:

\[
O(S^2)
\]

where \(S\) is the sequence length.

This means long-context training can create very high memory requirements.

**why inference uses fewer GPUs ?**
## Why Can Inference Run on Fewer GPUs Than Training?

Training requires both a forward pass and a backward pass.

During training, the system must store:

- Model parameters
- Gradients
- Optimizer states
- Activations needed for backpropagation
- Temporary computation buffers

Inference only performs the forward pass.

Inference usually does not require:

- Gradients
- Optimizer states
- FP32 master weights
- Saved activations for backpropagation

Therefore, inference consumes much less memory than training.

Inference still requires memory for:

- Model parameters
- Temporary forward-pass tensors
- KV cache
- Runtime and framework overhead

Quantization can also reduce parameter memory.

For example:

| Format | Approximate Memory for 70B Parameters |
|---|---:|
| FP32 | 280 GB |
| FP16/BF16 | 140 GB |
| INT8 | 70 GB |
| INT4 | 35 GB |

This is one reason a quantized model may run on fewer GPUs during inference.

**Warning**

These GPU-count estimates are simplified.

They only divide required memory by usable memory per GPU.

Real distributed training also depends on:

- Tensor-parallel communication
- Data-parallel replication
- Pipeline stages
- Activation memory
- Communication buffers
- CUDA and framework overhead
- Memory fragmentation
- Sequence length and batch size

Distributed training frameworks may shard parameters, gradients, and optimizer states differently.

## Why Can't We Simply Increase GPU RAM Forever?

### 1. Cost

High-bandwidth GPU memory is expensive. Increasing memory capacity increases the manufacturing and purchase cost of the GPU.

### 2. Physical Packaging Limits

GPU packages have limited physical area. Memory stacks, interconnects, and computing components must fit within the hardware package.

### 3. Memory Bandwidth

A larger memory capacity is not useful if the GPU cannot move data quickly enough.

Large neural networks require very high memory bandwidth to continuously feed data to the GPU compute units.

### 4. Power Consumption

More memory and compute require more electricity.

Large GPU systems already consume significant power, so continuously increasing capacity creates data-center power-delivery challenges.

### 5. Heat and Cooling

More powerful hardware produces more heat.

Cooling extremely large single GPUs becomes difficult and expensive.

### 6. Manufacturing Complexity

Larger chips and more complex memory systems are harder to manufacture reliably.

Manufacturing defects also become more costly when the hardware package is extremely large.

### 7. Computation Would Still Be Slow

Even if one GPU could hold the complete model, one GPU would still need to perform all model computations.

Training a large model on one device could take an impractically long time.

Multiple GPUs are used to distribute both:

- Memory
- Computation

### 8. Reliability

A single extremely large GPU creates a single point of failure.

Distributed systems provide better scalability and allow the workload to be divided across many devices.

# Conclusion

Llama-70B contains approximately 70 billion parameters.

The parameters alone require approximately:

- 280 GB in FP32
- 140 GB in FP16
- 140 GB in BF16

Training requires much more memory than storing parameters alone.

For simplified FP32 training with Adam:

- Parameters: 280 GB
- Gradients: 280 GB
- Optimizer states: 560 GB
- Total before activations: 1,120 GB

Mixed-precision training reduces the memory required for some tensors, but it may also maintain FP32 master weights and FP32 optimizer states.

Activations add additional memory, and their size depends on batch size, sequence length, number of layers, hidden dimension, and the training implementation.

Inference can run on fewer GPUs because it does not require gradients, optimizer states, or saved activations for backpropagation. Quantization can reduce inference memory even further.

One GPU is often not enough because large-model training requires both a large amount of memory and enormous computational power. Distributed training allows parameters, gradients, optimizer states, activations, and computation to be divided across multiple GPUs.

| Memory Component  | Training                    | Inference         |
| ----------------- | --------------------------- | ----------------- |
| Parameters        | Yes                         | Yes               |
| Gradients         | Yes                         | No                |
| Optimizer States  | Yes                         | No                |
| Master Weights    | Often                       | No                |
| Activations       | Saved for backpropagation   | Temporary only    |
| KV Cache          | Usually not dominant        | Yes               |
| Temporary Buffers | Yes                         | Yes               |
