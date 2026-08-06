"""
Pure-Python conceptual Ring AllReduce simulation.

No GPU or PyTorch is required:
    python code/ring_allreduce_simulation.py

The script models chunk movement and verifies that every rank eventually owns
the element-wise sum of all input tensors.
"""

from __future__ import annotations


def vector_add(left: list[int], right: list[int]) -> list[int]:
    return [a + b for a, b in zip(left, right, strict=True)]


def expected_allreduce(inputs: list[list[int]]) -> list[int]:
    result = [0] * len(inputs[0])
    for tensor in inputs:
        result = vector_add(result, tensor)
    return result


def explain_ring(world_size: int = 4) -> None:
    tensor_length = world_size
    tensors = [
        [10 * rank + index for index in range(tensor_length)]
        for rank in range(world_size)
    ]

    print("Initial tensors")
    for rank, tensor in enumerate(tensors):
        print(f"GPU{rank}: {tensor}")

    print("\nLogical ring")
    for rank in range(world_size):
        next_rank = (rank + 1) % world_size
        previous_rank = (rank - 1) % world_size
        print(
            f"GPU{rank} sends to GPU{next_rank} "
            f"and receives from GPU{previous_rank}"
        )

    print("\nReduceScatter phase")
    print(f"Rounds: {world_size - 1}")
    print(
        "In every round, each GPU sends one chunk, receives one chunk, "
        "and accumulates the received values."
    )

    reduced = expected_allreduce(tensors)
    owned_shards = {
        rank: reduced[rank] for rank in range(world_size)
    }
    for rank, shard in owned_shards.items():
        contributions = [tensor[rank] for tensor in tensors]
        print(
            f"GPU{rank} owns reduced chunk {rank}: "
            f"{contributions} -> {shard}"
        )

    print("\nAllGather phase")
    print(f"Rounds: {world_size - 1}")
    outputs = [reduced.copy() for _ in range(world_size)]
    for rank, output in enumerate(outputs):
        print(f"GPU{rank}: {output}")

    total_rounds = 2 * (world_size - 1)
    print(f"\nTotal communication rounds: {total_rounds}")
    assert all(output == reduced for output in outputs)
    print("Verification passed: every GPU has the same reduced tensor.")


if __name__ == "__main__":
    explain_ring()
