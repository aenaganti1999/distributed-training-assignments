"""
Demonstrate the collective operations used in distributed training.

GPU machine:
    torchrun --standalone --nproc-per-node=4 code/collectives_demo.py --backend nccl

CPU/Mac learning mode:
    torchrun --standalone --nproc-per-node=4 code/collectives_demo.py --backend gloo

The Gloo path emulates ReduceScatter when the installed backend does not
support reduce_scatter_tensor. The NCCL path performs the real GPU collective.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="Use NCCL for NVIDIA GPUs or Gloo for a CPU learning run.",
    )
    return parser.parse_args()


def choose_backend(requested: str) -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if requested == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires a CUDA-capable NVIDIA GPU.")
    return requested


def ordered_print(rank: int, world_size: int, title: str, value: object) -> None:
    """Print rank outputs in deterministic order."""
    for current_rank in range(world_size):
        dist.barrier()
        if rank == current_rank:
            print(f"{title:<24} rank={rank}: {value}", flush=True)
    dist.barrier()


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def demonstrate_broadcast(
    rank: int, world_size: int, device: torch.device
) -> None:
    tensor = torch.tensor(
        [100.0 + rank], device=device
    )  # only rank 0 should survive
    dist.broadcast(tensor=tensor, src=0)
    ordered_print(rank, world_size, "Broadcast output", tensor_to_list(tensor))


def demonstrate_reduce(rank: int, world_size: int, device: torch.device) -> None:
    tensor = torch.tensor([float(rank + 1)], device=device)
    dist.reduce(tensor=tensor, dst=0, op=dist.ReduceOp.SUM)

    # Only rank 0's result is meaningful after Reduce.
    displayed = tensor_to_list(tensor) if rank == 0 else "not defined for non-root"
    ordered_print(rank, world_size, "Reduce output", displayed)


def demonstrate_all_reduce(
    rank: int, world_size: int, device: torch.device
) -> None:
    tensor = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    ordered_print(rank, world_size, "AllReduce output", tensor_to_list(tensor))


def demonstrate_all_gather(
    rank: int, world_size: int, device: torch.device
) -> None:
    local_shard = torch.tensor([float(rank)], device=device)
    gathered = [torch.empty_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered, local_shard)
    output = [tensor.item() for tensor in gathered]
    ordered_print(rank, world_size, "AllGather output", output)


def demonstrate_reduce_scatter(
    rank: int,
    world_size: int,
    device: torch.device,
    backend: str,
) -> None:
    # Each rank contributes a full vector. Position j is destined for rank j.
    # Rank r contributes:
    # [10*r + 0, 10*r + 1, ..., 10*r + world_size-1]
    input_tensor = torch.tensor(
        [10.0 * rank + index for index in range(world_size)],
        device=device,
    )
    output = torch.empty(1, device=device)

    try:
        dist.reduce_scatter_tensor(
            output,
            input_tensor,
            op=dist.ReduceOp.SUM,
        )
        implementation = f"native {backend}"
    except RuntimeError:
        # Learning fallback for Gloo builds without native ReduceScatter:
        # AllReduce the complete vector, then retain this rank's shard.
        reduced = input_tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        output.copy_(reduced[rank : rank + 1])
        implementation = "emulated with AllReduce + local shard"

    result = {
        "value": output.item(),
        "implementation": implementation,
    }
    ordered_print(rank, world_size, "ReduceScatter output", result)


def main() -> None:
    args = parse_args()
    backend = choose_backend(args.backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)

    if rank == 0:
        print(
            f"\nRunning {world_size} ranks with backend={backend}, device={device}\n",
            flush=True,
        )

    demonstrate_broadcast(rank, world_size, device)
    demonstrate_reduce(rank, world_size, device)
    demonstrate_all_reduce(rank, world_size, device)
    demonstrate_all_gather(rank, world_size, device)
    demonstrate_reduce_scatter(rank, world_size, device, backend)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
