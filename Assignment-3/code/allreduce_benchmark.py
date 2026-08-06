"""
Measure the cost of AllReduce for different tensor sizes.

GPU/NCCL example:
    torchrun --standalone --nproc-per-node=4 \
      code/allreduce_benchmark.py --backend nccl --sizes-mb 1 16 64 256

This is a learning benchmark, not a replacement for a production profiler.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
    )
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def choose_backend(requested: str) -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if requested == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires NVIDIA CUDA GPUs.")
    return requested


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    backend = choose_backend(args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)

    if rank == 0:
        print(
            f"backend={backend}, world_size={world_size}, "
            f"iterations={args.iterations}"
        )
        print("size_mb | median_ms | approx_per_rank_payload_mb | approx_GB/s")

    for size_mb in args.sizes_mb:
        number_of_float32_values = size_mb * 1024 * 1024 // 4
        tensor = torch.ones(number_of_float32_values, device=device)

        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        synchronize(device)
        dist.barrier()

        timings_ms: list[float] = []
        for _ in range(args.iterations):
            dist.barrier()
            synchronize(device)
            started = time.perf_counter()
            dist.all_reduce(tensor)
            synchronize(device)
            timings_ms.append((time.perf_counter() - started) * 1_000)

        median_ms = statistics.median(timings_ms)

        # Ring AllReduce's approximate per-rank sent volume:
        # 2 * (p - 1) / p * tensor size.
        approximate_payload_mb = (
            2 * (world_size - 1) / world_size * size_mb
        )
        approximate_gbps = (
            approximate_payload_mb / 1024 / (median_ms / 1_000)
        )

        if rank == 0:
            print(
                f"{size_mb:7d} | "
                f"{median_ms:9.3f} | "
                f"{approximate_payload_mb:26.2f} | "
                f"{approximate_gbps:11.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
