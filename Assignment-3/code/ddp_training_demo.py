"""
Minimal synchronous data-parallel training experiment.

GPU/NCCL:
    torchrun --standalone --nproc-per-node=4 code/ddp_training_demo.py --backend nccl

CPU learning mode:
    torchrun --standalone --nproc-per-node=4 code/ddp_training_demo.py --backend gloo

Observe that:
1. Every rank receives a different mini-batch.
2. Local predictions and losses differ.
3. DDP synchronizes gradients during backward().
4. After optimizer.step(), every rank still has identical parameters.
"""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def choose_backend(requested: str) -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if requested == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires NVIDIA CUDA GPUs.")
    return requested


def parameter_checksum(model: nn.Module) -> torch.Tensor:
    values = [parameter.detach().float().sum() for parameter in model.parameters()]
    return torch.stack(values).sum()


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

    # Same initial model on all ranks.
    torch.manual_seed(7)
    random.seed(7)
    model = TinyNetwork().to(device)

    if backend == "nccl":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_function = nn.MSELoss()

    for step in range(args.steps):
        # Different seed per rank -> different local mini-batch.
        generator = torch.Generator(device=device)
        generator.manual_seed(1_000 * step + rank)

        features = torch.randn(
            args.batch_size,
            8,
            generator=generator,
            device=device,
        )
        targets = features.sum(dim=1, keepdim=True) + 0.1 * rank

        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = loss_function(predictions, targets)

        # DDP's gradient synchronization is triggered as backward computes
        # gradients. The training code does not call AllReduce directly.
        loss.backward()
        optimizer.step()

        checksum = parameter_checksum(model.module)
        gathered_checksums = [
            torch.empty_like(checksum) for _ in range(world_size)
        ]
        dist.all_gather(gathered_checksums, checksum)

        if rank == 0:
            checksums = [round(value.item(), 6) for value in gathered_checksums]
            print(
                f"step={step:02d} "
                f"rank0_local_loss={loss.item():.6f} "
                f"parameter_checksums={checksums}",
                flush=True,
            )

        # Fail loudly if model replicas have diverged.
        stacked = torch.stack(gathered_checksums)
        if not torch.allclose(stacked, stacked[0], atol=1e-6, rtol=1e-6):
            raise RuntimeError("DDP replicas diverged.")

    if rank == 0:
        print(
            "\nSuccess: ranks used different data but retained identical parameters.\n"
            "The missing step was gradient synchronization during backward().",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
