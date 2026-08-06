#!/usr/bin/env bash
set -euo pipefail

PROCESSES="${1:-4}"
BACKEND="${2:-auto}"

echo "1. Running collective demonstrations"
torchrun --standalone --nproc-per-node="${PROCESSES}" \
  code/collectives_demo.py --backend "${BACKEND}"

echo
echo "2. Running DDP synchronization demonstration"
torchrun --standalone --nproc-per-node="${PROCESSES}" \
  code/ddp_training_demo.py --backend "${BACKEND}"

echo
echo "3. Running conceptual Ring AllReduce simulation"
python code/ring_allreduce_simulation.py
