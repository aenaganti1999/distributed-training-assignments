# Submission Checklist

- [x] `README.md` answers all five required parts.
- [x] `architecture.png` shows where NCCL sits in the stack.
- [x] `communication_flow.png` shows when DDP communicates.
- [x] `ring_allreduce.png` shows the ring and its two phases.
- [x] `reflection.md` explains the learning in first-person language.

## Suggested Study Method

1. Read the executive summary.
2. Redraw Ring AllReduce without looking at the supplied image.
3. Explain each collective using four small arrays.
4. Recalculate the 8-GPU, 28-GB example.
5. Answer aloud: “Why is NCCL necessary?” without naming an API.
