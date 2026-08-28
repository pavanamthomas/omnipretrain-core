# FSDP notes from the laptop runs

I keep hitting the same things when I try to make this look like the cluster job.

## Backend

NCCL on a machine with no visible GPU leaves zombie processes. gloo is slow and that is the point of this file: we are testing wrap + checkpointing, not allreduce speed.

torch 2.13 FSDP raises `needs a non-CPU accelerator device` if you wrap on CPU. The spawn harness therefore DDP+gloo when `cuda.is_available()` is false. Do not read a CPU DDP run as an FSDP sharding result.

## Wrap policy

`ModuleWrapPolicy({CheckpointBlock})` has to wrap the block class, not TinyVLM.
If you wrap the whole model you still "run FSDP" and then none of the blocks
are actually sharded. If activation ckpt doesn't drop fwd memory, that's why.

## Memory barriers / collectives

FSDP `state_dict()` is an all-gather. If rank0 enters the async saver while rank1 is still in `backward()`, you deadlock on the next step's reduce-scatter. The rule we ended up with:

1. every rank finishes `opt.step()`
2. barrier
3. rank0 clones the already-gathered CPU dict and queues it
4. ranks resume the next batch together

Do not put the clone inside the thread. The thread must not participate in a collective. `AsyncCheckpointSaver.submit` is a queue.put of CPU tensors only.

Gradient accumulation has the same trap: `no_sync()` on FSDP delays the reduce-scatter until the last micro-batch. If you checkpoint on a micro-step that was under `no_sync()`, the shard is stale and the next all-gather waits forever. We only save on optimizer-step boundaries.

## Sharding hazards

- Mixed CPU offload + CUDA params: grads land on cpu, the next FSDP hook tries to view them as cuda. Pick one.
- `use_orig_params=True` plus activation checkpointing on the same block double-wraps the param handle. Memory looks fine until step 2.
- Uneven last batch with drop_last=False: one rank runs a smaller micro-batch, FSDP still expects matching all-gather sizes. TinyVLM's fake batch is fixed per rank to avoid that. A real dataloader should use `BucketBatchSampler(..., num_replicas=world, rank=rank)` (it pads or drop_lasts so both ranks step) or a `DistributedSampler` with `drop_last=True`.

## CPU offload

On in the CPU path. Off on a real GPU run. Mixing them in one script was how I ended up with params on cuda and grads on cpu.

## Spawn vs pytest

Do not `init_process_group` inside the pytest process. The next test inherits a dead group. `run_local` always spawn()s. The DDP coverage lives in `tests/test_ddp_spawn.py`.
