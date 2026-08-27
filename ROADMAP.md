# Roadmap

Written down so I stop re-litigating the same forks in chat.

## Done

- Async ingest with full-jitter backoff and the cheap corrupt filters.
- N-gram curriculum + step schedule, plus a cheap image junk penalty on the same score. DistilGPT-2 hook exists but is not the default.
- Aspect-ratio buckets with pad-not-crop. Sampler refuses mixed ratios in one batch.
- FSDP spawn on gloo + activation checkpointing on the block class.
- Background checkpoint thread. Flush-on-join after the missing last-step bug.
- Prometheus textfile + wandb best-effort.
- L_inf PGD and greedy token edits.
- ECE with a closed last bin; temperature scaling via LBFGS on log T.
- 4/8-bit linear wrap with a CPU fake-quant fallback.

## Next (when there is a real multi-node job)

- Swap gloo for NCCL and drop CPU offload. The wrap policy should survive that.
- Replace the n-gram scorer with a frozen 124M LM once we have a GPU for ranking.
- Image-text contrastive loss on the tiny VLM instead of teacher-forced CE. The CE path is a placeholder so FSDP has a graph.
- Persist bucket occupancy and curriculum scores into the jsonl so we can plot them.

## Not doing

- DeepSpeed. FSDP is enough on a single 8x node and I do not want two launchers.
- Megatron-style tensor parallel. Wrong size model.
- A general web crawler. This repo consumes a URL list, it does not discover one.
- Hosted-model jailbreak prompt catalogues. Token mutation is for calibration tests.
