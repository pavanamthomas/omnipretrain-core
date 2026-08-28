# Changelog

## 1.0.1

- latin n-gram was shoving CJK captions past OCR junk. `script_family`
  now skips that ppl (noise flags still apply).
- BucketBatchSampler takes num_replicas/rank so two gpus agree on epoch 0.
- jsonl rotates at 32MiB. a 50k-step file was ungrepable.
- PGD default eps 4/255. 8/255 was washing already-jpeg screenshots.
- roadmap still said we fit T with LBFGS. we don't. grid on ECE.
- README was a pipeline poster. stripped it.
- `make train` on one GPU used to FSDP-wrap both ranks onto device 0.
  DDP unless `device_count() >= world_size`.
- loader test was asserting `shape[2] == shape[2]`. sampler batches are
  checked against the bucket index now.

## 1.0.0

tagged after the cpu suite + 2-proc ddp smoke passed.

## 0.4.4

- Curriculum difficulty is text log-ppl + caption noise + a cheap image junk penalty (blank / salt / 8x8).
- CLI accepts `--world_size` as well as `--world-size`. `make train` uses the underscore form.
- Bench matrix is batch in {1, 4, 16, 32, 64} and upserts both PERFORMANCE.md and README.md.

## 0.4.3

- FSDP wrap on CPU dies in torch 2.13 (`needs a non-CPU accelerator`). Spawn path now DDP+gloo when there is no GPU; FSDP stays on the CUDA branch.

## 0.4.2

- Benchmarker writes a delimited table so re-runs do not duplicate markdown.
- Quantizer slices padded NF4 groups back to `in_features`. Without that, Linear(16, *) blew up after wrap.

## 0.4.1

- Temperature scaler uses `log T` so T stays positive.
- Dropped NLL/LBFGS for a 1D grid on ECE. NLL-optimal T made ECE worse on the overconfident + label-noise set, which is the number we actually publish.

## 0.4.0

- Compression + latency profiler. CPU table is the one in PERFORMANCE.md.

## 0.3.2

- ECE last bin is closed on the right. Confidence 1.0 was falling off the end and looking like a mass of "uncalibrated" predictions.

## 0.3.1

- PGD + token mutation CLI.

## 0.3.0

- Calibration module. First ECE implementation had the bin bug above; kept the version anyway so the test that pins last-bin occupancy has a story.

## 0.2.1

- Async saver now `join()`s before the process exits. Rank0 was reporting "saved step 8000" and the file was not there.

## 0.2.0

- FSDP spawn harness. Backend is gloo on purpose.

## 0.1.0

- Ingest, curriculum, bucketing. First curriculum pass used a uniform noise penalty and shoved every HTML-ish caption to the tail; switched to the weighted flags in `inspect_noise`.
