# omnipretrain-core

Training-time bits for a vision-language mix. Not weights.

I got tired of rewriting the same four pieces (ingest, ratio batches, a local
FSDP/DDP spawn, then PGD + ECE) every time a new VLM run started, so they live
here. `TinyVLM` in `training/distributed.py` is a stub transformer with a vision
stem. It exists so wrap / activation checkpointing / the async saver have a
real graph. Swap it.

CPU is enough for tests. You need a GPU for actual FSDP and for bitsandbytes
NF4 that matches the CUDA kernels.

```
make test
make train    # 2 proc. GPU -> FSDP, otherwise DDP+gloo
```

More commands under `python -m <module> --help`. `make bench` rewrites the
table below (and the copy in PERFORMANCE.md).

## Layout

- `data/` — aiohttp fetch, n-gram curriculum, pad-to-fit buckets
- `training/` — spawn harness, background ckpt thread, prometheus textfiles
- `robustness/` — Linf PGD, token edits, ECE + temperature
- `optimization/` — bnb wrap or fake quant, latency table
- `notes/fsdp_gotchas.md` — read this before copying the spawn loop onto a node

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

bnb import often dies on CPU. `LinearQuantizer` prints `backend=fallback` and
keeps going. Those fake-quant weights will not load on a GPU job, don't try.

W&B: no key, no dashboard, training does not care. Prometheus is a textfile in
`artifacts/metrics/rankN.prom`.

## What broke

- latin n-gram was shoving CJK captions past OCR junk. `script_family` skips that ppl.
- torch 2.13 FSDP: `needs a non-CPU accelerator`. CPU path is DDP now.
- ECE last bin used to be half-open so conf=1.0 vanished. Closed it.
- Async saver returned before the last `torch.save`. `flush()` on join.
- Fitting T on NLL made ECE *worse* on the attacked set. Grid on ECE instead.
- First CI job pulled the CUDA torch wheel and OOM'd. cpu index is pinned.

Details in CHANGELOG.md / notes/.

## Performance

CPU toy decoder, not an A100. `python -m optimization.benchmarker` refreshes this.

<!-- omnipretrain:perf-table:begin -->
Toy decoder throughput (cpu)

_seq_len=32._

| batch | TTFT (ms) | tok/s | decode (ms) | VRAM (MiB) | device |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.2 | 4012.5 | 4.0 | 0.0 | cpu |
| 4 | 0.4 | 9847.7 | 6.5 | 0.0 | cpu |
| 16 | 0.6 | 23183.0 | 11.0 | 0.0 | cpu |
| 32 | 0.9 | 29331.7 | 17.5 | 0.0 | cpu |
| 64 | 1.3 | 36741.6 | 27.9 | 0.0 | cpu |
<!-- omnipretrain:perf-table:end -->
