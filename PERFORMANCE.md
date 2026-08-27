# Performance

Toy decoder on CPU. Regenerated with:

```
python -m optimization.benchmarker --out PERFORMANCE.md --batches 1,4,16,32,64
```

Do not compare these to A100 vendor slides. The same table is embedded in README.md.

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
