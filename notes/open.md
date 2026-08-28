# leftover

cleared this pass:

- n-gram gates on `script_family`. latin-fitted model no longer exiles CJK
  to the hard tail. bilingual alt-text still scores the ascii residue if
  there is enough of it.
- `BucketBatchSampler(num_replicas=, rank=)` shuffles once, then strides.
  drop_last truncates so ranks stay in lockstep; otherwise we pad by
  repeating from the front (same as DistributedSampler).
- telemetry jsonl rotates at 32MiB, keeps 3. rankN.jsonl.1 is the newest
  rotated file.
- PGD default eps is 4/255. 8/255 is still a flag if you want ImageNet.

still open:

- [ ] freeze a 124M LM for ranking instead of the char model (needs a box)
- [ ] dump bucket hist + curriculum scores into the jsonl, plot later
- [ ] contrastive head on TinyVLM. CE is only there so FSDP has a loss
