# leftover

things that are real and not worth a ticket yet

- [ ] n-gram ranker treats all unicode as equally weird. CJK captions get
      shoved to the hard tail. fine for the english-heavy debug mix, wrong
      for the actual dump.
- [ ] BucketBatchSampler shuffles buckets then batches; epoch 0 on 2 gpus
      can disagree about which ratio comes first. doesn't matter with the
      fake loader. will matter with DistributedSampler.
- [ ] telemetry jsonl is append-only and never rotated. a 50k step run
      will be annoying to grep. I keep meaning to add a size cap.
- [ ] PGD default eps=8/255 is ImageNet folklore. screenshots in the mix
      are already jpeg-noisy; 4/255 might be the number we actually want.
      didn't rerun the sweep.
