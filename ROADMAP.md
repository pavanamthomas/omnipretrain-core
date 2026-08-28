# Roadmap

scratch list so I stop arguing with myself in PR comments

## landed

- ingest + full jitter + dropping truncated jpeg/html
- n-gram rank + image junk penalty (blank/salt/8x8). distilgpt2 hook is there, too slow, off.
- script_family gate so CJK captions are not ranked as OCR because the ref is english
- ratio buckets, pad not crop. mixed-ratio collate raises now; used to silently pad twice
- BucketBatchSampler strides by rank so epoch 0 is the same mix on 2 gpus
- spawn: FSDP if cuda else DDP/gloo. act ckpt on CheckpointBlock
- bg ckpt thread + flush on join (lost the last step once, see notes/lost-ckpt.md)
- prometheus textfile, wandb if a key exists. jsonl rotates at 32MiB
- PGD + greedy token edits. default eps 4/255 (screenshots are already jpeg)
- ECE, last bin closed. T from a 1d grid on ECE, not NLL
- 4/8bit wrap, fake quant when bnb won't import

## if we actually get an 8x box

- nccl, drop cpu offload
- freeze a 124M LM for ranking instead of the char model
- contrastive head on TinyVLM. CE is only there so FSDP has a loss
- dump bucket hist + curriculum scores into the jsonl, plot later

## no

deepspeed — two launchers
megatron TP — model isn't that big
a crawler — we take a url list
jailbreak prompt packs — token mutation is for ece, not for hosting a red-team dump
