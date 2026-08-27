# Roadmap

scratch list so I stop arguing with myself in PR comments

## landed

- ingest + full jitter + dropping truncated jpeg/html
- n-gram rank + image junk penalty (blank/salt/8x8). distilgpt2 hook is there, too slow, off.
- ratio buckets, pad not crop. mixed-ratio collate raises now; used to silently pad twice
- spawn: FSDP if cuda else DDP/gloo. act ckpt on CheckpointBlock
- bg ckpt thread + flush on join (lost the last step once, see notes/lost-ckpt.md)
- prometheus textfile, wandb if a key exists
- PGD + greedy token edits
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
