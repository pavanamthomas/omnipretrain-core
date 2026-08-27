# Tests

`pytest` on CPU. No GPU, no wandb key.

If you add a spawn test, keep it behind `OMNI_SPAWN=1`. `init_process_group` inside the pytest process will poison later tests; see `notes/fsdp_gotchas.md`.
