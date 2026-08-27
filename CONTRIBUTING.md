pytest on cpu. no gpu, no wandb key.

spawn tests go through `run_local` (new processes). do not call
init_process_group inside the pytest process — the next test inherits a
dead group. see notes/fsdp_gotchas.md.

`make test` is the whole suite. the ddp spawn case is a couple seconds.
