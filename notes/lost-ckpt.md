lost the last checkpoint of a 2h cpu run (2026-08)

rank0 logged "saved step 8000" and artifacts/ckpts/step-008000.pt was not there.

what happened: AsyncCheckpointSaver.submit() returned as soon as the job
hit the queue. the training process then joined the process group and
exited. daemon thread died with the job still sitting in the queue.

fix: flush() does queue.join() before close(). saver is not a daemon in
spirit even if the Thread is — we wait.

also: do not pass cuda tensors into the queue. clone to cpu on the
training thread. the writer fighting the allocator showed up as random
CUDA error: unspecified launch failure one box over, which was a gift.
