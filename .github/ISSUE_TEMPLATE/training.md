---
name: Training divergence
about: Loss blew up, NaNs, or FSDP hang
labels: training
---

**Symptom**
- [ ] NaN loss
- [ ] grad norm inf
- [ ] hang on all_gather
- [ ] checkpoint never lands

**Config**
world_size / batch / activation ckpt / cpu offload:

**Last good step**
