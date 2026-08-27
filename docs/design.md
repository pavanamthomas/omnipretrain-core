# Design notes

## Why pad instead of crop

The web mix is not ImageNet. UI screenshots, slides, and infographics lose the thing we care about under a center crop. InternVL / Qwen-VL style ratio buckets are the compromise: a short list of canvases, scale to fit, pad. Distortion is banned in `pad_to_bucket`. The batch sampler will raise if two ratios land in the same collate — that used to fail silently as extra padding.

## Curriculum is ranking, not filtering

Hard drops stay in the streamer (truncated JPEG, host denylist, tiny stubs). The n-gram LM + caption noise + image junk penalty only *orders* what survived. I want the first epoch to be captions a human would write, and the OCR / blank-tile garbage to show up after the optimiser has a prior.

A frozen DistilGPT-2 scorer is wired (`TransformerReference`) and is slower than reading the jsonl. Leave it off unless you are debugging a ranking disagreement.

## Robustness is a loop with calibration

PGD on the vision tower makes ECE worse even when accuracy only moves a little, because the classifier gets louder. Temperature scaling is fitted on the attacked logits, not the clean ones. Fitting on clean and evaluating on attacked is how we used to report a flattering ECE.

Token mutation is HotFlip-lite: subsample positions, subsample vocab, climb a scalar score. It is not a prompt library.

## Compression fallback

bitsandbytes is the production path. The fake NF4/int8 modules exist because import fails on CPU CI and on a surprising number of "I pip installed it" laptops. Reports include `backend=` so we do not mix numbers.
