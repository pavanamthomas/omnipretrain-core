.PHONY: test stream curriculum bucket train attack calibrate quant bench

# this image only ships python3
PYTHON ?= python3

test:
	WANDB_MODE=disabled $(PYTHON) -m pytest -q --tb=short

stream:
	$(PYTHON) -m data.async_streamer --urls data/fixtures/urls.txt --out artifacts/raw.jsonl

curriculum:
	$(PYTHON) -m data.curriculum --in data/fixtures/captions.jsonl --out artifacts/curriculum.jsonl --epoch 4 --max-epoch 10

bucket:
	$(PYTHON) -m data.bucketing --n 80 --out artifacts/buckets.json

train:
	$(PYTHON) -m training.distributed --world_size 2 --steps 8 --port 29731

attack:
	$(PYTHON) -m robustness.attack_engine --mode both --out artifacts/attacks.json

calibrate:
	$(PYTHON) -m robustness.calibration --out artifacts/calibration.json

quant:
	$(PYTHON) -m optimization.quantizer --mode nf4 --no-bnb --out artifacts/quant.json

bench:
	$(PYTHON) -m optimization.benchmarker --out PERFORMANCE.md --readme README.md --batches 1,4,16,32,64
