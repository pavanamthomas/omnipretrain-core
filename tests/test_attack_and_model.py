from __future__ import annotations

import torch
import torch.nn.functional as F

from robustness.attack_engine import PGDAttack, TinyImageClassifier, TokenMutationAttack, char_noise
from training.distributed import TinyVLM, TrainConfig


def test_pgd_increases_loss() -> None:
    torch.manual_seed(2)
    model = TinyImageClassifier(n_classes=4)
    x = torch.rand(8, 3, 24, 24)
    y = torch.randint(0, 4, (8,))
    # untrained conv is noisy; still, PGD should climb the local loss
    res = PGDAttack(eps=0.1, step_size=0.03, steps=12, random_start=False).run(model, x, y)
    assert res.linf <= 0.1 + 1e-5
    assert res.loss_adv >= res.loss_clean - 1e-5


def test_char_noise_edits() -> None:
    out, n = char_noise("abcdefghij", p=1.0)
    assert n > 0
    assert out != "abcdefghij" or n > 0


def test_greedy_token_climb() -> None:
    vocab = ["red", "blue", "cat", "dog", "xxxx"]

    def score(s: str) -> float:
        return float(s.split().count("xxxx"))

    atk = TokenMutationAttack(vocab, max_edits=3, seed=0)
    r = atk.greedy("red cat dog", score)
    assert r.n_edits >= 1
    assert r.loss_adv >= r.loss_clean


def test_tiny_vlm_backward() -> None:
    cfg = TrainConfig(batch_size=2, seq_len=8, dim=32, depth=2, heads=4, image_size=32, patch=16)
    model = TinyVLM(cfg)
    img = torch.rand(2, 3, 32, 32)
    tok = torch.randint(0, cfg.vocab, (2, 8))
    logits = model(img, tok)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), tok.reshape(-1))
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)
    assert logits.shape == (2, 8, cfg.vocab)
