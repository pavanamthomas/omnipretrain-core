"""PGD on images + discrete token mutations for the robustness suite.

Vision: standard L_inf PGD. Text: HotFlip-style greedy substitutions and
cheap character noise. This is for measuring whether the fused VLM stays
calibrated under perturbation, not for generating prompts against a hosted
model.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("omni.attack")


class AttackError(RuntimeError):
    pass


@dataclass
class PGDResult:
    x_adv: torch.Tensor
    loss_clean: float
    loss_adv: float
    linf: float
    steps: int
    success: bool


@dataclass
class TokenAttackResult:
    original: str
    mutated: str
    method: str
    loss_clean: float
    loss_adv: float
    n_edits: int


class PGDAttack:
    def __init__(
        self,
        *,
        eps: float = 8 / 255,
        step_size: float = 2 / 255,
        steps: int = 10,
        clamp: tuple[float, float] = (0.0, 1.0),
        random_start: bool = True,
    ) -> None:
        if eps < 0 or step_size < 0 or steps < 1:
            raise AttackError("eps/step_size must be >= 0 and steps >= 1")
        self.eps = eps
        self.step_size = step_size
        self.steps = steps
        self.clamp = clamp
        self.random_start = random_start

    @torch.enable_grad()
    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> PGDResult:
        if x.shape[0] != y.shape[0]:
            raise AttackError("x and y batch sizes differ")
        criterion = loss_fn or (lambda logits, tgt: F.cross_entropy(logits, tgt))
        was_training = model.training
        model.eval()
        with torch.no_grad():
            clean_logits = model(x)
            loss_clean = float(criterion(clean_logits, y).detach())
        x0 = x.detach()
        adv = x0.clone()
        if self.random_start and self.eps > 0:
            adv = adv + torch.empty_like(adv).uniform_(-self.eps, self.eps)
            adv = torch.max(torch.min(adv, x0 + self.eps), x0 - self.eps)
            adv = adv.clamp(*self.clamp)
        for _ in range(self.steps):
            adv = adv.detach().requires_grad_(True)
            logits = model(adv)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, adv)[0]
            if grad is None:
                raise AttackError("no gradient w.r.t. input; graph was disconnected")
            adv = adv.detach() + self.step_size * grad.sign()
            adv = torch.max(torch.min(adv, x0 + self.eps), x0 - self.eps)
            adv = adv.clamp(*self.clamp)
        with torch.no_grad():
            loss_adv = float(criterion(model(adv), y).detach())
            linf = float((adv - x0).abs().max().item())
        if was_training:
            model.train()
        return PGDResult(
            x_adv=adv.detach(),
            loss_clean=loss_clean,
            loss_adv=loss_adv,
            linf=linf,
            steps=self.steps,
            success=loss_adv > loss_clean,
        )


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def char_noise(text: str, *, p: float = 0.08, rng: random.Random | None = None) -> tuple[str, int]:
    rng = rng or random.Random()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    edits = 0
    for ch in text:
        if ch.isalpha() and rng.random() < p:
            edits += 1
            roll = rng.random()
            if roll < 0.34 and ch:
                continue  # delete
            if roll < 0.67:
                out.append(rng.choice(alphabet))  # replace
            else:
                out.append(ch)
                out.append(rng.choice(alphabet))  # insert
        else:
            out.append(ch)
    return "".join(out), edits


def swap_noise(text: str, *, n: int = 2, rng: random.Random | None = None) -> tuple[str, int]:
    rng = rng or random.Random()
    words = _tokenize_words(text)
    if len(words) < 2:
        return text, 0
    edits = 0
    words = list(words)
    for _ in range(min(n, len(words) - 1)):
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
        edits += 1
    return " ".join(words), edits


class TokenMutationAttack:
    """Greedy discrete substitutions using a white-box score function."""

    def __init__(self, vocab: Sequence[str], *, max_edits: int = 4, seed: int = 0) -> None:
        if not vocab:
            raise AttackError("empty vocab")
        self.vocab = list(vocab)
        self.max_edits = max_edits
        self.rng = random.Random(seed)

    def greedy(
        self,
        text: str,
        score_fn: Callable[[str], float],
    ) -> TokenAttackResult:
        words = _tokenize_words(text)
        if not words:
            return TokenAttackResult(text, text, "greedy", 0.0, 0.0, 0)
        base = score_fn(text)
        current = list(words)
        edits = 0
        for _ in range(self.max_edits):
            best_delta = 0.0
            best: tuple[int, str] | None = None
            # subsample positions; full search is O(L * |V|) and not worth it
            positions = list(range(len(current)))
            self.rng.shuffle(positions)
            for i in positions[: min(6, len(positions))]:
                for cand in self.rng.sample(self.vocab, k=min(8, len(self.vocab))):
                    if cand == current[i]:
                        continue
                    trial = list(current)
                    trial[i] = cand
                    s = score_fn(" ".join(trial))
                    delta = s - base
                    if delta > best_delta:
                        best_delta = delta
                        best = (i, cand)
            if best is None:
                break
            i, cand = best
            current[i] = cand
            edits += 1
            base = score_fn(" ".join(current))
        mutated = " ".join(current)
        return TokenAttackResult(
            original=text,
            mutated=mutated,
            method="greedy",
            loss_clean=score_fn(text),
            loss_adv=score_fn(mutated),
            n_edits=edits,
        )

    def random_walk(self, text: str, score_fn: Callable[[str], float]) -> TokenAttackResult:
        noisy, edits = char_noise(text, rng=self.rng)
        return TokenAttackResult(
            original=text,
            mutated=noisy,
            method="char",
            loss_clean=score_fn(text),
            loss_adv=score_fn(noisy),
            n_edits=edits,
        )


class TinyImageClassifier(nn.Module):
    """Stand-in victim for PGD tests / CLI smoke."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _cli_pgd(steps: int, eps: float) -> dict[str, Any]:
    torch.manual_seed(0)
    model = TinyImageClassifier()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    res = PGDAttack(eps=eps, steps=steps, step_size=eps / max(steps, 1)).run(model, x, y)
    return {
        "loss_clean": res.loss_clean,
        "loss_adv": res.loss_adv,
        "linf": res.linf,
        "success": res.success,
        "steps": res.steps,
    }


def _cli_token() -> dict[str, Any]:
    vocab = "cat dog car tree house person sky road red blue".split()
    # dummy score: longer rare words are "harder" so greedy has something to climb
    def score(s: str) -> float:
        return sum(len(w) + (0.5 if w not in vocab else 0.0) for w in s.split())

    atk = TokenMutationAttack(vocab, max_edits=3, seed=1)
    r = atk.greedy("a red cat on the road", score)
    return asdict(r)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run PGD or token-mutation smoke tests.")
    p.add_argument("--mode", choices=("pgd", "token", "both"), default="both")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--out", type=Path, default=Path("artifacts/attacks.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload: dict[str, Any] = {}
    if args.mode in {"pgd", "both"}:
        payload["pgd"] = _cli_pgd(args.steps, args.eps)
        LOG.info("pgd clean=%.4f adv=%.4f", payload["pgd"]["loss_clean"], payload["pgd"]["loss_adv"])
    if args.mode in {"token", "both"}:
        payload["token"] = _cli_token()
        LOG.info("token edits=%s", payload["token"]["n_edits"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
