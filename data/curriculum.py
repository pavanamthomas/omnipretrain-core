"""Rank web text from easy to hard before it hits the VLM.

Perplexity comes from a tiny character n-gram LM that we fit on a "clean"
reference dump (Wikipedia-ish captions, alt-text that already survived a
human pass). I tried hooking distilgpt2 here and it made the ranking job
slower than the download. The n-gram is good enough to push boilerplate to
the front of the epoch and leave OCR-garbled PDF dumps for later.

The difficulty score is text log-ppl + caption noise + a cheap image
junk penalty (blank / salt / 8x8 block energy). Noise is a penalty, not
a hard drop. Hard drops belong in the streamer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

LOG = logging.getLogger("omni.curriculum")

_HTML_TAG = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WORD = re.compile(r"[A-Za-z0-9']+")
_REPEAT = re.compile(r"(.{8,})\1{3,}")

# CJK unified + ext-A + kana + hangul. not exhaustive, enough for the dump.
_CJK_RANGES = (
    (0x1100, 0x11FF),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def script_family(text: str) -> str:
    """latin | cjk | other.

    The English char-LM treats every CJK codepoint as unseen, so a fine
    中文 / 日本語 caption used to land in the hard tail next to OCR junk.
    Rankers that fitted on a latin dump should skip that ppl.
    """
    cjk = 0
    latin = 0
    other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if _is_cjk_char(ch):
            cjk += 1
        elif ch.isascii():
            latin += 1
        else:
            other += 1
    n = cjk + latin + other
    if n == 0:
        return "other"
    if cjk / n >= 0.4:
        return "cjk"
    if latin / n >= 0.5:
        return "latin"
    return "other"


def latin_residue(text: str) -> str:
    """Keep ascii letters so bilingual alt-text still has something to score."""
    kept: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalpha() or ch.isspace() or ch in ".,'-"):
            kept.append(ch)
        elif not ch.isascii() and ch.isspace():
            kept.append(" ")
    return " ".join("".join(kept).split())


class Scorer(Protocol):
    def log_perplexity(self, text: str) -> float: ...


@dataclass
class NoiseFlags:
    html_residue: float = 0.0
    control_chars: float = 0.0
    repetition: float = 0.0
    short: float = 0.0
    script_mix: float = 0.0

    def penalty(self) -> float:
        # weights from a one-off sweep on the 12k debug mix; not sacred
        return (
            2.4 * self.html_residue
            + 3.0 * self.control_chars
            + 1.6 * self.repetition
            + 0.8 * self.short
            + 0.5 * self.script_mix
        )


@dataclass
class ImageNoise:
    """Cheap visual junk detectors. Not a quality model.

    ``blank`` catches nearly-constant tiles (failed downloads that still
    have a valid PNG header). ``salt`` is impulse noise. ``blocky`` is the
    8x8 JPEG tell. ``low_entropy`` is posterised / 2-color garbage.
    """

    blank: float = 0.0
    salt: float = 0.0
    blocky: float = 0.0
    low_entropy: float = 0.0

    def penalty(self) -> float:
        return (
            1.8 * self.blank
            + 1.3 * self.salt
            + 1.0 * self.blocky
            + 0.8 * self.low_entropy
        )


@dataclass
class RankedSample:
    text: str
    log_ppl: float
    noise: NoiseFlags
    score: float
    source: str = ""
    image_noise: ImageNoise | None = None

    def to_jsonable(self) -> dict[str, Any]:
        row = asdict(self)
        return row


def inspect_noise(text: str) -> NoiseFlags:
    n = max(len(text), 1)
    tags = len(_HTML_TAG.findall(text))
    html_ratio = min(1.0, (tags * 8) / n)
    ctrl = min(1.0, len(_CTRL.findall(text)) / 8.0)
    rep = 1.0 if _REPEAT.search(text) else 0.0
    if len(text) < 24:
        short = 1.0
    elif len(text) < 80:
        short = 0.4
    else:
        short = 0.0
    # crude: lots of non-latin mixed into mostly-latin captions is usually OCR junk
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(ch.isascii() for ch in letters)
    mix = 0.0
    if letters:
        frac = latin / len(letters)
        if 0.15 < frac < 0.85 and len(letters) > 40:
            mix = 1.0 - abs(0.5 - frac) * 2
    return NoiseFlags(
        html_residue=html_ratio,
        control_chars=ctrl,
        repetition=rep,
        short=short,
        script_mix=mix,
    )


def inspect_image_noise(image: torch.Tensor | None) -> ImageNoise:
    """Offline visual difficulty. Operates on a single C,H,W tensor.

    Kept numpy-free so the curriculum job does not grow a second stack.
    Returns zeros when ``image`` is None so text-only dumps still rank.
    """
    if image is None:
        return ImageNoise()
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a tensor, got {type(image)!r}")
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected C,H,W got {tuple(image.shape)}")
    x = image.detach().float()
    gray = x.mean(dim=0) if x.shape[0] in (1, 3) else x[0]
    var = float(gray.var().clamp_min(0.0))
    blank = float(min(1.0, max(0.0, 1.0 - var / 0.02)))
    lo = float((gray < 0.02).float().mean())
    hi = float((gray > 0.98).float().mean())
    salt = float(min(1.0, (lo + hi) * 2.0)) if var > 1e-4 else 0.0
    h, w = int(gray.shape[0]), int(gray.shape[1])
    bh, bw = h - h % 8, w - w % 8
    blocky = 0.0
    if bh >= 8 and bw >= 8:
        g = gray[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8)
        within = float(g.var(dim=(1, 3)).mean())
        between = float(g.mean(dim=(1, 3)).var())
        if within > 1e-8:
            blocky = float(min(1.0, (between / (within + 1e-6)) / 8.0))
    gmin = float(gray.min())
    gmax = float(gray.max())
    hist = torch.histc(gray.flatten(), bins=16, min=gmin, max=gmax + 1e-6)
    p = hist / hist.sum().clamp_min(1.0)
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum()) if nz.numel() else 0.0
    low_entropy = float(max(0.0, 1.0 - ent / math.log(16.0)))
    return ImageNoise(blank=blank, salt=salt, blocky=blocky, low_entropy=low_entropy)


class NGramReference:
    """Interpolated char unigram + bigram. Cheap, deterministic, no weights file."""

    def __init__(self, order: int = 2, k: float = 0.4) -> None:
        if order not in (1, 2):
            raise ValueError("only unigram/bigram are implemented")
        self.order = order
        self.k = k
        self.uni: Counter[str] = Counter()
        self.bi: Counter[tuple[str, str]] = Counter()
        self._n = 0
        self._fitted = False
        self._ref_script = "latin"
        # stand-in ppl when the query is a different script than the ref.
        # overwritten after fit() from the ref itself.
        self._cross_script_ppl = 8.0

    def fit(self, texts: Iterable[str]) -> "NGramReference":
        held: list[str] = []
        for text in texts:
            padded = f"\n{text}\n"
            self.uni.update(padded)
            self._n += len(padded)
            self.bi.update(zip(padded, padded[1:]))
            if len(held) < 48:
                held.append(text)
        if self._n == 0:
            raise ValueError("empty reference corpus")
        self._fitted = True
        scripts = [script_family(t) for t in held if any(ch.isalpha() for ch in t)]
        if scripts:
            self._ref_script = Counter(scripts).most_common(1)[0][0]
        self._cross_script_ppl = sum(self._nll(t) if t else 20.0 for t in held) / len(held)
        return self

    def _uni_p(self, ch: str) -> float:
        v = len(self.uni) or 1
        return (self.uni[ch] + self.k) / (self._n + self.k * v)

    def _bi_p(self, prev: str, ch: str) -> float:
        # stupid interpolation; KN is overkill for ranking
        lam = 0.65
        denom = self.uni[prev] + self.k * (len(self.uni) or 1)
        cond = (self.bi[(prev, ch)] + self.k) / denom
        return lam * cond + (1.0 - lam) * self._uni_p(ch)

    def _nll(self, text: str) -> float:
        if not text:
            return 20.0
        padded = f"\n{text}\n"
        nll = 0.0
        prev = padded[0]
        toks = 0
        for ch in padded[1:]:
            p = self._bi_p(prev, ch) if self.order == 2 else self._uni_p(ch)
            nll -= math.log(max(p, 1e-12))
            prev = ch
            toks += 1
        return nll / max(toks, 1)

    def log_perplexity(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        if not text:
            return 20.0
        fam = script_family(text)
        if fam not in {"other", self._ref_script}:
            residue = latin_residue(text)
            if len(residue) >= 24:
                return self._nll(residue)
            return self._cross_script_ppl
        return self._nll(text)


class TransformerReference:
    """Optional HF hook. Off by default; the n-gram path is what CI runs."""

    def __init__(self, model_id: str = "distilgpt2", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

    def log_perplexity(self, text: str) -> float:
        import torch

        self._load()
        if not text.strip():
            return 20.0
        batch = self._tok(text, return_tensors="pt", truncation=True, max_length=256)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._model(**batch, labels=batch["input_ids"])
        return float(out.loss)


@dataclass
class CurriculumEngine:
    scorer: Scorer
    pace: str = "linear"  # linear | sqrt | step
    step_cuts: tuple[float, ...] = (0.33, 0.66)

    def rank(
        self,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        images: Sequence[torch.Tensor | None] | None = None,
    ) -> list[RankedSample]:
        if sources is not None and len(sources) != len(texts):
            raise ValueError("sources length must match texts")
        if images is not None and len(images) != len(texts):
            raise ValueError("images length must match texts")
        ranked: list[RankedSample] = []
        for i, text in enumerate(texts):
            noise = inspect_noise(text)
            img_n = inspect_image_noise(None if images is None else images[i])
            lp = self.scorer.log_perplexity(text)
            score = lp + noise.penalty() + img_n.penalty()
            ranked.append(
                RankedSample(
                    text=text,
                    log_ppl=lp,
                    noise=noise,
                    score=score,
                    source="" if sources is None else sources[i],
                    image_noise=img_n,
                )
            )
        ranked.sort(key=lambda s: s.score)
        return ranked

    def schedule(self, ranked: Sequence[RankedSample], epoch: int, max_epoch: int) -> list[RankedSample]:
        """Reveal harder slices as epochs go on.

        ``step`` is what we actually train with. linear/sqrt were useful while
        debugging whether the tail was just noise.
        """
        if max_epoch < 1:
            raise ValueError("max_epoch must be >= 1")
        epoch = min(max(epoch, 0), max_epoch)
        n = len(ranked)
        if n == 0:
            return []
        frac = (epoch + 1) / max_epoch
        if self.pace == "sqrt":
            frac = math.sqrt(frac)
        elif self.pace == "step":
            cuts = (0.0,) + self.step_cuts + (1.0,)
            # map epoch fraction onto the next cut
            idx = min(int(frac * (len(cuts) - 1)), len(cuts) - 1)
            frac = cuts[idx]
        elif self.pace != "linear":
            raise ValueError(f"unknown pace {self.pace!r}")
        k = max(1, int(math.ceil(frac * n)))
        return list(ranked[:k])


_DEFAULT_REF = [
    "A red bicycle leans against a brick wall at dusk.",
    "Two people sit on a bench facing a small lake.",
    "Close-up of a ceramic bowl filled with rice and scallions.",
    "The storefront has gold lettering and a striped awning.",
    "A child holds a paper kite on a windy hill.",
    "Interior of a train car, empty seats, afternoon light.",
    "A calico cat sleeps on a stack of newspapers.",
    "Mountain ridge above the treeline under a clear sky.",
]


def _load_texts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            rows.append({"text": obj.get("text") or obj.get("caption") or "", "source": obj.get("source", "")})
        else:
            rows.append({"text": line, "source": ""})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rank captions by n-gram perplexity + noise.")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/curriculum.jsonl"))
    p.add_argument("--ref", type=Path, default=None, help="optional extra reference corpus")
    p.add_argument("--pace", choices=("linear", "sqrt", "step"), default="step")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--max-epoch", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ref_texts = list(_DEFAULT_REF)
    if args.ref is not None:
        ref_texts.extend(r["text"] for r in _load_texts(args.ref) if r["text"])
    scorer = NGramReference().fit(ref_texts)
    rows = _load_texts(args.inp)
    engine = CurriculumEngine(scorer=scorer, pace=args.pace)
    ranked = engine.rank([r["text"] for r in rows], sources=[r["source"] for r in rows])
    sliced = engine.schedule(ranked, epoch=args.epoch, max_epoch=args.max_epoch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for sample in sliced:
            fh.write(json.dumps(sample.to_jsonable(), ensure_ascii=False) + "\n")
    LOG.info("ranked %d -> wrote %d to %s", len(ranked), len(sliced), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
