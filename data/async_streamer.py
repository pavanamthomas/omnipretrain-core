"""Async fetch for the uncurated web mix.

This is not a crawler. We take a URL list (or a local jsonl of already-scraped
pages) and pull bytes with bounded concurrency. The first pass is cheap:
timeouts, truncated bodies, and obviously corrupt payloads get dropped here so
the curriculum ranker is not scoring garbage.

Backoff is exponential with full jitter. I originally used equal jitter and
still stampeded a CDN during a retry storm; the full-jitter version in
``RetryPolicy.sleep_s`` is the one that stuck.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import mimetypes
import random
import sys
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

LOG = logging.getLogger("omni.stream")

# Drop anything smaller than this; it's almost always an error page stub.
_MIN_HTML_BYTES = 256
_MIN_IMAGE_BYTES = 64
_DEFAULT_UA = "omnipretrain-core/1.0 (+research ingest; contact via repo issues)"


class StreamError(RuntimeError):
    """Fetch or decode failed after retries were exhausted."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_s: float = 0.4
    cap_s: float = 20.0
    retry_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)

    def sleep_s(self, attempt: int) -> float:
        # full jitter: U(0, min(cap, base * 2^attempt))
        exp = min(self.cap_s, self.base_s * (2**attempt))
        return random.random() * exp


@dataclass
class FetchConfig:
    timeout_s: float = 20.0
    connect_s: float = 8.0
    max_bytes: int = 8 * 1024 * 1024
    concurrency: int = 8
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    user_agent: str = _DEFAULT_UA
    allow_hosts: frozenset[str] | None = None
    follow_redirects: bool = True


@dataclass
class StreamRecord:
    url: str
    kind: str  # "html" | "image" | "skip"
    content_type: str
    sha256: str
    nbytes: int
    payload: bytes | None
    reason: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    def to_jsonable(self, include_payload: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if include_payload and self.payload is not None:
            row["payload_hex"] = self.payload[:64].hex()
        row.pop("payload", None)
        return row


@dataclass
class StreamStats:
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    deduped: int = 0
    bytes_in: int = 0

    def bump(self, rec: StreamRecord) -> None:
        self.bytes_in += rec.nbytes
        if rec.kind == "skip":
            if rec.reason == "duplicate":
                self.deduped += 1
            else:
                self.skipped += 1
        elif rec.kind in {"html", "image"}:
            self.ok += 1
        else:
            self.failed += 1


def _guess_kind(content_type: str, url: str) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("text/html") or ct in {"application/xhtml+xml", "text/plain"}:
        return "html"
    if ct.startswith("image/"):
        return "image"
    ext_ct, _ = mimetypes.guess_type(urlparse(url).path)
    if ext_ct:
        return _guess_kind(ext_ct, "")
    return "skip"


def _looks_corrupt_html(blob: bytes) -> str | None:
    if len(blob) < _MIN_HTML_BYTES:
        return "html_too_small"
    # truncated gzip or random binary almost never has a tag and is high NUL
    nul = blob.count(b"\x00")
    if nul > max(8, len(blob) // 50):
        return "binary_in_html"
    head = blob[:4096].lower()
    if b"<html" not in head and b"<!doctype" not in head and b"<body" not in head:
        # still accept long text dumps (common in the alt-text crawl)
        if len(blob) < 1024:
            return "not_html"
    return None


def _looks_corrupt_image(blob: bytes) -> str | None:
    if len(blob) < _MIN_IMAGE_BYTES:
        return "image_too_small"
    sigs = (
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xff",
        b"GIF87a",
        b"GIF89a",
        b"RIFF",
        b"BM",
    )
    if not any(blob.startswith(s) for s in sigs):
        return "bad_image_magic"
    # JPEG with no EOI marker is a classic truncated CDN write
    if blob.startswith(b"\xff\xd8\xff") and not blob.rstrip().endswith(b"\xff\xd9"):
        if len(blob) < 4 * 1024:
            return "truncated_jpeg"
    return None


class WebLogStreamer:
    def __init__(self, cfg: FetchConfig | None = None) -> None:
        self.cfg = cfg or FetchConfig()
        self.stats = StreamStats()
        self._seen: set[str] = set()

    def reset_stats(self) -> None:
        self.stats = StreamStats()
        self._seen.clear()

    async def drain(
        self,
        urls: Sequence[str],
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> list[StreamRecord]:
        out: list[StreamRecord] = []
        async for rec in self.iter_urls(urls, session=session):
            out.append(rec)
        return out

    async def iter_urls(
        self,
        urls: Iterable[str],
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> AsyncIterator[StreamRecord]:
        sem = asyncio.Semaphore(self.cfg.concurrency)
        own_session = session is None
        if own_session:
            timeout = aiohttp.ClientTimeout(
                total=self.cfg.timeout_s,
                connect=self.cfg.connect_s,
            )
            session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.cfg.user_agent},
                raise_for_status=False,
            )
        assert session is not None
        try:
            tasks = [asyncio.create_task(self._bounded(sem, session, u)) for u in urls]
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                self.stats.bump(rec)
                yield rec
        finally:
            if own_session:
                await session.close()

    async def _bounded(
        self,
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        url: str,
    ) -> StreamRecord:
        async with sem:
            return await self._fetch_one(session, url)

    def _host_allowed(self, url: str) -> bool:
        if self.cfg.allow_hosts is None:
            return True
        host = (urlparse(url).hostname or "").lower()
        return host in self.cfg.allow_hosts

    async def _fetch_one(self, session: aiohttp.ClientSession, url: str) -> StreamRecord:
        if not self._host_allowed(url):
            return StreamRecord(
                url=url,
                kind="skip",
                content_type="",
                sha256="",
                nbytes=0,
                payload=None,
                reason="host_blocked",
            )
        last_err: str | None = None
        t0 = time.perf_counter()
        for attempt in range(self.cfg.retry.max_attempts):
            try:
                rec = await self._attempt(session, url, attempt + 1, t0)
                return rec
            except (aiohttp.ClientError, asyncio.TimeoutError, StreamError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= self.cfg.retry.max_attempts:
                    break
                delay = self.cfg.retry.sleep_s(attempt)
                LOG.debug("retry %s in %.2fs (%s)", url, delay, last_err)
                await asyncio.sleep(delay)
        return StreamRecord(
            url=url,
            kind="failed",
            content_type="",
            sha256="",
            nbytes=0,
            payload=None,
            reason=last_err or "unknown",
            latency_ms=(time.perf_counter() - t0) * 1000,
            attempts=self.cfg.retry.max_attempts,
        )

    async def _attempt(
        self,
        session: aiohttp.ClientSession,
        url: str,
        attempt: int,
        t0: float,
    ) -> StreamRecord:
        async with session.get(url, allow_redirects=self.cfg.follow_redirects) as resp:
            if resp.status in self.cfg.retry.retry_statuses:
                raise StreamError(f"http {resp.status}")
            if resp.status >= 400:
                return StreamRecord(
                    url=str(resp.url),
                    kind="skip",
                    content_type=resp.headers.get("Content-Type", ""),
                    sha256="",
                    nbytes=0,
                    payload=None,
                    reason=f"http_{resp.status}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    attempts=attempt,
                )
            raw = await resp.content.read(self.cfg.max_bytes + 1)
            if len(raw) > self.cfg.max_bytes:
                return StreamRecord(
                    url=str(resp.url),
                    kind="skip",
                    content_type=resp.headers.get("Content-Type", ""),
                    sha256="",
                    nbytes=len(raw),
                    payload=None,
                    reason="too_large",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    attempts=attempt,
                )
            ct = resp.headers.get("Content-Type", "") or ""
            kind = _guess_kind(ct, str(resp.url))
            reason: str | None = None
            if kind == "html":
                reason = _looks_corrupt_html(raw)
            elif kind == "image":
                reason = _looks_corrupt_image(raw)
            else:
                reason = "unsupported_type"
            digest = hashlib.sha256(raw).hexdigest()
            if reason is None and digest in self._seen:
                reason = "duplicate"
            if reason is None:
                self._seen.add(digest)
                keep_kind = kind
            else:
                keep_kind = "skip"
            return StreamRecord(
                url=str(resp.url),
                kind=keep_kind,
                content_type=ct,
                sha256=digest if keep_kind != "skip" or reason == "duplicate" else "",
                nbytes=len(raw),
                payload=raw if keep_kind in {"html", "image"} else None,
                reason=reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
                attempts=attempt,
            )


def load_url_list(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _write_jsonl(path: Path, rows: Sequence[StreamRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec.to_jsonable(), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch a URL list into a jsonl log.")
    p.add_argument("--urls", type=Path, required=True, help="one URL per line")
    p.add_argument("--out", type=Path, default=Path("artifacts/raw.jsonl"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--allow-host", action="append", default=[], help="repeatable allowlist")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    urls = load_url_list(args.urls)
    if not urls:
        LOG.error("no urls in %s", args.urls)
        return 2
    cfg = FetchConfig(
        timeout_s=args.timeout,
        concurrency=args.concurrency,
        max_bytes=args.max_bytes,
        allow_hosts=frozenset(args.allow_host) if args.allow_host else None,
    )
    streamer = WebLogStreamer(cfg)
    rows = asyncio.run(streamer.drain(urls))
    _write_jsonl(args.out, rows)
    s = streamer.stats
    LOG.info(
        "wrote %s  ok=%d skip=%d fail=%d dedup=%d bytes=%d",
        args.out,
        s.ok,
        s.skipped,
        s.failed,
        s.deduped,
        s.bytes_in,
    )
    return 0 if s.failed == 0 or s.ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
