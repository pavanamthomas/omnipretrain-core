from __future__ import annotations

from aiohttp import web

from data.async_streamer import FetchConfig, RetryPolicy, WebLogStreamer


def _app() -> web.Application:
    app = web.Application()

    async def ok_html(_req: web.Request) -> web.Response:
        body = (
            "<!doctype html><html><head><title>x</title></head>"
            "<body>" + ("paragraph text " * 40) + "</body></html>"
        )
        return web.Response(text=body, content_type="text/html")

    async def tiny(_req: web.Request) -> web.Response:
        return web.Response(text="<html></html>", content_type="text/html")

    async def png(_req: web.Request) -> web.Response:
        blob = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
            b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return web.Response(body=blob, content_type="image/png")

    hits = {"n": 0}

    async def flaky(_req: web.Request) -> web.Response:
        hits["n"] += 1
        if hits["n"] < 3:
            return web.Response(status=503, text="busy")
        body = "<!doctype html><html><body>" + ("ok " * 80) + "</body></html>"
        return web.Response(text=body, content_type="text/html")

    async def boom(_req: web.Request) -> web.Response:
        return web.Response(status=500, text="nope")

    app.router.add_get("/ok", ok_html)
    app.router.add_get("/tiny", tiny)
    app.router.add_get("/img", png)
    app.router.add_get("/flaky", flaky)
    app.router.add_get("/boom", boom)
    return app


async def _serve() -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _drain(paths: list[str], cfg: FetchConfig | None = None):
    runner, base = await _serve()
    try:
        streamer = WebLogStreamer(cfg or FetchConfig(concurrency=4, timeout_s=5))
        urls = [base + p for p in paths]
        rows = await streamer.drain(urls)
        return rows, streamer
    finally:
        await runner.cleanup()


async def test_ok_and_image() -> None:
    rows, streamer = await _drain(["/ok", "/img"])
    kinds = sorted(r.kind for r in rows)
    assert kinds == ["html", "image"]
    assert streamer.stats.ok == 2
    assert all(r.payload for r in rows)


async def test_corrupt_html_dropped() -> None:
    rows, _ = await _drain(["/tiny"])
    assert rows[0].kind == "skip"
    assert rows[0].reason == "html_too_small"


async def test_backoff_recovers() -> None:
    cfg = FetchConfig(
        concurrency=1,
        timeout_s=5,
        retry=RetryPolicy(max_attempts=5, base_s=0.01, cap_s=0.05),
    )
    rows, streamer = await _drain(["/flaky"], cfg)
    assert rows[0].kind == "html"
    assert rows[0].attempts >= 3
    assert streamer.stats.ok == 1


async def test_exhausted_retries() -> None:
    cfg = FetchConfig(
        retry=RetryPolicy(max_attempts=2, base_s=0.01, cap_s=0.02, retry_statuses=(500,)),
        timeout_s=5,
    )
    rows, streamer = await _drain(["/boom"], cfg)
    assert rows[0].kind == "failed"
    assert streamer.stats.failed == 1


async def test_dedup() -> None:
    rows, streamer = await _drain(["/ok", "/ok"])
    kinds = [r.kind for r in rows]
    assert kinds.count("html") == 1
    assert kinds.count("skip") == 1
    assert streamer.stats.deduped == 1
