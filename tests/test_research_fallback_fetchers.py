"""Contract tests for fallback fetchers: CDP browser, Scrape.do, Bright Data."""

from __future__ import annotations

import asyncio
import html
import json
import math
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from crawl4ai import CacheMode
from pytest_httpx import HTTPXMock

from mcps.config import (
    BrowserSettings,
    FetchSettings,
    RagSettings,
    RouterSettings,
    ScraperSettings,
    ServerConfig,
)
from mcps.research.tools import browser as browser_module
from mcps.research.tools.bright_data import BrightDataFetch
from mcps.research.tools.browser import SessionPool, create_browser_fetch
from mcps.research.tools.default import HttpFetch
from mcps.research.tools.fetch import build_fetch_tool, create_page_filter
from mcps.research.tools.filtering import ChunkedRelevanceFilter
from mcps.research.tools.github import GitHubBlobFetch
from mcps.research.tools.models import FetchResult, FetchStatus
from mcps.research.tools.scrape_do import ScrapeDoFetch
from tests.research_fakes import fake_crawler_strategy

TARGET = "https://blocked.example/article"
URL_A = "https://a.example"
URL_B = "https://b.example"
URL_C = "https://c.example"
BLANK = "goto:about:blank"
# A hung fake gives up after this long, so a missing production timeout fails a
# test instead of hanging the run.
_UNBOUNDED_WAIT = 2.0

MARKDOWN_PAGE = "## Quantum\n\n" + "quantum optimization improves routing " * 30


def outcome(result: FetchResult) -> tuple[FetchStatus, int | None]:
    return result.status, result.http_status


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Browser (crawl4ai over CDP)
# ---------------------------------------------------------------------------


NO_HOOK_CALL = object()


@dataclass
class FakePage:
    """Playwright page stub recording navigations into the crawler's trace."""

    trace: list[str]
    urls: list[str] = field(default_factory=list)
    goto_error: Exception | None = None
    goto_gate: asyncio.Event | None = None
    goto_entered: asyncio.Event = field(default_factory=asyncio.Event)

    async def goto(self, url: str, **_kwargs: Any) -> None:
        self.goto_entered.set()
        if self.goto_gate is not None:
            await self.goto_gate.wait()
        if self.goto_error is not None:
            raise self.goto_error
        self.urls.append(url)
        self.trace.append(f"goto:{url}")


@dataclass
class FakeCrawler:
    """Crawler stub over a connection that can be opened/closed.

    Mimics the crawl4ai pieces the fetcher touches: a session store whose pages
    appear on the first render of a ``session_id``, ``kill_session`` and the
    browser connection flag. ``trace`` orders renders and tab navigations.
    """

    result: Any = None
    error: Exception | None = None
    open: bool = True
    opens: int = 0
    closes: int = 0
    urls: list[str] = field(default_factory=list)
    configs: list[Any] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    connected: bool = True
    sessions: dict[str, Any] = field(default_factory=dict)
    killed: list[str] = field(default_factory=list)
    kill_error: Exception | None = None
    kill_gate: asyncio.Event | None = None
    close_error: Exception | None = None
    close_gate: asyncio.Event | None = None
    error_message: str | None = None
    arun_gate: asyncio.Event | None = None
    arun_entered: asyncio.Event = field(default_factory=asyncio.Event)
    url_gates: dict[str, asyncio.Event] = field(default_factory=dict)
    # Final navigation response handed to the ``after_goto`` hook; the hook is
    # not called while this stays ``NO_HOOK_CALL``.
    final_response: Any = NO_HOOK_CALL
    crawler_strategy: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        async def kill(session_id: str) -> None:
            if self.kill_gate is not None:
                await self.kill_gate.wait()
            if self.kill_error is not None:
                raise self.kill_error
            self.killed.append(session_id)
            self.sessions.pop(session_id, None)

        self.crawler_strategy = fake_crawler_strategy(
            self.sessions, lambda: self.connected, kill
        )

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        if not self.open:
            raise RuntimeError("crawler used while its connection is closed")
        self.urls.append(url)
        self.configs.append(config)
        self.trace.append(f"render:{url}")
        session_id = getattr(config, "session_id", None)
        if session_id and session_id not in self.sessions:
            self.sessions[session_id] = (None, FakePage(self.trace), 0.0)
        self.arun_entered.set()
        gate = self.url_gates.get(url, self.arun_gate)
        if gate is not None:
            await gate.wait()
        await self._run_after_goto_hook(url, config, session_id)
        if self.error is not None:
            raise self.error
        if self.error_message is not None:
            failed = crawl_result(success=False)
            failed.error_message = self.error_message
            return failed
        return self.result

    async def _run_after_goto_hook(
        self, url: str, config: Any, session_id: str | None
    ) -> None:
        hook = self.crawler_strategy.hooks.get("after_goto")
        if hook is None or self.final_response is NO_HOOK_CALL:
            return
        page = self.sessions[session_id][1] if session_id else None
        await hook(
            page,
            context=None,
            url=url,
            response=self.final_response,
            config=config,
        )


def navigation_response(content_type: str | None) -> SimpleNamespace:
    """Final Playwright response stub carrying only a ``content-type`` header."""
    headers = {} if content_type is None else {"content-type": content_type}
    return SimpleNamespace(headers=headers)


class _RaisingHeadersResponse:
    @property
    def headers(self) -> dict[str, str]:
        raise RuntimeError("response is gone")


def crawl_result(
    *,
    success: bool = True,
    status_code: int | None = 200,
    cleaned_html: str = "",
    html: str = "",
    response_headers: dict[str, str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        status_code=status_code,
        cleaned_html=cleaned_html,
        html=html,
        redirected_url=None,
        markdown=None,
        response_headers=response_headers,
        error_message="" if success else "boom",
    )


class GatedCrawler:
    """Crawler whose per-URL waits overlap until released, recording configs."""

    def __init__(self, gates: dict[str, asyncio.Event]):
        self.gates = gates
        self.entered = {url: asyncio.Event() for url in gates}
        self.configs: list[Any] = []
        self.session_ids: list[str | None] = []
        self.trace: list[str] = []
        self.sessions: dict[str, Any] = {}
        self.crawler_strategy = fake_crawler_strategy(self.sessions)
        self.open = True
        self.opens = 0
        self.closes = 0

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        config.url = url
        self.configs.append(config)
        self.session_ids.append(config.session_id)
        self.trace.append(f"render:{url}")
        if config.session_id and config.session_id not in self.sessions:
            self.sessions[config.session_id] = (None, FakePage(self.trace), 0.0)
        self.entered[url].set()
        async with asyncio.timeout(5):
            await self.gates[url].wait()
        return crawl_result(cleaned_html=f"<h1>{config.url}</h1>")


@dataclass
class _ConstructFailure:
    """Script item making ``FakeAsyncWebCrawler`` raise while constructing."""

    error: Exception


@dataclass
class _GatedStart:
    """Script item whose ``__aenter__`` signals ``entered`` and waits for ``gate``."""

    crawler: FakeCrawler
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None


class FakeAsyncWebCrawler:
    """Boundary fake for ``crawl4ai.AsyncWebCrawler``.

    Each construction consumes the next item of :attr:`script`: ``None`` uses a
    fresh successful :class:`FakeCrawler`; a :class:`FakeCrawler` is used as-is;
    a :class:`_GatedStart` is used as-is but pauses in ``__aenter__``; a
    :class:`_ConstructFailure` raises while constructing; any exception is
    raised from ``__aenter__``.
    """

    script: ClassVar[list[Any]] = []
    constructed: ClassVar[list[FakeAsyncWebCrawler]] = []

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.crawler = FakeCrawler()
        item = type(self).script.pop(0) if type(self).script else None
        if isinstance(item, _ConstructFailure):
            raise item.error
        self.start_gate = item if isinstance(item, _GatedStart) else None
        if isinstance(item, _GatedStart):
            self.crawler = item.crawler
        elif isinstance(item, FakeCrawler | GatedCrawler):
            self.crawler = item
        self.enter_error = item if isinstance(item, BaseException) else None
        type(self).constructed.append(self)

    @property
    def crawler_strategy(self) -> Any:
        return self.crawler.crawler_strategy

    async def __aenter__(self) -> FakeAsyncWebCrawler:
        if self.enter_error is not None:
            raise self.enter_error
        if self.start_gate is not None:
            self.start_gate.entered.set()
            await self.start_gate.gate.wait()
            if self.start_gate.error is not None:
                raise self.start_gate.error
        self.crawler.open = True
        self.crawler.opens += 1
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.crawler.open = False
        self.crawler.closes += 1
        close_gate = getattr(self.crawler, "close_gate", None)
        if close_gate is not None:
            try:
                async with asyncio.timeout(_UNBOUNDED_WAIT):
                    await close_gate.wait()
            except TimeoutError:
                raise AssertionError("crawler close was not time-bounded") from None
        close_error = getattr(self.crawler, "close_error", None)
        if close_error is not None:
            raise close_error

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        return await self.crawler.arun(url, config=config, **_kwargs)


def _script_crawlers(monkeypatch, *items: Any) -> None:
    monkeypatch.setattr(browser_module, "AsyncWebCrawler", FakeAsyncWebCrawler)
    FakeAsyncWebCrawler.script = list(items)
    FakeAsyncWebCrawler.constructed = []


def _browser_config(**overrides: Any) -> ServerConfig:
    overrides.setdefault("reconnect_delay_seconds", 0)
    return ServerConfig(browser=BrowserSettings(cdp_url="ws://cdp", **overrides))


@asynccontextmanager
async def _open_browser(monkeypatch, *script: Any, **overrides: Any):
    """Open the production browser fetcher over scripted fake crawlers."""
    _script_crawlers(monkeypatch, *script)
    async with create_browser_fetch(_browser_config(**overrides)) as fetch:
        assert fetch is not None
        yield fetch


def _only_page(crawler: FakeCrawler) -> FakePage:
    """The page of the crawler's single session (created on first render)."""
    assert crawler.sessions, "fetch rendered without a session_id"
    ((_, page, _),) = crawler.sessions.values()
    return page


OK_PAGE = "<h1>X</h1>"


def _ok_crawler(**fields: Any) -> FakeCrawler:
    return FakeCrawler(result=crawl_result(cleaned_html=OK_PAGE), **fields)


async def test_browser_fetch_returns_cleaned_html(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>Rendered</h1>"))

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Rendered</h1>")
    assert crawler.urls == [TARGET]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            crawl_result(status_code=403, cleaned_html="denied"),
            (FetchStatus.HTTP_ERROR, 403),
        ),
        (
            crawl_result(status_code=404, cleaned_html="gone"),
            (FetchStatus.HTTP_ERROR, 404),
        ),
        (crawl_result(cleaned_html="   "), (FetchStatus.EMPTY, None)),
        (
            crawl_result(success=False, status_code=None),
            (FetchStatus.UNAVAILABLE, None),
        ),
    ],
)
async def test_browser_fetch_maps_crawl_outcomes(
    monkeypatch, result: Any, expected: tuple[FetchStatus, int | None]
):
    async with _open_browser(monkeypatch, FakeCrawler(result=result)) as fetch:
        assert outcome(await fetch(TARGET)) == expected


async def test_browser_fetch_crawl_failure_is_unavailable(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(success=False))

    async with _open_browser(monkeypatch, crawler) as fetch:
        assert outcome(await fetch(TARGET)) == (FetchStatus.UNAVAILABLE, None)


@pytest.mark.parametrize("cleaned_html", ["", "   "])
async def test_browser_fetch_blank_cleaned_html_is_empty(
    monkeypatch, cleaned_html: str
):
    crawler = FakeCrawler(result=crawl_result(cleaned_html=cleaned_html))

    async with _open_browser(monkeypatch, crawler) as fetch:
        assert outcome(await fetch(TARGET)) == (FetchStatus.EMPTY, None)


SOURCE_TEXT = '{\n  "quantum": "routing",\n  "literal": "<p>not markup</p>"\n}\n'


def _pre_wrapper(text: str) -> str:
    return f"<html><body><pre>{html.escape(text)}</pre></body></html>"


async def test_browser_declared_text_unwraps_original_pre_without_markdown_conversion(
    monkeypatch,
):
    wrapper = _pre_wrapper(SOURCE_TEXT)
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=wrapper,
            html=wrapper,
            response_headers={"Content-Type": "application/json; charset=utf-8"},
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch("https://source.example/content")

    assert result.status is FetchStatus.OK
    assert result.mime == "text/plain"
    assert result.content == SOURCE_TEXT


async def test_browser_declared_text_restores_whitespace_from_original_html(
    monkeypatch,
):
    exact = _pre_wrapper(SOURCE_TEXT)
    collapsed = _pre_wrapper(SOURCE_TEXT.replace("\n", " ").replace("  ", " "))
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=collapsed,
            html=exact,
            response_headers={"content-type": "text/plain"},
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch("https://source.example/content")

    assert result.content == SOURCE_TEXT


async def test_browser_declared_markdown_preserves_native_source(monkeypatch):
    wrapper = _pre_wrapper(MARKDOWN_PAGE)
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=wrapper,
            html=wrapper,
            response_headers={"content-type": "text/markdown"},
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch("https://source.example/content")

    assert result.status is FetchStatus.OK
    assert result.mime == "text/markdown"
    assert result.content == MARKDOWN_PAGE


async def test_browser_binary_type_is_unsupported_even_with_viewer_html(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html="<h1>viewer</h1>",
            html="<html><body><h1>viewer</h1></body></html>",
            response_headers={"content-type": "image/png"},
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert outcome(result) == (FetchStatus.UNSUPPORTED, None)
    assert (result.content, result.mime) == ("", "image/png")


async def test_browser_malformed_declared_text_is_unsupported(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html="<html><body><p>no pre</p></body></html>",
            html="<html><body><p>no pre</p></body></html>",
            response_headers={"content-type": "text/plain"},
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        assert (await fetch(TARGET)).status is FetchStatus.UNSUPPORTED


async def test_github_plain_code_is_not_markdown(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/src/a.py",
        text="    # literal\n    print(1)",
        headers={"content-type": "text/plain"},
    )
    url = "https://github.com/o/r/blob/main/src/a.py"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert result.mime == "text/plain"
    assert result.content == "# literal\n    print(1)"
    assert result.url == url


# ---------------------------------------------------------------------------
# Browser crawler connection lifetime
# ---------------------------------------------------------------------------


async def test_browser_fetch_repeated_requests_reuse_open_connection(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>X</h1>"))
    _script_crawlers(monkeypatch, crawler)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is not None
        first = await browser("https://a.example")
        second = await browser("https://b.example")
        assert crawler.opens == 1 and crawler.closes == 0 and crawler.open

    assert first.ok and second.ok
    assert first.content == second.content == "<h1>X</h1>"
    assert crawler.urls == ["https://a.example", "https://b.example"]
    assert crawler.closes == 1 and crawler.open is False


async def test_browser_fetch_overlapping_requests_share_connection(monkeypatch):
    gates = {URL_A: asyncio.Event(), URL_B: asyncio.Event()}
    crawler = GatedCrawler(gates)

    async with _open_browser(monkeypatch, crawler, pool_size=2) as fetch:
        tasks = [asyncio.ensure_future(fetch(url)) for url in gates]
        try:
            async with asyncio.timeout(5):
                await asyncio.gather(*(crawler.entered[url].wait() for url in gates))
            assert len(crawler.configs) == 2
            assert crawler.configs[0] is not crawler.configs[1]
        finally:
            for gate in gates.values():
                gate.set()
        results = await asyncio.gather(*tasks)

    assert [result.content for result in results] == [
        f"<h1>{URL_A}</h1>",
        f"<h1>{URL_B}</h1>",
    ]


async def test_browser_fetch_failure_keeps_connection_for_next_request(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(success=False))
    _script_crawlers(monkeypatch, crawler)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is not None
        failure = await browser("https://a.example")
        crawler.result = crawl_result(cleaned_html="<h1>Recovered</h1>")
        recovered = await browser("https://b.example")
        assert outcome(failure) == (FetchStatus.UNAVAILABLE, None)
        assert recovered.content == "<h1>Recovered</h1>"
        assert crawler.opens == 1 and crawler.closes == 0

    assert crawler.closes == 1


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(ConnectionError("cdp down"), id="enter"),
        pytest.param(_ConstructFailure(ConnectionError("bad config")), id="construct"),
    ],
)
async def test_browser_fetch_startup_failure_yields_none(monkeypatch, item: Any):
    _script_crawlers(monkeypatch, item)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is None


async def test_browser_fetch_without_endpoint_opens_no_connection(monkeypatch):
    _script_crawlers(monkeypatch)

    config = ServerConfig(browser=BrowserSettings(cdp_url=""))
    async with create_browser_fetch(config) as browser:
        assert browser is None

    assert FakeAsyncWebCrawler.constructed == []


async def test_browser_fetch_exceptional_exit_closes_connection(monkeypatch):
    crawler = FakeCrawler()
    _script_crawlers(monkeypatch, crawler)

    with pytest.raises(ValueError, match="body boom"):
        async with create_browser_fetch(_browser_config()):
            raise ValueError("body boom")

    assert crawler.open is False and crawler.closes == 1


async def test_browser_fetch_cancellation_closes_connection(monkeypatch):
    crawler = FakeCrawler()
    _script_crawlers(monkeypatch, crawler)
    entered = asyncio.Event()

    async def consume() -> None:
        async with create_browser_fetch(_browser_config()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert crawler.open is False and crawler.closes == 1


async def test_body_exception_wins_over_close_error(monkeypatch):
    crawler = FakeCrawler(close_error=RuntimeError("close failed"))
    _script_crawlers(monkeypatch, crawler)

    with pytest.raises(ValueError, match="body failed"):
        async with create_browser_fetch(_browser_config()):
            raise ValueError("body failed")


# ---------------------------------------------------------------------------
# Browser session pool: tab reuse, tab reset, expiry, crawl4ai wiring
# ---------------------------------------------------------------------------


async def test_session_pool_blocks_until_release():
    pool = SessionPool(1)
    name = await pool.acquire()

    waiter = asyncio.create_task(pool.acquire())
    for _ in range(3):
        await asyncio.sleep(0)
    assert not waiter.done()

    pool.release(name)
    async with asyncio.timeout(5):
        assert await waiter == name


async def test_session_pool_hands_out_distinct_names():
    size = 3
    pool = SessionPool(size)

    names = [await pool.acquire() for _ in range(size)]

    assert len(set(names)) == size


async def test_browser_fetch_reuses_session_tab_across_fetches(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        await fetch(URL_A)
        await fetch(URL_B)

    first, second = (config.session_id for config in crawler.configs)
    assert first and first == second
    assert len(crawler.sessions) == 1
    assert crawler.opens == 1


async def test_browser_fetch_concurrent_fetches_use_distinct_sessions(monkeypatch):
    gates = {URL_A: asyncio.Event(), URL_B: asyncio.Event()}
    crawler = GatedCrawler(gates)

    async with _open_browser(
        monkeypatch, crawler, pool_size=len(gates)
    ) as fetch:
        tasks = [asyncio.ensure_future(fetch(url)) for url in gates]
        try:
            async with asyncio.timeout(5):
                await asyncio.gather(*(crawler.entered[url].wait() for url in gates))
            assert len(set(crawler.session_ids)) == len(gates)
        finally:
            for gate in gates.values():
                gate.set()
            await asyncio.gather(*tasks)


async def test_browser_fetch_caps_concurrent_renders_at_pool_size(monkeypatch):
    size = 2
    urls = [f"https://{name}.example" for name in "abc"]
    assert len(urls) > size
    gates = {url: asyncio.Event() for url in urls}
    crawler = GatedCrawler(gates)

    def entered() -> list[str]:
        return [url for url in urls if crawler.entered[url].is_set()]

    async with _open_browser(monkeypatch, crawler, pool_size=size) as fetch:
        tasks = [asyncio.ensure_future(fetch(url)) for url in urls]
        try:
            async with asyncio.timeout(5):
                while len(entered()) < size:
                    await asyncio.sleep(0)
            for _ in range(5):
                await asyncio.sleep(0)
            assert len(entered()) == size

            gates[entered()[0]].set()
            async with asyncio.timeout(5):
                await asyncio.gather(*(crawler.entered[url].wait() for url in urls))
        finally:
            for gate in gates.values():
                gate.set()
        results = await asyncio.gather(*tasks)

    assert all(result.ok for result in results)


async def test_browser_fetch_resets_tab_after_each_fetch(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        await fetch(URL_A)
        await fetch(URL_B)

    assert crawler.trace == [f"render:{URL_A}", BLANK, f"render:{URL_B}", BLANK]
    assert len(crawler.sessions) == 1
    assert crawler.killed == []


async def test_browser_fetch_resets_tab_after_ordinary_failure(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(status_code=404, cleaned_html="gone"))

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        result = await fetch(URL_A)

    assert result.status is FetchStatus.HTTP_ERROR
    assert crawler.trace[-1] == BLANK


async def test_browser_fetch_cancelled_during_render_resets_before_next_render(
    monkeypatch,
):
    crawler = _ok_crawler(url_gates={URL_A: asyncio.Event()})

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        task = asyncio.create_task(fetch(URL_A))
        async with asyncio.timeout(5):
            await crawler.arun_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(5):
            second = await fetch(URL_B)

    assert second.ok
    assert crawler.trace == [f"render:{URL_A}", BLANK, f"render:{URL_B}", BLANK]


async def test_browser_fetch_cancelled_during_reset_finishes_reset_before_next_render(
    monkeypatch,
):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        await fetch(URL_A)
        page = _only_page(crawler)
        page.goto_gate = asyncio.Event()
        page.goto_entered.clear()
        try:
            task = asyncio.create_task(fetch(URL_B))
            async with asyncio.timeout(5):
                await page.goto_entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            third = asyncio.create_task(fetch(URL_C))
        finally:
            page.goto_gate.set()
        async with asyncio.timeout(5):
            await third

    assert crawler.trace == [
        f"render:{URL_A}",
        BLANK,
        f"render:{URL_B}",
        BLANK,
        f"render:{URL_C}",
        BLANK,
    ]


async def test_browser_fetch_kills_session_when_reset_fails(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        await fetch(URL_A)
        first_page = _only_page(crawler)
        first_page.goto_error = RuntimeError("reset failed")
        second = await fetch(URL_B)
        third = await fetch(URL_C)

    assert second.ok and third.ok
    assert crawler.killed == [crawler.configs[1].session_id]
    assert _only_page(crawler) is not first_page


async def test_browser_fetch_kills_session_when_reset_times_out(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(
        monkeypatch,
        crawler,
        pool_size=1,
        cleanup_timeout_seconds=0.05,
    ) as fetch:
        await fetch(URL_A)
        _only_page(crawler).goto_gate = asyncio.Event()
        async with asyncio.timeout(5):
            second = await fetch(URL_B)

    assert second.ok
    assert crawler.configs[1].session_id in crawler.killed


@pytest.mark.parametrize("kill_fault", ["raises", "hangs"])
async def test_browser_fetch_reconnects_when_session_kill_fails(
    monkeypatch, kill_fault: str
):
    first, second = _ok_crawler(), _ok_crawler()

    async with _open_browser(
        monkeypatch,
        first,
        second,
        pool_size=1,
        cleanup_timeout_seconds=0.05,
    ) as fetch:
        await fetch(URL_A)
        _only_page(first).goto_error = RuntimeError("reset failed")
        if kill_fault == "raises":
            first.kill_error = RuntimeError("kill failed")
        else:
            first.kill_gate = asyncio.Event()
        async with asyncio.timeout(5):
            after_failed_cleanup = await fetch(URL_B)
            on_replacement = await fetch(URL_C)

    assert after_failed_cleanup.ok and on_replacement.ok
    assert second.urls == [URL_C]
    assert first.closes == 1


async def test_browser_fetch_disables_session_expiry(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler):
        manager = crawler.crawler_strategy.browser_manager
        assert manager.session_ttl == math.inf


def _drop_strategy(crawler: FakeCrawler) -> None:
    crawler.crawler_strategy = SimpleNamespace()


def _drop_session_ttl(crawler: FakeCrawler) -> None:
    del crawler.crawler_strategy.browser_manager.session_ttl


def _drop_sessions(crawler: FakeCrawler) -> None:
    del crawler.crawler_strategy.browser_manager.sessions


def _drop_is_connected(crawler: FakeCrawler) -> None:
    crawler.crawler_strategy.browser_manager.browser = SimpleNamespace()


def _drop_kill_session(crawler: FakeCrawler) -> None:
    del crawler.crawler_strategy.kill_session


def _drop_set_hook(crawler: FakeCrawler) -> None:
    del crawler.crawler_strategy.set_hook


@pytest.mark.parametrize(
    "break_internals",
    [
        pytest.param(_drop_strategy, id="no-browser-manager"),
        pytest.param(_drop_session_ttl, id="no-session-expiry"),
        pytest.param(_drop_sessions, id="no-session-store"),
        pytest.param(_drop_is_connected, id="no-connection-check"),
        pytest.param(_drop_kill_session, id="no-kill-session"),
        pytest.param(_drop_set_hook, id="no-hook-api"),
    ],
)
async def test_create_browser_fetch_yields_none_without_crawl4ai_internals(
    monkeypatch, break_internals: Callable[[FakeCrawler], None]
):
    crawler = FakeCrawler()
    break_internals(crawler)
    _script_crawlers(monkeypatch, crawler)

    async with create_browser_fetch(_browser_config()) as fetch:
        assert fetch is None

    assert crawler.closes == 1


async def test_browser_fetch_uses_session_and_bypasses_cache(monkeypatch):
    crawler = _ok_crawler()

    async with _open_browser(monkeypatch, crawler) as fetch:
        await fetch(TARGET)

    (config,) = crawler.configs
    assert config.session_id
    assert config.cache_mode is CacheMode.BYPASS
    assert config.wait_until == "domcontentloaded"


def test_new_crawler_shares_cdp_connection_without_cleanup(monkeypatch):
    captured: dict[str, Any] = {}

    class CapturingCrawler:
        def __init__(self, config: Any = None) -> None:
            captured["config"] = config

    monkeypatch.setattr(browser_module, "AsyncWebCrawler", CapturingCrawler)

    browser_module._new_crawler("ws://cdp")

    config = captured["config"]
    assert config.cache_cdp_connection is True
    assert config.create_isolated_context is False
    assert config.cdp_cleanup_on_close is False


async def test_exit_waits_for_pending_reset(monkeypatch):
    crawler = _ok_crawler()
    _script_crawlers(monkeypatch, crawler)
    context = create_browser_fetch(_browser_config(pool_size=1))
    fetch = await context.__aenter__()
    assert fetch is not None
    await fetch(URL_A)
    page = _only_page(crawler)
    page.goto_gate = asyncio.Event()
    page.goto_entered.clear()
    try:
        task = asyncio.create_task(fetch(URL_B))
        async with asyncio.timeout(5):
            await page.goto_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        leaving = asyncio.create_task(context.__aexit__(None, None, None))
        for _ in range(5):
            await asyncio.sleep(0)
        closes_while_resetting = crawler.closes
    finally:
        page.goto_gate.set()
    async with asyncio.timeout(5):
        await leaving

    assert closes_while_resetting == 0
    assert crawler.trace[-1] == BLANK
    assert crawler.closes == 1


# ---------------------------------------------------------------------------
# Browser connection loss: detection, shared recovery, retry
# ---------------------------------------------------------------------------

LOST = "Target page, context or browser has been closed"
DOWN = "endpoint down"


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0)


def _lost_crawler(**fields: Any) -> FakeCrawler:
    return FakeCrawler(error_message=LOST, **fields)


def _constructed_crawlers() -> list[Any]:
    return [created.crawler for created in FakeAsyncWebCrawler.constructed]


_CONNECTION_LOSSES: dict[str, Callable[[], FakeCrawler]] = {
    "loss-marker": lambda: FakeCrawler(error_message=f"Error: {LOST}"),
    "escaped-exception": lambda: FakeCrawler(error=RuntimeError("anything")),
    "browser-disconnected": lambda: FakeCrawler(
        result=crawl_result(success=False), connected=False
    ),
    "marker-in-lower-case": lambda: FakeCrawler(error_message="no page for session"),
}


@pytest.mark.parametrize(
    "make_lost", _CONNECTION_LOSSES.values(), ids=list(_CONNECTION_LOSSES)
)
async def test_browser_fetch_reconnects_and_retries_after_connection_loss(
    monkeypatch, make_lost: Callable[[], FakeCrawler]
):
    lost, healthy = make_lost(), _ok_crawler()

    async with _open_browser(
        monkeypatch,
        lost,
        healthy,
        fetch_retries=1,
        reconnect_attempts=1,
    ) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert result.content == OK_PAGE
    assert lost.closes == 1
    assert healthy.urls == [TARGET]


@pytest.mark.parametrize(
    "make_final",
    [
        pytest.param(
            lambda: FakeCrawler(
                result=crawl_result(status_code=404, cleaned_html="gone"),
                connected=False,
            ),
            id="http-error-even-when-disconnected",
        ),
        pytest.param(
            lambda: FakeCrawler(result=crawl_result(success=False)),
            id="ordinary-crawl-failure",
        ),
    ],
)
async def test_browser_fetch_does_not_retry_final_results(
    monkeypatch, make_final: Callable[[], FakeCrawler]
):
    crawler = make_final()

    async with _open_browser(
        monkeypatch, crawler, _ok_crawler(), fetch_retries=1
    ) as fetch:
        await fetch(TARGET)

    assert len(crawler.urls) == 1
    assert len(FakeAsyncWebCrawler.constructed) == 1


async def test_browser_fetch_stops_after_configured_retries(monkeypatch):
    retries = 2
    lost = [_lost_crawler() for _ in range(retries + 1)]
    spare = _ok_crawler()

    async with _open_browser(
        monkeypatch,
        *lost,
        spare,
        fetch_retries=retries,
        reconnect_attempts=1,
    ) as fetch:
        result = await fetch(TARGET)

    assert result.status is FetchStatus.UNAVAILABLE
    assert sum(len(crawler.urls) for crawler in lost) == retries + 1
    assert spare in _constructed_crawlers()
    assert spare.urls == []


@pytest.mark.parametrize("teardown_fault", ["raises", "hangs"])
async def test_browser_fetch_survives_failing_teardown(
    monkeypatch, teardown_fault: str
):
    faults: dict[str, Any] = (
        {"close_error": RuntimeError("close failed")}
        if teardown_fault == "raises"
        else {"close_gate": asyncio.Event()}
    )
    lost = _lost_crawler(**faults)

    async with _open_browser(
        monkeypatch,
        lost,
        _ok_crawler(),
        fetch_retries=1,
        cleanup_timeout_seconds=0.05,
    ) as fetch:
        async with asyncio.timeout(5):
            result = await fetch(TARGET)

    assert result.ok


async def test_browser_fetch_survives_failing_startup_attempt(monkeypatch):
    async with _open_browser(
        monkeypatch,
        _lost_crawler(),
        ConnectionError(DOWN),
        _ok_crawler(),
        fetch_retries=1,
        reconnect_attempts=2,
    ) as fetch:
        result = await fetch(TARGET)

    assert result.ok


async def test_browser_fetch_concurrent_losses_recover_once(monkeypatch):
    urls = [URL_A, URL_B]
    lost = _lost_crawler(url_gates={url: asyncio.Event() for url in urls})

    async with _open_browser(
        monkeypatch,
        lost,
        _ok_crawler(),
        _ok_crawler(),
        pool_size=len(urls),
        fetch_retries=1,
    ) as fetch:
        tasks = [asyncio.ensure_future(fetch(url)) for url in urls]
        try:
            await _until(lambda: {f"render:{url}" for url in urls} <= set(lost.trace))
        finally:
            for gate in lost.url_gates.values():
                gate.set()
        async with asyncio.timeout(5):
            results = await asyncio.gather(*tasks)

    assert all(result.ok for result in results)
    assert len(FakeAsyncWebCrawler.constructed) == 2


async def test_browser_fetch_waits_for_running_recovery(monkeypatch):
    lost = _lost_crawler()
    replacement = _GatedStart(_ok_crawler())

    async with _open_browser(
        monkeypatch,
        lost,
        replacement,
        pool_size=2,
        fetch_retries=1,
    ) as fetch:
        first = asyncio.ensure_future(fetch(URL_A))
        second = None
        try:
            await _until(lambda: replacement.entered.is_set() or first.done())
            assert replacement.entered.is_set(), "no recovery started"
            second = asyncio.ensure_future(fetch(URL_B))
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            replacement.gate.set()
        async with asyncio.timeout(5):
            results = await asyncio.gather(first, *([second] if second else []))

    assert all(result.ok for result in results)
    assert len(FakeAsyncWebCrawler.constructed) == 2
    assert URL_B in replacement.crawler.urls
    assert URL_B not in lost.urls


async def test_browser_fetch_cancelled_waiter_does_not_stop_recovery(monkeypatch):
    replacement = _GatedStart(_ok_crawler())

    async with _open_browser(
        monkeypatch,
        _lost_crawler(),
        replacement,
        pool_size=2,
        fetch_retries=1,
    ) as fetch:
        first = asyncio.ensure_future(fetch(URL_A))
        waiter = None
        try:
            await _until(lambda: replacement.entered.is_set() or first.done())
            assert replacement.entered.is_set(), "no recovery started"
            waiter = asyncio.ensure_future(fetch(URL_B))
            for _ in range(5):
                await asyncio.sleep(0)
            assert not waiter.done()
            waiter.cancel()
        finally:
            replacement.gate.set()
        async with asyncio.timeout(5):
            recovered = await first
        with pytest.raises(asyncio.CancelledError):
            await waiter

    assert recovered.ok
    assert replacement.crawler.opens == 1


async def test_browser_fetch_late_loss_shares_failed_recovery(monkeypatch):
    attempts = 2
    urls = [URL_A, URL_B]
    lost = _lost_crawler(url_gates={url: asyncio.Event() for url in urls})
    healthy = _ok_crawler()

    async with _open_browser(
        monkeypatch,
        lost,
        *[ConnectionError(DOWN)] * attempts,
        healthy,
        pool_size=len(urls),
        fetch_retries=1,
        reconnect_attempts=attempts,
    ) as fetch:
        first = asyncio.ensure_future(fetch(URL_A))
        late = asyncio.ensure_future(fetch(URL_B))
        try:
            await _until(lambda: {f"render:{url}" for url in urls} <= set(lost.trace))
            lost.url_gates[URL_A].set()
            async with asyncio.timeout(5):
                result_a = await first
                result_c = await fetch(URL_C)
            lost.url_gates[URL_B].set()
            async with asyncio.timeout(5):
                result_b = await late
        finally:
            for gate in lost.url_gates.values():
                gate.set()
            await asyncio.gather(first, late, return_exceptions=True)

    assert result_a.status is FetchStatus.UNAVAILABLE
    assert result_c.ok
    assert result_b.status is FetchStatus.UNAVAILABLE
    assert healthy.urls == [URL_C]
    assert len(FakeAsyncWebCrawler.constructed) == 2 + attempts


async def test_browser_fetch_recovers_on_next_call_after_failed_recovery(monkeypatch):
    attempts = 2

    async with _open_browser(
        monkeypatch,
        _lost_crawler(),
        *[ConnectionError(DOWN)] * attempts,
        _ok_crawler(),
        fetch_retries=0,
        reconnect_attempts=attempts,
    ) as fetch:
        first = await fetch(URL_A)
        second = await fetch(URL_B)

    assert first.status is FetchStatus.UNAVAILABLE
    assert second.ok


async def test_browser_fetch_exit_closes_replacement_crawler(monkeypatch):
    replacement = _ok_crawler()

    async with _open_browser(
        monkeypatch, _lost_crawler(), replacement, fetch_retries=1
    ) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert replacement.closes == 1


@pytest.fixture
def drops(monkeypatch) -> list[str]:
    """Record every forced close of crawl4ai's shared CDP connection."""
    calls: list[str] = []

    async def record() -> None:
        calls.append("dropped")

    monkeypatch.setattr(browser_module, "_drop_cached_connections", record)
    return calls


@pytest.mark.parametrize("teardown_fault", ["raises", "hangs"])
async def test_browser_fetch_drops_shared_connection_when_teardown_fails(
    monkeypatch, drops: list[str], teardown_fault: str
):
    faults: dict[str, Any] = (
        {"close_error": RuntimeError("close failed")}
        if teardown_fault == "raises"
        else {"close_gate": asyncio.Event()}
    )

    async with _open_browser(
        monkeypatch,
        _lost_crawler(**faults),
        _ok_crawler(),
        fetch_retries=1,
        cleanup_timeout_seconds=0.05,
    ) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert drops == ["dropped"]


async def test_browser_fetch_keeps_shared_connection_when_teardown_succeeds(
    monkeypatch, drops: list[str]
):
    async with _open_browser(
        monkeypatch, _lost_crawler(), _ok_crawler(), fetch_retries=1
    ) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert drops == []


async def test_invalidate_during_recovery_leaves_teardown_to_the_recovery(
    monkeypatch,
):
    old = _ok_crawler()
    replacement = _GatedStart(_ok_crawler())
    _script_crawlers(monkeypatch, old, replacement)
    connection = browser_module.CrawlerConnection(
        lambda: browser_module._new_crawler("ws://cdp"),
        _browser_config().browser,
    )
    await connection.open()
    lease = await connection.acquire()
    assert lease is not None

    recovering = asyncio.create_task(connection.recover(lease.generation))
    try:
        await _until(lambda: replacement.entered.is_set() or recovering.done())
        assert replacement.entered.is_set(), "no recovery started"
        await connection.invalidate(lease.generation)
    finally:
        replacement.gate.set()
    async with asyncio.timeout(5):
        recovered = await recovering
    await connection.close()

    assert recovered
    assert old.closes == 1


async def test_browser_fetch_arriving_during_failed_recovery_shares_the_failure(
    monkeypatch,
):
    failing = _GatedStart(FakeCrawler(), error=ConnectionError(DOWN))

    async with _open_browser(
        monkeypatch,
        _lost_crawler(),
        failing,
        _ok_crawler(),
        pool_size=2,
        fetch_retries=1,
        reconnect_attempts=1,
    ) as fetch:
        first = asyncio.ensure_future(fetch(URL_A))
        second = None
        try:
            await _until(lambda: failing.entered.is_set() or first.done())
            assert failing.entered.is_set(), "no recovery started"
            second = asyncio.ensure_future(fetch(URL_B))
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            failing.gate.set()
        async with asyncio.timeout(5):
            results = await asyncio.gather(first, *([second] if second else []))

    assert [result.status for result in results] == [FetchStatus.UNAVAILABLE] * 2
    assert len(FakeAsyncWebCrawler.constructed) == 2


async def test_exit_does_not_hang_on_a_stalled_close(monkeypatch):
    crawler = _ok_crawler(close_gate=asyncio.Event())

    async with asyncio.timeout(5):
        async with _open_browser(
            monkeypatch, crawler, cleanup_timeout_seconds=0.05
        ) as fetch:
            await fetch(URL_A)

    assert crawler.closes == 1


# ---------------------------------------------------------------------------
# Scrape.do
# ---------------------------------------------------------------------------


async def test_scrape_do_requests_rendered_html_with_unblocking(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(text="<h1>Article</h1>")
    fetch = ScrapeDoFetch("tok", http_client=client)

    result = await fetch(TARGET)

    request = httpx_mock.get_request()
    assert request is not None
    assert (request.method, request.url.host) == ("GET", "api.scrape.do")
    assert dict(request.url.params) == {
        "token": "tok",
        "url": TARGET,
        "super": "true",
        "render": "true",
    }
    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Article</h1>")


@pytest.mark.parametrize(
    ("status_code", "text", "expected"),
    [
        (200, "  ", (FetchStatus.EMPTY, None)),
        (404, "not found", (FetchStatus.HTTP_ERROR, 404)),
        (400, "bad target", (FetchStatus.HTTP_ERROR, 400)),
        (401, "no credits", (FetchStatus.UNAVAILABLE, None)),
        (429, "concurrency", (FetchStatus.UNAVAILABLE, None)),
        (502, "failed", (FetchStatus.UNAVAILABLE, None)),
    ],
)
async def test_scrape_do_maps_api_status(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    text: str,
    expected: tuple[FetchStatus, int | None],
):
    httpx_mock.add_response(status_code=status_code, text=text)
    fetch = ScrapeDoFetch("tok", http_client=client)

    assert outcome(await fetch(TARGET)) == expected


async def test_scrape_do_transport_error_is_unavailable(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    fetch = ScrapeDoFetch("tok", http_client=client)

    assert outcome(await fetch(TARGET)) == (FetchStatus.UNAVAILABLE, None)


# ---------------------------------------------------------------------------
# Bright Data Web Unlocker
# ---------------------------------------------------------------------------


async def test_bright_data_posts_zone_request_for_html(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(json={"status_code": 200, "body": "<h1>Article</h1>"})
    fetch = BrightDataFetch("key", "unlocker", http_client=client)

    result = await fetch(TARGET)

    request = httpx_mock.get_request()
    assert request is not None
    assert (request.method, str(request.url)) == (
        "POST",
        "https://api.brightdata.com/request",
    )
    assert request.headers["authorization"] == "Bearer key"
    assert json.loads(request.content) == {
        "zone": "unlocker",
        "url": TARGET,
        "format": "json",
    }
    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Article</h1>")


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (
            200,
            {"status_code": 403, "body": "denied"},
            (FetchStatus.HTTP_ERROR, 403),
        ),
        (200, {"status_code": 200, "body": ""}, (FetchStatus.EMPTY, None)),
        (200, {"unexpected": True}, (FetchStatus.UNAVAILABLE, None)),
        (401, {"error": "bad key"}, (FetchStatus.UNAVAILABLE, None)),
        (502, {"error": "upstream"}, (FetchStatus.UNAVAILABLE, None)),
    ],
)
async def test_bright_data_maps_outcomes(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    payload: dict[str, Any],
    expected: tuple[FetchStatus, int | None],
):
    httpx_mock.add_response(status_code=status_code, json=payload)
    fetch = BrightDataFetch("key", "unlocker", http_client=client)

    assert outcome(await fetch(TARGET)) == expected


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


async def test_fetch_tool_escalates_blocked_page_to_configured_provider(
    monkeypatch, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url=TARGET, status_code=403)
    httpx_mock.add_response(
        url=httpx.URL(
            "https://api.scrape.do/",
            params={
                "token": "t",
                "url": TARGET,
                "super": "true",
                "render": "true",
            },
        ),
        text="<html><body><h1>Unblocked</h1></body></html>",
    )
    config = ServerConfig(
        scraper=ScraperSettings(SCRAPER_PROVIDER="scrape_do", SCRAPE_DO_TOKEN="t"),
        browser=BrowserSettings(cdp_url=""),
    )

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert result.ok
    assert result.content == "# Unblocked"


async def test_fetch_tool_escalates_blocked_page_to_bright_data(
    monkeypatch, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url=TARGET, status_code=403)
    httpx_mock.add_response(
        url="https://api.brightdata.com/request",
        json={
            "status_code": 200,
            "body": "<html><body><h1>Unblocked</h1></body></html>",
        },
    )
    config = ServerConfig(
        scraper=ScraperSettings(
            SCRAPER_PROVIDER="bright_data",
            BRIGHT_DATA_API_KEY="key",
            BRIGHT_DATA_ZONE="unlocker",
        ),
        browser=BrowserSettings(cdp_url=""),
    )

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert result.ok
    assert result.content == "# Unblocked"


async def test_unconfigured_provider_keeps_target_failure(
    monkeypatch,
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
):
    httpx_mock.add_response(url=TARGET, status_code=403)
    config = ServerConfig(browser=BrowserSettings(cdp_url=""))

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert outcome(result) == (FetchStatus.HTTP_ERROR, 403)


ROUTER = "http://router/v1"


@pytest.mark.parametrize(
    ("make_config", "router_path", "warns"),
    [
        pytest.param(ServerConfig, None, (), id="unset"),
        pytest.param(
            lambda: ServerConfig(
                rag=RagSettings(embedding_model="e", embedding_dimensions=8),
                router=RouterSettings(api_base=ROUTER),
            ),
            None,
            (),
            id="unrelated-rag-config-falls-to-bm25",
        ),
        pytest.param(
            lambda: ServerConfig(
                fetch=FetchSettings(DECISION_MODEL="jev"),
                router=RouterSettings(api_base=ROUTER),
            ),
            "/v1/systemone",
            (),
            id="decision-configured",
        ),
        pytest.param(
            lambda: ServerConfig(fetch=FetchSettings(DECISION_MODEL="jev")),
            None,
            ("DECISION_MODEL",),
            id="decision-empty-router",
        ),
    ],
)
async def test_create_page_filter_selects_by_config(
    caplog,
    client: httpx.AsyncClient,
    make_config: Callable[[], ServerConfig],
    router_path: str | None,
    warns: tuple[str, ...],
):
    caplog.set_level("WARNING", logger="mcps.config")
    router_calls: list[httpx.Request] = []

    def router(request: httpx.Request) -> httpx.Response:
        router_calls.append(request)
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(router)) as mock:
        page_filter = create_page_filter(make_config(), mock)
        source = FetchResult(
            "https://a.example", FetchStatus.OK, "text/plain", "alpha beta"
        )

        result = await page_filter(source, "alpha")

    assert isinstance(page_filter, ChunkedRelevanceFilter)
    if router_path is None:
        assert router_calls == []
        assert result.status is not FetchStatus.FILTER_FAILED
    else:
        assert router_calls[0].url.path.endswith(router_path)
        assert result.status is FetchStatus.FILTER_FAILED
    messages = " ".join(record.getMessage() for record in caplog.records)
    warned = {name for name in ("DECISION_MODEL",) if name in messages}
    assert warned == set(warns)


async def test_default_page_filter_drops_unrelated_section_of_short_page(
    client: httpx.AsyncClient,
):
    # Arrange: both sections fit one window of the decision ranker's chunker.
    relevant = "Quantum routing reduces latency in mesh networks. " * 6
    unrelated = "Bread flour needs water, salt and time to rise. " * 30
    source = FetchResult(
        "https://a.example",
        FetchStatus.OK,
        "text/markdown",
        f"{relevant}\n\n{unrelated}DISTANT_MARKER",
    )
    page_filter = create_page_filter(ServerConfig(), client)

    # Act
    result = await page_filter(source, "quantum routing latency")

    # Assert
    assert result.status is FetchStatus.OK
    assert relevant.strip() in result.content
    assert "DISTANT_MARKER" not in result.content


# ---------------------------------------------------------------------------
# Non-textual pages: the browser reports the final media type for retry
# ---------------------------------------------------------------------------

PDF_MIME = "application/pdf"
VIEWER_HTML = "<h1>viewer</h1>"


def _handoff(url: str, mime: str) -> tuple[FetchStatus, str, str, int | None, str]:
    return FetchStatus.UNSUPPORTED, mime, "", None, url


def _handoff_of(result: FetchResult) -> tuple[FetchStatus, str, str, int | None, str]:
    return result.status, result.mime, result.content, result.http_status, result.url


async def test_browser_redirect_to_pdf_reports_final_type(monkeypatch):
    # The crawl headers describe the first redirect hop; the hook sees the last.
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=VIEWER_HTML, response_headers={"content-type": "text/html"}
        ),
        final_response=navigation_response(PDF_MIME),
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert _handoff_of(result) == _handoff(TARGET, PDF_MIME)


async def test_browser_blocked_viewer_page_still_reports_pdf(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(success=False),
        final_response=navigation_response("Application/PDF; qs=0.001"),
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert _handoff_of(result) == _handoff(TARGET, PDF_MIME)


async def test_browser_without_hook_call_uses_crawl_headers_for_pdf(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=VIEWER_HTML, response_headers={"content-type": PDF_MIME}
        )
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert _handoff_of(result) == _handoff(TARGET, PDF_MIME)


async def test_browser_pdf_handoff_is_logged_at_info_with_url_and_type(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    crawler = FakeCrawler(
        result=crawl_result(cleaned_html=VIEWER_HTML),
        final_response=navigation_response(PDF_MIME),
    )

    with caplog.at_level("DEBUG", logger=browser_module.logger.name):
        async with _open_browser(monkeypatch, crawler) as fetch:
            await fetch(TARGET)

    handoffs = [r for r in caplog.records if TARGET in r.getMessage()]
    assert [r.levelname for r in handoffs] == ["INFO"]
    assert PDF_MIME in handoffs[0].getMessage()


async def test_browser_http_error_keeps_its_status_for_binary_type(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(status_code=404, cleaned_html=VIEWER_HTML),
        final_response=navigation_response(PDF_MIME),
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert outcome(result) == (FetchStatus.HTTP_ERROR, 404)


async def test_browser_textual_final_type_overrides_stale_pdf_header(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(
            cleaned_html=OK_PAGE, response_headers={"content-type": PDF_MIME}
        ),
        final_response=navigation_response("text/html; charset=utf-8"),
    )

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert result.content == OK_PAGE


async def test_browser_pdf_type_does_not_leak_into_next_fetch_of_session(monkeypatch):
    crawler = FakeCrawler(
        result=crawl_result(cleaned_html=VIEWER_HTML),
        final_response=navigation_response(PDF_MIME),
    )

    async with _open_browser(monkeypatch, crawler, pool_size=1) as fetch:
        first = await fetch(TARGET)
        crawler.result = crawl_result(cleaned_html=OK_PAGE)
        crawler.final_response = NO_HOOK_CALL
        second = await fetch(TARGET)

    assert first.status is FetchStatus.UNSUPPORTED
    assert second.ok
    assert second.content == OK_PAGE


@pytest.mark.parametrize(
    "final_response",
    [None, _RaisingHeadersResponse(), navigation_response(None)],
    ids=["no-response", "headers-raise", "no-content-type"],
)
async def test_browser_unreadable_final_response_keeps_render_result(
    monkeypatch, final_response: Any
):
    crawler = _ok_crawler(final_response=final_response)

    async with _open_browser(monkeypatch, crawler) as fetch:
        result = await fetch(TARGET)

    assert result.ok
    assert result.content == OK_PAGE


async def test_browser_hook_failure_is_logged_at_debug(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    crawler = _ok_crawler(final_response=_RaisingHeadersResponse())

    with caplog.at_level("DEBUG", logger=browser_module.logger.name):
        async with _open_browser(monkeypatch, crawler) as fetch:
            await fetch(TARGET)

    levels = {
        record.levelname
        for record in caplog.records
        if "response is gone" in record.getMessage()
    }
    assert levels == {"DEBUG"}


async def test_browser_recovered_crawler_still_reports_pdf(monkeypatch):
    lost = _lost_crawler()
    healthy = FakeCrawler(
        result=crawl_result(cleaned_html=VIEWER_HTML),
        final_response=navigation_response(PDF_MIME),
    )

    async with _open_browser(
        monkeypatch,
        lost,
        healthy,
        fetch_retries=1,
        reconnect_attempts=1,
    ) as fetch:
        result = await fetch(TARGET)

    assert _handoff_of(result) == _handoff(TARGET, PDF_MIME)


async def test_browser_connection_loss_wins_over_recorded_pdf_type(monkeypatch):
    lost = _lost_crawler(final_response=navigation_response(PDF_MIME))
    healthy = _ok_crawler()

    async with _open_browser(
        monkeypatch,
        lost,
        healthy,
        pool_size=1,
        fetch_retries=1,
        reconnect_attempts=1,
    ) as fetch:
        retried = await fetch(TARGET)
        healthy.final_response = NO_HOOK_CALL
        next_fetch = await fetch(TARGET)

    assert retried.ok
    assert next_fetch.ok
    assert next_fetch.content == OK_PAGE
