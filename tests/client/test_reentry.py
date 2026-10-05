"""
Tests for reentrant (nested) requests - requests issued from event hooks,
authentication flows or transport wrappers while the same client is already
sending a request on the same thread/task.
"""

from __future__ import annotations

import threading
import time
import typing
import warnings

import anyio
import pytest

import httpx

BASE = "http://127.0.0.1:8000"


def app(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/redirect":
        return httpx.Response(303, headers={"location": "/final"})
    if path == "/redirect-cookie":
        return httpx.Response(303, headers={"location": "/echo"})
    if path == "/set-cookie":
        return httpx.Response(200, headers={"set-cookie": "nested=1"})
    if path == "/challenge":
        challenge = 'Digest realm="httpx@example.org", nonce="abc123", qop="auth"'
        return httpx.Response(401, headers={"www-authenticate": challenge})
    if path == "/exhaust":
        raise httpx.PoolTimeout("mock pool exhausted")
    if path == "/echo":
        return httpx.Response(200, text=request.headers.get("cookie", ""))
    return httpx.Response(200, text="ok")


# ---------------------------------------------------------------------------
# Configuration / construction
# ---------------------------------------------------------------------------


def test_default_configuration() -> None:
    client = httpx.Client(transport=httpx.MockTransport(app))
    assert client.max_nested_depth == 4
    assert client.on_nested_depth is httpx.NestedRequestPolicy.REJECT
    assert client.nested_depth == 0
    assert client.nested_source is None
    client.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_nested_depth": 0},
        {"max_nested_depth": -1},
        {"max_nested_depth": 1.5},
        {"max_nested_depth": True},
        {"max_nested_depth": "1"},
        {"on_nested_depth": "bogus"},
        {"on_nested_depth": "ALLOW"},
        {"on_nested_depth": 3},
    ],
)
def test_invalid_configuration_at_construction_sync(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        httpx.Client(transport=httpx.MockTransport(app), **kwargs)


@pytest.mark.anyio
async def test_invalid_configuration_at_construction_async() -> None:
    with pytest.raises(ValueError):
        httpx.AsyncClient(transport=httpx.MockTransport(app), max_nested_depth=0)
    with pytest.raises(ValueError):
        httpx.AsyncClient(
            transport=httpx.MockTransport(app), on_nested_depth="nope"
        )


# ---------------------------------------------------------------------------
# Origin identification and nesting depth - sync
# ---------------------------------------------------------------------------


def test_nested_depth_resets_after_send() -> None:
    client = httpx.Client(transport=httpx.MockTransport(app))
    client.get(BASE + "/")
    assert client.nested_depth == 0
    assert client.nested_source is None
    client.close()


def test_nested_request_from_request_hook() -> None:
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(app))

    def request_hook(request: httpx.Request) -> None:
        if request.url.path == "/":
            client.get(BASE + "/inner")
        else:
            seen.append((client.nested_depth, client.nested_source))

    client.event_hooks = {"request": [request_hook]}
    client.get(BASE + "/")
    assert seen == [(1, "request_hook")]
    assert client.nested_depth == 0
    client.close()


def test_nested_request_from_response_hook() -> None:
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(app))

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            client.get(BASE + "/inner")
        else:
            seen.append((client.nested_depth, client.nested_source))

    client.event_hooks = {"response": [response_hook]}
    client.get(BASE + "/")
    assert seen == [(1, "response_hook")]
    client.close()


class TokenAuth(httpx.Auth):
    """Authentication flow that fetches a token using the same client."""

    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.token: str | None = None

    def auth_flow(
        self, request: httpx.Request
    ) -> typing.Generator[httpx.Request, httpx.Response, None]:
        if self.token is None:
            response = self.client.send(
                httpx.Request("GET", BASE + "/token"), auth=None
            )
            self.token = response.text
        request.headers["Authorization"] = self.token
        yield request


def test_nested_request_from_auth_flow() -> None:
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(app), auth=None)

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/token":
            seen.append((client.nested_depth, client.nested_source))

    client.auth = TokenAuth(client)
    client.event_hooks = {"response": [response_hook]}

    response = client.get(BASE + "/")
    assert seen == [(1, "auth")]
    assert response.request.headers["Authorization"] == "ok"
    client.close()


class ReentrantSyncTransport(httpx.BaseTransport):
    """Transport wrapper that calls back into the same client."""

    def __init__(self, client: httpx.Client, inner: httpx.BaseTransport) -> None:
        self.client = client
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reenter":
            return self.client.send(
                httpx.Request("GET", BASE + "/inner"), auth=None
            )
        return self.inner.handle_request(request)


def test_nested_request_from_transport_wrapper() -> None:
    seen = []
    inner = httpx.MockTransport(app)
    client = httpx.Client(transport=inner)
    client._transport = ReentrantSyncTransport(client, inner)

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/inner":
            seen.append((client.nested_depth, client.nested_source))

    client.event_hooks = {"response": [response_hook]}
    response = client.get(BASE + "/reenter")
    assert response.status_code == 200
    assert seen == [(1, "transport")]
    client.close()


# ---------------------------------------------------------------------------
# Depth cap policies - sync
# ---------------------------------------------------------------------------


def _install_levels(client: httpx.Client, record: list[str]) -> None:
    """
    /a nests /b, /b nests /c - so /c runs two levels deep inside /a.
    """

    def response_hook(response: httpx.Response) -> None:
        path = response.request.url.path
        record.append((path, client.nested_depth))
        if path == "/a":
            client.get(BASE + "/b")
        elif path == "/b":
            client.get(BASE + "/c")

    client.event_hooks = {"response": [response_hook]}


def test_nested_depth_rejected() -> None:
    rejected = []
    client = httpx.Client(
        transport=httpx.MockTransport(app), max_nested_depth=1
    )

    def response_hook(response: httpx.Response) -> None:
        path = response.request.url.path
        if path == "/a":
            client.get(BASE + "/b")
        elif path == "/b":
            try:
                client.get(BASE + "/c")
            except httpx.NestedDepthExceeded as exc:
                rejected.append(exc)

    client.event_hooks = {"response": [response_hook]}

    outer = client.get(BASE + "/a")
    assert outer.status_code == 200
    assert len(rejected) == 1
    exc = rejected[0]
    assert isinstance(exc, httpx.NestedRequestError)
    assert isinstance(exc, httpx.RequestError)
    assert exc.request.url.path == "/c"
    message = str(exc)
    assert "nesting depth 2" in message
    assert "max_nested_depth=1" in message
    assert "response_hook" in message
    assert client.nested_depth == 0
    client.close()


def test_nested_depth_warned_and_allowed() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app),
        max_nested_depth=1,
        on_nested_depth="warn",
    )
    assert client.on_nested_depth is httpx.NestedRequestPolicy.WARN
    record: list = []
    _install_levels(client, record)

    with pytest.warns(httpx.NestedRequestWarning, match="nesting depth 2"):
        client.get(BASE + "/a")

    # The nested requests were still sent.
    paths = [path for path, _ in record]
    assert "/b" in paths and "/c" in paths
    client.close()


def test_nested_depth_allow_policy_silently_allows() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app),
        max_nested_depth=1,
        on_nested_depth=httpx.NestedRequestPolicy.ALLOW,
    )
    record: list = []
    _install_levels(client, record)

    with warnings.catch_warnings():
        warnings.simplefilter("error", httpx.NestedRequestWarning)
        client.get(BASE + "/a")

    assert {path for path, _ in record} == {"/a", "/b", "/c"}
    client.close()


def test_unlimited_depth_with_none() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app),
        max_nested_depth=None,
        on_nested_depth="reject",
    )
    record: list = []
    _install_levels(client, record)
    client.get(BASE + "/a")  # depth 2 is fine with no cap
    assert {path for path, _ in record} == {"/a", "/b", "/c"}
    client.close()


# ---------------------------------------------------------------------------
# State isolation - sync
# ---------------------------------------------------------------------------


def test_nested_set_cookie_does_not_write_back() -> None:
    client = httpx.Client(transport=httpx.MockTransport(app))

    seen_in_nested = []

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            nested = client.get(BASE + "/set-cookie")
            seen_in_nested.append(nested.cookies.get("nested"))

    client.event_hooks = {"response": [response_hook]}
    client.get(BASE + "/")

    # The nested request could observe its own Set-Cookie writeback, but the
    # client jar and the outer send must remain untouched.
    assert seen_in_nested == ["1"]
    assert client.cookies.get("nested") is None
    client.close()


def test_nested_cookies_do_not_reach_outer_redirect_chain() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app), follow_redirects=True
    )

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/redirect-cookie":
            client.get(BASE + "/set-cookie")

    client.event_hooks = {"response": [response_hook]}
    response = client.get(BASE + "/redirect-cookie")

    assert response.url.path == "/echo"
    assert response.text == ""  # No 'Cookie: nested=1' on the outer redirect.
    assert client.cookies.get("nested") is None
    client.close()


def test_nested_request_does_not_rewrite_outer_timeout() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app), timeout=httpx.Timeout(12.0, pool=7.0)
    )
    outer_request = client.build_request("GET", BASE + "/")
    original_timeout = dict(outer_request.extensions["timeout"])
    seen = []

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            client.get(BASE + "/inner", timeout=33.0)
        elif response.request.url.path == "/inner":
            seen.append(dict(response.request.extensions["timeout"]))

    client.event_hooks = {"response": [response_hook]}
    client.send(outer_request)

    assert outer_request.extensions["timeout"] == original_timeout
    assert outer_request.extensions["timeout"]["pool"] == 7.0
    # While in flight, nested requests use a zero pool timeout as a capacity
    # guard, but keep the rest of their own timeout override.
    assert seen[0]["pool"] == 0.0
    assert seen[0]["read"] == 33.0
    client.close()


def test_nested_request_does_not_rewrite_outer_history() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app), follow_redirects=True
    )

    def response_hook(response: httpx.Response) -> None:
        if (
            response.request.url.path == "/redirect"
            and client.nested_depth == 0
        ):
            nested = client.get(BASE + "/redirect")
            # The nested send owns its redirect history.
            nested_paths = [r.request.url.path for r in nested.history]
            assert nested_paths == ["/redirect"]

    client.event_hooks = {"response": [response_hook]}
    response = client.get(BASE + "/redirect")

    # Outer chain contains only outer responses.
    assert [r.request.url.path for r in response.history] == ["/redirect"]
    client.close()


def test_nested_request_does_not_mutate_outer_auth_state() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app),
        auth=httpx.DigestAuth("user", "password"),
    )
    assert client.auth is not None
    assert client.auth._last_challenge is None

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            # A 401 digest challenge inside the nested request mutates the
            # auth state of an isolated copy, not the client-level auth.
            client.get(BASE + "/challenge")

    client.event_hooks = {"response": [response_hook]}
    response = client.get(BASE + "/")

    assert response.status_code == 200
    assert client.auth._last_challenge is None
    client.close()


# ---------------------------------------------------------------------------
# Capacity contention - sync
# ---------------------------------------------------------------------------


def test_nested_capacity_contention_is_distinguishable() -> None:
    client = httpx.Client(transport=httpx.MockTransport(app))
    seen = []

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            try:
                client.get(BASE + "/exhaust")
            except httpx.NestedCapacityError as exc:
                seen.append(exc)

    client.event_hooks = {"response": [response_hook]}
    outer = client.get(BASE + "/")

    assert outer.status_code == 200
    assert len(seen) == 1
    exc = seen[0]
    assert isinstance(exc, httpx.PoolTimeout)
    assert isinstance(exc, httpx.NestedRequestError)
    assert exc.request.url.path == "/exhaust"
    message = str(exc)
    assert "nested" in message
    assert "reentrancy capacity contention" in message
    client.close()


def test_nested_capacity_contention_fails_fast(server) -> None:
    seen = []
    client = httpx.Client(
        limits=httpx.Limits(max_connections=1),
    )

    def response_hook(response: httpx.Response) -> None:
        started = time.perf_counter()
        try:
            client.get(str(server.url))
        except httpx.NestedCapacityError as exc:
            elapsed = time.perf_counter() - started
            seen.append((elapsed, isinstance(exc, httpx.PoolTimeout)))

    client.event_hooks = {"response": [response_hook]}
    response = client.get(str(server.url))

    assert response.status_code == 200
    assert len(seen) == 1
    elapsed, is_pool_timeout = seen[0]
    assert is_pool_timeout
    # The default pool timeout is 5s; nested contention must surface at once
    # rather than waiting silently for the outer request.
    assert elapsed < 3.0
    client.close()


def test_top_level_pool_contention_keeps_pool_timeout(server) -> None:
    # Holding a response open *outside* of `send()` is not reentrancy, so the
    # ordinary exception type must be preserved unchanged.
    client = httpx.Client(
        limits=httpx.Limits(max_connections=1),
        timeout=httpx.Timeout(None, pool=0.05),
    )
    with client.stream("GET", str(server.url)):
        with pytest.raises(httpx.PoolTimeout) as exc_info:
            client.get(str(server.url))
    assert type(exc_info.value) is httpx.PoolTimeout
    client.close()


# ---------------------------------------------------------------------------
# Per-thread isolation - sync
# ---------------------------------------------------------------------------


def test_nested_state_isolated_between_threads() -> None:
    barrier = threading.Barrier(2, timeout=10)
    recorded: list[tuple[int, int]] = []
    client = httpx.Client(transport=httpx.MockTransport(app))

    def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/nested":
            # Both threads rendezvous here, inside their own nested send.
            barrier.wait()
            recorded.append((threading.get_ident(), client.nested_depth))
        elif response.request.url.path == "/":
            client.get(BASE + "/nested")

    client.event_hooks = {"response": [response_hook]}

    threads = [
        threading.Thread(target=client.get, args=(BASE + "/",)) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(recorded) == 2
    assert {ident for ident, _ in recorded} == {t.ident for t in threads}
    assert {depth for _, depth in recorded} == {1}
    assert client.nested_depth == 0
    client.close()


# ---------------------------------------------------------------------------
# Async equivalents
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_nested_from_request_and_response_hook() -> None:
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(app))

    async def request_hook(request: httpx.Request) -> None:
        if request.url.path == "/req":
            # Nested request issued from a request hook.
            await client.get(BASE + "/inner-req")
        elif request.url.path == "/inner-req":
            seen.append((client.nested_depth, client.nested_source))

    async def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/resp":
            # Nested request issued from a response hook.
            await client.get(BASE + "/inner-resp")
        elif response.request.url.path == "/inner-resp":
            seen.append((client.nested_depth, client.nested_source))

    client.event_hooks = {"request": [request_hook], "response": [response_hook]}
    await client.get(BASE + "/req")
    await client.get(BASE + "/resp")
    assert (1, "request_hook") in seen
    assert (1, "response_hook") in seen
    assert client.nested_depth == 0
    await client.aclose()


class AsyncTokenAuth(httpx.Auth):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.token: str | None = None

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> typing.AsyncGenerator[httpx.Request, httpx.Response]:
        if self.token is None:
            response = await self.client.send(
                httpx.Request("GET", BASE + "/token"), auth=None
            )
            self.token = response.text
        request.headers["Authorization"] = self.token
        yield request


@pytest.mark.anyio
async def test_async_nested_from_auth_flow() -> None:
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(app))

    async def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/token":
            seen.append((client.nested_depth, client.nested_source))

    client.auth = AsyncTokenAuth(client)
    client.event_hooks = {"response": [response_hook]}

    response = await client.get(BASE + "/")
    assert seen == [(1, "auth")]
    assert response.request.headers["Authorization"] == "ok"
    await client.aclose()


class ReentrantAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(
        self, client: httpx.AsyncClient, inner: httpx.AsyncBaseTransport
    ) -> None:
        self.client = client
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reenter":
            return await self.client.send(
                httpx.Request("GET", BASE + "/inner"), auth=None
            )
        return await self.inner.handle_async_request(request)


@pytest.mark.anyio
async def test_async_nested_from_transport_wrapper() -> None:
    seen = []
    inner = httpx.MockTransport(app)
    client = httpx.AsyncClient(transport=inner)
    client._transport = ReentrantAsyncTransport(client, inner)

    async def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/inner":
            seen.append((client.nested_depth, client.nested_source))

    client.event_hooks = {"response": [response_hook]}
    response = await client.get(BASE + "/reenter")
    assert response.status_code == 200
    assert seen == [(1, "transport")]
    await client.aclose()


@pytest.mark.anyio
async def test_async_nested_depth_rejected_and_recovered() -> None:
    rejected = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(app), max_nested_depth=1
    )

    async def response_hook(response: httpx.Response) -> None:
        path = response.request.url.path
        if path == "/a":
            await client.get(BASE + "/b")
        elif path == "/b":
            try:
                await client.get(BASE + "/c")
            except httpx.NestedDepthExceeded as exc:
                rejected.append(exc)

    client.event_hooks = {"response": [response_hook]}
    outer = await client.get(BASE + "/a")

    assert outer.status_code == 200
    assert len(rejected) == 1
    assert isinstance(rejected[0], httpx.NestedRequestError)
    assert "response_hook" in str(rejected[0])
    assert client.nested_depth == 0

    # Subsequent top-level requests are unaffected.
    follow_up = await client.get(BASE + "/")
    assert follow_up.status_code == 200
    assert client.nested_depth == 0
    await client.aclose()


@pytest.mark.anyio
async def test_async_nested_state_isolation() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(app), follow_redirects=True
    )

    async def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            await client.get(BASE + "/set-cookie")

    client.event_hooks = {"response": [response_hook]}
    await client.get(BASE + "/")
    assert client.cookies.get("nested") is None

    # Timeout override of an outer request survives nested sends.
    outer_request = client.build_request("GET", BASE + "/")
    original_timeout = dict(outer_request.extensions["timeout"])

    async def timeout_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            await client.get(BASE + "/inner", timeout=33.0)

    client.event_hooks = {"response": [timeout_hook]}
    await client.send(outer_request)
    assert outer_request.extensions["timeout"] == original_timeout
    await client.aclose()


@pytest.mark.anyio
async def test_async_nested_capacity_contention_is_distinguishable() -> None:
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(app))

    async def response_hook(response: httpx.Response) -> None:
        if response.request.url.path == "/":
            try:
                await client.get(BASE + "/exhaust")
            except httpx.NestedCapacityError as exc:
                seen.append(exc)

    client.event_hooks = {"response": [response_hook]}
    outer = await client.get(BASE + "/")

    assert outer.status_code == 200
    assert len(seen) == 1
    assert isinstance(seen[0], httpx.PoolTimeout)
    assert "reentrancy capacity contention" in str(seen[0])
    await client.aclose()


@pytest.mark.anyio
async def test_async_nested_capacity_contention_fails_fast(server) -> None:
    seen = []
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=1))

    async def response_hook(response: httpx.Response) -> None:
        started = time.perf_counter()
        try:
            await client.get(str(server.url))
        except httpx.NestedCapacityError as exc:
            elapsed = time.perf_counter() - started
            seen.append((elapsed, isinstance(exc, httpx.PoolTimeout)))

    client.event_hooks = {"response": [response_hook]}
    response = await client.get(str(server.url))

    assert response.status_code == 200
    assert len(seen) == 1
    elapsed, is_pool_timeout = seen[0]
    assert is_pool_timeout
    assert elapsed < 3.0
    await client.aclose()


@pytest.mark.anyio
async def test_async_nested_state_isolated_between_tasks() -> None:
    # Two tasks rendezvous while both are inside a nested send, so that any
    # leaking of nesting state between tasks would be observed.
    a_ready = anyio.Event()
    b_ready = anyio.Event()
    recorded: list[str] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(app))

    async def response_hook(response: httpx.Response) -> None:
        path = response.request.url.path
        if path == "/a-root":
            await client.get(BASE + "/nested-a")
        elif path == "/b-root":
            await client.get(BASE + "/nested-b")
        elif path == "/nested-a":
            a_ready.set()
            await b_ready.wait()
            recorded.append(f"a:{client.nested_depth}")
        elif path == "/nested-b":
            b_ready.set()
            await a_ready.wait()
            recorded.append(f"b:{client.nested_depth}")

    client.event_hooks = {"response": [response_hook]}

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(client.get, BASE + "/a-root")
        task_group.start_soon(client.get, BASE + "/b-root")

    assert sorted(recorded) == ["a:1", "b:1"]
    assert client.nested_depth == 0
    await client.aclose()


# ---------------------------------------------------------------------------
# Non-reentrant behaviour remains unchanged
# ---------------------------------------------------------------------------


def test_plain_send_keeps_headers_and_history() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(app), follow_redirects=True
    )
    response = client.get(BASE + "/redirect")
    assert response.status_code == 200
    assert response.history[0].status_code == 303
    assert response.request.headers["host"] == "127.0.0.1:8000"
    assert client.nested_depth == 0
    client.close()
