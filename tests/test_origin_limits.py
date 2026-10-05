"""
Tests for per-origin connection quotas and in-flight accounting.
"""

import threading
import time
import typing

import anyio
import pytest
from uvicorn.config import Config

import httpx
from tests import conftest
from tests.concurrency import sleep


def _origin_key(url: httpx.URL) -> tuple[str, str, int]:
    default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443}
    return (url.scheme, url.host, url.port or default_ports[url.scheme])


def _wait_for(condition: typing.Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


async def _await_for(
    condition: typing.Callable[[], typing.Awaitable[bool]], timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await condition():
            return
        await sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


@pytest.fixture
def second_server():
    config = Config(app=conftest.app, lifespan="off", loop="asyncio", port=8011)
    server = conftest.TestServer(config=config)
    yield from conftest.serve_in_thread(server)


# ---------------------------------------------------------------------------
# Construction time validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"max_connections": 0}, id="zero-max-connections"),
        pytest.param({"max_connections": -1}, id="negative-max-connections"),
        pytest.param({"max_connections": 1.5}, id="float-max-connections"),
        pytest.param({"max_connections": True}, id="bool-max-connections"),
        pytest.param({"max_connections": "2"}, id="str-max-connections"),
        pytest.param({"max_keepalive_connections": -1}, id="negative-keepalive"),
        pytest.param(
            {"max_connections": 2, "max_keepalive_connections": 3},
            id="keepalive-above-cap",
        ),
        pytest.param({"pool_timeout": -1.0}, id="negative-pool-timeout"),
        pytest.param({"pool_timeout": "soon"}, id="str-pool-timeout"),
    ],
)
def test_invalid_origin_limits_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        httpx.OriginLimits(**kwargs)


@pytest.mark.parametrize(
    "key",
    [
        "ftp://example.com",
        "not a url",
        "https://",
        12345,
        ("https", "example.com"),
        ("https", "example.com", "443"),
    ],
)
def test_invalid_origin_keys_raise_at_construction(key):
    with pytest.raises(ValueError):
        httpx.Limits(per_origin={key: httpx.OriginLimits(max_connections=1)})


def test_invalid_origin_limit_value_type_raises():
    with pytest.raises(ValueError):
        httpx.Limits(per_origin={"https://example.com": 5})


def test_origin_keys_are_normalized():
    limits = httpx.Limits(
        per_origin={
            "https://Example.COM": httpx.OriginLimits(max_connections=2),
        }
    )
    assert limits == httpx.Limits(
        per_origin={
            ("https", "example.com", 443): httpx.OriginLimits(max_connections=2),
        }
    )


def test_client_construction_rejects_invalid_origin_limits():
    with pytest.raises(ValueError):
        httpx.Client(
            limits=httpx.Limits(
                per_origin={
                    "http://127.0.0.1:8000": httpx.OriginLimits(max_connections=0)
                }
            )
        )


# ---------------------------------------------------------------------------
# Sync transport
# ---------------------------------------------------------------------------


def test_sync_unconfigured_origin_keeps_global_pool_timeout(server):
    # Global pool semantics are untouched: the global PoolTimeout is raised,
    # not the new OriginPoolTimeout.
    client = httpx.Client(
        limits=httpx.Limits(max_connections=1),
        timeout=httpx.Timeout(None, pool=1e-4),
        trust_env=False,
    )
    with client:
        with client.stream("GET", server.url):
            try:
                client.get(server.url)
            except httpx.PoolTimeout as exc:
                assert type(exc) is httpx.PoolTimeout
            else:  # pragma: no cover
                raise AssertionError("Expected PoolTimeout")


def test_sync_unconfigured_origin_is_still_accounted(server):
    client = httpx.Client(trust_env=False)
    key = _origin_key(server.url)
    with client:

        def get() -> None:
            response = client.get(server.url.copy_with(path="/slow_response"))
            assert response.status_code == 200

        thread = threading.Thread(target=get)
        thread.start()
        _wait_for(
            lambda: (
                client.get_origin_stats().get(key) is not None
                and client.get_origin_stats()[key].in_flight == 1
            )
        )

        stats = client.get_origin_stats()
        assert stats[key].in_flight == 1
        assert stats[key].waiting == 0
        assert stats[key].max_connections is None

        thread.join()

        stats = client.get_origin_stats()
        assert stats[key].in_flight == 0
        assert stats[key].idle == 1


def test_sync_origin_cap_isolation_wait_timeout(server, second_server):
    origin_a = _origin_key(server.url)
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=0.2),
        }
    )
    client = httpx.Client(limits=limits, trust_env=False)
    with client:
        # Saturate origin A.
        with client.stream("GET", server.url):
            stats = client.get_origin_stats()
            assert stats[origin_a].in_flight == 1
            assert stats[origin_a].max_connections == 1

            # Origin B is not blocked by origin A's quota.
            response = client.get(second_server.url)
            assert response.status_code == 200

            # A queued request for A fails with the independent origin
            # timeout, naming the origin that was holding the quota.
            started = time.monotonic()
            with pytest.raises(httpx.OriginPoolTimeout) as exc_info:
                client.get(server.url.copy_with(path="/slow_response"))
            elapsed = time.monotonic() - started

            assert exc_info.value.origin == origin_a
            assert isinstance(exc_info.value, httpx.PoolTimeout)
            assert 0.15 < elapsed < 0.8

            stats = client.get_origin_stats()
            assert stats[origin_a].waiting == 0

        # Quota is fully usable again once the holder releases it.
        response = client.get(server.url)
        assert response.status_code == 200
        assert response.extensions["origin_waited"] == (False,)


def test_sync_fifo_order_and_waited_flag(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=5.0)
        }
    )
    client = httpx.Client(limits=limits, trust_env=False)
    completed: list[int] = []
    with client:
        holder_ready = threading.Event()
        release_holder = threading.Event()

        def hold() -> None:
            with client.stream("GET", server.url) as response:
                assert response.extensions["origin_waited"] == (False,)
                holder_ready.set()
                release_holder.wait(5.0)

        holder = threading.Thread(target=hold)
        holder.start()
        holder_ready.wait(5.0)

        def waiter(tag: int) -> None:
            response = client.get(server.url)
            assert response.extensions["origin_waited"] == (True,)
            completed.append(tag)

        first = threading.Thread(target=waiter, args=(1,))
        second = threading.Thread(target=waiter, args=(2,))
        first.start()
        key = _origin_key(server.url)
        _wait_for(lambda: client.get_origin_stats()[key].waiting == 1)
        second.start()
        _wait_for(lambda: client.get_origin_stats()[key].waiting == 2)

        release_holder.set()
        first.join()
        second.join()
        holder.join()

        assert completed == [1, 2]


def test_sync_idle_reserve_eviction(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(
                max_connections=2,
                max_keepalive_connections=1,
                pool_timeout=5.0,
            )
        }
    )
    client = httpx.Client(limits=limits, trust_env=False)
    key = _origin_key(server.url)
    with client:
        barrier = threading.Barrier(2)

        def request() -> None:
            response = client.get(server.url.copy_with(path="/slow_response"))
            assert response.status_code == 200
            barrier.wait(5.0)

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stats = client.get_origin_stats()
        assert stats[key].idle == 1
        assert stats[key].in_flight == 0


def test_sync_zero_idle_reserve_evicts_all(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(
                max_connections=2,
                max_keepalive_connections=0,
                pool_timeout=5.0,
            )
        }
    )
    client = httpx.Client(limits=limits, trust_env=False)
    key = _origin_key(server.url)
    with client:
        barrier = threading.Barrier(2)

        def request() -> None:
            response = client.get(server.url.copy_with(path="/slow_response"))
            assert response.status_code == 200
            barrier.wait(5.0)

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stats = client.get_origin_stats()
        assert key not in stats


def test_sync_ledger_converges_after_close(server):
    client = httpx.Client(
        limits=httpx.Limits(
            per_origin={
                str(server.url): httpx.OriginLimits(max_connections=2),
            }
        ),
        trust_env=False,
    )
    with client:
        response = client.get(server.url)
        assert response.status_code == 200
        assert client.get_origin_stats()

    # Client closed: all connections are gone, so the ledger must converge.
    assert client.get_origin_stats() == {}


def test_sync_custom_transport_has_no_origin_stats():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    with httpx.Client(transport=transport) as client:
        assert client.get_origin_stats() == {}
        response = client.get("https://example.com")
        assert response.status_code == 200
        assert client.get_origin_stats() == {}


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_origin_cap_isolation_wait_timeout(server, second_server):
    origin_a = _origin_key(server.url)
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=0.2),
        }
    )
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        async with client.stream("GET", server.url):
            stats = await client.get_origin_stats()
            assert stats[origin_a].in_flight == 1

            response = await client.get(second_server.url)
            assert response.status_code == 200

            started = time.monotonic()
            with pytest.raises(httpx.OriginPoolTimeout) as exc_info:
                await client.get(server.url.copy_with(path="/slow_response"))
            elapsed = time.monotonic() - started

            assert exc_info.value.origin == origin_a
            assert isinstance(exc_info.value, httpx.PoolTimeout)
            assert 0.15 < elapsed < 0.8

            stats = await client.get_origin_stats()
            assert stats[origin_a].waiting == 0

        response = await client.get(server.url)
        assert response.status_code == 200
        assert response.extensions["origin_waited"] == (False,)


@pytest.mark.anyio
async def test_async_fifo_order_and_waited_flag(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=5.0)
        }
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        holder_ready = anyio.Event()
        release_holder = anyio.Event()
        completed: list[int] = []

        async def hold() -> None:
            async with client.stream("GET", server.url) as response:
                assert response.extensions["origin_waited"] == (False,)
                holder_ready.set()
                await release_holder.wait()

        async def waiter(tag: int) -> None:
            response = await client.get(server.url)
            assert response.extensions["origin_waited"] == (True,)
            completed.append(tag)

        async def two_waiting() -> bool:
            stats = await client.get_origin_stats()
            return key in stats and stats[key].waiting == 2

        async def one_waiting() -> bool:
            stats = await client.get_origin_stats()
            return key in stats and stats[key].waiting == 1

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(hold)
            await holder_ready.wait()
            # Enqueue the waiters one at a time, since task start order is
            # not guaranteed to equal task scheduling order (eg on trio).
            task_group.start_soon(waiter, 1)
            await _await_for(one_waiting)
            task_group.start_soon(waiter, 2)
            await _await_for(two_waiting)
            release_holder.set()

        assert completed == [1, 2]


@pytest.mark.anyio
async def test_async_wait_cancellation_releases_quota(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=30.0)
        }
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        async with client.stream("GET", server.url):

            async def waiter() -> None:
                await client.get(server.url)

            async def one_waiting() -> bool:
                stats = await client.get_origin_stats()
                return key in stats and stats[key].waiting == 1

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(waiter)
                await _await_for(one_waiting)
                task_group.cancel_scope.cancel()

            # Cancelled waiter must not hold a token or remain queued.
            stats = await client.get_origin_stats()
            assert stats[key].in_flight == 1
            assert stats[key].waiting == 0

        # The quota must be fully available (no permanent leak).
        response = await client.get(server.url)
        assert response.status_code == 200
        stats = await client.get_origin_stats()
        assert stats[key].in_flight == 0
        assert stats[key].waiting == 0


@pytest.mark.anyio
async def test_async_origin_wait_timeout_is_independent(server):
    # pool_timeout=None on the origin quota means the waiter ignores the
    # request-level pool timeout and succeeds once the quota is released.
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=1, pool_timeout=None)
        }
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(5.0, pool=0.05),
        trust_env=False,
    ) as client:
        holder_ready = anyio.Event()
        release_holder = anyio.Event()
        result = {}

        async def hold() -> None:
            async with client.stream("GET", server.url):
                holder_ready.set()
                await release_holder.wait()

        async def waiter() -> None:
            response = await client.get(server.url)
            result["status"] = response.status_code
            result["waited"] = response.extensions["origin_waited"][0]

        async def one_waiting() -> bool:
            stats = await client.get_origin_stats()
            return key in stats and stats[key].waiting == 1

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(hold)
            await holder_ready.wait()
            task_group.start_soon(waiter)
            await _await_for(one_waiting)
            # Well beyond the 0.05s global pool timeout, still queued.
            await sleep(0.3)
            stats = await client.get_origin_stats()
            assert stats[key].waiting == 1
            release_holder.set()

        assert result == {"status": 200, "waited": True}


@pytest.mark.anyio
async def test_async_quota_is_never_overissued(server):
    total_requests = 6
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=2, pool_timeout=30.0)
        }
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        peak = {"in_flight": 0, "waiting": 0}
        stop = anyio.Event()

        async def monitor() -> None:
            while not stop.is_set():
                stats = await client.get_origin_stats()
                if key in stats:
                    peak["in_flight"] = max(peak["in_flight"], stats[key].in_flight)
                    peak["waiting"] = max(peak["waiting"], stats[key].waiting)
                await sleep(0.005)

        async def request() -> None:
            response = await client.get(server.url.copy_with(path="/slow_response"))
            assert response.status_code == 200

        async def worker() -> None:
            async with anyio.create_task_group() as task_group:
                for _ in range(total_requests):
                    task_group.start_soon(request)
            stop.set()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(monitor)
            task_group.start_soon(worker)

        assert peak["in_flight"] == 2
        assert peak["waiting"] >= 1

        stats = await client.get_origin_stats()
        assert stats[key].in_flight == 0
        assert stats[key].waiting == 0
        assert 1 <= stats[key].idle <= 2


@pytest.mark.anyio
async def test_async_idle_reserve_eviction(server):
    limits = httpx.Limits(
        per_origin={
            str(server.url): httpx.OriginLimits(
                max_connections=2,
                max_keepalive_connections=1,
                pool_timeout=30.0,
            )
        }
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:

        async def request() -> None:
            response = await client.get(server.url.copy_with(path="/slow_response"))
            assert response.status_code == 200

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(request)
            task_group.start_soon(request)

        stats = await client.get_origin_stats()
        assert stats[key].idle == 1
        assert stats[key].in_flight == 0


@pytest.mark.anyio
async def test_async_ledger_converges_after_keepalive_expiry(server):
    limits = httpx.Limits(
        keepalive_expiry=0.1,
        per_origin={
            str(server.url): httpx.OriginLimits(max_connections=2),
        },
    )
    key = _origin_key(server.url)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        response = await client.get(server.url)
        assert response.status_code == 200
        assert (await client.get_origin_stats())[key].idle == 1

        await sleep(0.5)

        stats = await client.get_origin_stats()
        assert key not in stats


@pytest.mark.anyio
async def test_async_ledger_converges_after_close(server):
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            per_origin={
                str(server.url): httpx.OriginLimits(max_connections=2),
            }
        ),
        trust_env=False,
    ) as client:
        response = await client.get(server.url)
        assert response.status_code == 200

    assert await client.get_origin_stats() == {}
