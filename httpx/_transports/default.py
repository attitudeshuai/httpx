"""
Custom transports, with nicely configured defaults.

The following additional keyword arguments are currently supported by httpcore...

* uds: str
* local_address: str
* retries: int

Example usages...

# Disable HTTP/2 on a single specific domain.
mounts = {
    "all://": httpx.HTTPTransport(http2=True),
    "all://*example.org": httpx.HTTPTransport()
}

# Using advanced httpcore configuration, with connection retries.
transport = httpx.HTTPTransport(retries=1)
client = httpx.Client(transport=transport)

# Using advanced httpcore configuration, with unix domain sockets.
transport = httpx.HTTPTransport(uds="socket.uds")
client = httpx.Client(transport=transport)
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import typing
from collections import deque
from types import TracebackType

import anyio

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

    import httpx  # pragma: no cover

from .._config import (
    DEFAULT_LIMITS,
    Limits,
    OriginKey,
    OriginLimits,
    Proxy,
    create_ssl_context,
    origin_key,
)
from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    LocalProtocolError,
    NetworkError,
    OriginPoolTimeout,
    PoolTimeout,
    ProtocolError,
    ProxyError,
    ReadError,
    ReadTimeout,
    RemoteProtocolError,
    TimeoutException,
    UnsupportedProtocol,
    WriteError,
    WriteTimeout,
)
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport

T = typing.TypeVar("T", bound="HTTPTransport")
A = typing.TypeVar("A", bound="AsyncHTTPTransport")

SOCKET_OPTION = typing.Union[
    typing.Tuple[int, int, int],
    typing.Tuple[int, int, typing.Union[bytes, bytearray]],
    typing.Tuple[int, int, None, int],
]

__all__ = ["AsyncHTTPTransport", "HTTPTransport", "OriginPoolStats"]

HTTPCORE_EXC_MAP: dict[type[Exception], type[httpx.HTTPError]] = {}


def _load_httpcore_exceptions() -> dict[type[Exception], type[httpx.HTTPError]]:
    import httpcore

    return {
        httpcore.TimeoutException: TimeoutException,
        httpcore.ConnectTimeout: ConnectTimeout,
        httpcore.ReadTimeout: ReadTimeout,
        httpcore.WriteTimeout: WriteTimeout,
        httpcore.PoolTimeout: PoolTimeout,
        httpcore.NetworkError: NetworkError,
        httpcore.ConnectError: ConnectError,
        httpcore.ReadError: ReadError,
        httpcore.WriteError: WriteError,
        httpcore.ProxyError: ProxyError,
        httpcore.UnsupportedProtocol: UnsupportedProtocol,
        httpcore.ProtocolError: ProtocolError,
        httpcore.LocalProtocolError: LocalProtocolError,
        httpcore.RemoteProtocolError: RemoteProtocolError,
    }


@contextlib.contextmanager
def map_httpcore_exceptions() -> typing.Iterator[None]:
    global HTTPCORE_EXC_MAP
    if len(HTTPCORE_EXC_MAP) == 0:
        HTTPCORE_EXC_MAP = _load_httpcore_exceptions()
    try:
        yield
    except Exception as exc:
        mapped_exc = None

        for from_exc, to_exc in HTTPCORE_EXC_MAP.items():
            if not isinstance(exc, from_exc):
                continue
            # We want to map to the most specific exception we can find.
            # Eg if `exc` is an `httpcore.ReadTimeout`, we want to map to
            # `httpx.ReadTimeout`, not just `httpx.TimeoutException`.
            if mapped_exc is None or issubclass(to_exc, mapped_exc):
                mapped_exc = to_exc

        if mapped_exc is None:  # pragma: no cover
            raise

        message = str(exc)
        raise mapped_exc(message) from exc


@dataclasses.dataclass(frozen=True)
class OriginPoolStats:
    """
    Read-only snapshot of one origin's connection quota accounting.

    * **origin** - The `(scheme, host, port)` triple of the origin.
    * **in_flight** - Requests currently holding a quota token, including any
            request that has not yet fully closed its response stream.
    * **waiting** - Requests queued in the ordered per-origin wait queue.
    * **idle** - Idle connections currently held by the underlying pool for
            the origin.
    * **max_connections** - The configured in-flight cap, or `None` when the
            origin follows the global limits.
    * **max_keepalive_connections** - The configured idle reserve, or `None`.
    """

    origin: OriginKey
    in_flight: int
    waiting: int
    idle: int
    max_connections: int | None
    max_keepalive_connections: int | None


@dataclasses.dataclass
class _OriginState:
    in_flight: int = 0
    waiters: deque[typing.Any] = dataclasses.field(default_factory=deque)


def _origin_display(key: OriginKey) -> str:
    return f"{key[0]}://{key[1]}:{key[2]}"


def _httpcore_origin(key: OriginKey) -> typing.Any:
    import httpcore

    return httpcore.Origin(
        scheme=key[0].encode("ascii"), host=key[1].encode("ascii"), port=key[2]
    )


def _pool_idle_connections(pool: typing.Any, key: OriginKey) -> list[typing.Any] | None:
    """
    Return the origin's idle, reusable connections from the underlying pool,
    or `None` when the pool does not expose its connections (eg proxies).
    """
    get_connections = getattr(pool, "connections", None)
    if get_connections is None:
        return None
    try:
        connections = list(get_connections)
    except Exception:  # pragma: no cover
        return None
    origin = _httpcore_origin(key)
    return [
        connection
        for connection in connections
        if connection.can_handle_request(origin)
        and connection.is_idle()
        and not connection.is_closed()
        and not connection.has_expired()
    ]


class _SyncOriginQuotaGate:
    """
    Thread-safe per-origin in-flight quota gate.

    FIFO waiting with an independent wait timeout, cancellation that never
    holds a token, and self-consistent in-flight/waiting/idle accounting.
    """

    def __init__(self, quotas: typing.Mapping[OriginKey, OriginLimits]) -> None:
        self._quotas = dict(quotas)
        self._lock = threading.Lock()
        self._states: dict[OriginKey, _OriginState] = {}
        # Bumped on close(); waiters from an older generation fail instead of
        # running against a closed pool, while the gate may be reused fresh.
        self._generation = 0

    def wait_timeout(self, key: OriginKey) -> float | None:
        quota = self._quotas.get(key)
        return None if quota is None else quota.pool_timeout

    def acquire(self, key: OriginKey) -> bool:
        """Acquire one quota token. Returns `True` if the request had to wait."""
        with self._lock:
            state = self._states.setdefault(key, _OriginState())
            cap = self._cap(key)
            if cap is None or (state.in_flight < cap and not state.waiters):
                state.in_flight += 1
                return False
            event = threading.Event()
            state.waiters.append(event)
            generation = self._generation

        timeout = self.wait_timeout(key)
        try:
            granted = event.wait(timeout)
        except BaseException:
            # Interrupted while queued: the token must never be held.
            self._abandon_waiter(key, state, event)
            raise

        if not granted:
            self._abandon_waiter(key, state, event)
            raise OriginPoolTimeout(
                self._timeout_message(key, state, cap, timeout), origin=key
            )

        with self._lock:
            if self._generation != generation:
                current = self._states.get(key)
                if current is not None:
                    self._handoff_or_release_locked(current)
                raise PoolTimeout(
                    "The connection pool was closed while waiting for the "
                    f"per-origin quota for {_origin_display(key)}."
                )
        return True

    def release(self, key: OriginKey, pool: typing.Any) -> None:
        # Evict surplus idle connections before waking the next waiter, so a
        # woken request can never be assigned a connection we then evict.
        self._enforce_idle_reserve(key, pool)
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                self._handoff_or_release_locked(state)

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            events = [
                event for state in self._states.values() for event in state.waiters
            ]
            self._states.clear()
        for event in events:
            event.set()

    def snapshot(self, pool: typing.Any) -> dict[OriginKey, OriginPoolStats]:
        with self._lock:
            states = list(self._states)
        idle_counts = {
            key: len(connections)
            for key in states
            if (connections := _pool_idle_connections(pool, key)) is not None
        }

        result: dict[OriginKey, OriginPoolStats] = {}
        with self._lock:
            for key in list(self._states):
                state = self._states[key]
                idle = idle_counts.get(key, 0)
                if state.in_flight == 0 and not state.waiters and idle == 0:
                    # Ledger convergence: nothing in flight, queued or idle.
                    del self._states[key]
                    continue
                quota = self._quotas.get(key)
                result[key] = OriginPoolStats(
                    origin=key,
                    in_flight=state.in_flight,
                    waiting=len(state.waiters),
                    idle=idle,
                    max_connections=None if quota is None else quota.max_connections,
                    max_keepalive_connections=(
                        None if quota is None else quota.max_keepalive_connections
                    ),
                )
        return result

    def _cap(self, key: OriginKey) -> int | None:
        quota = self._quotas.get(key)
        return None if quota is None else quota.max_connections

    def _abandon_waiter(
        self, key: OriginKey, state: _OriginState, event: threading.Event
    ) -> None:
        with self._lock:
            try:
                state.waiters.remove(event)
            except ValueError:
                # A token had already been handed to this waiter. Pass it on.
                self._handoff_or_release_locked(state)

    @staticmethod
    def _handoff_or_release_locked(state: _OriginState) -> None:
        if state.waiters:
            state.waiters.popleft().set()
        else:
            state.in_flight -= 1

    def _timeout_message(
        self,
        key: OriginKey,
        state: _OriginState,
        cap: int | None,
        timeout: float | None,
    ) -> str:
        with self._lock:
            in_flight = state.in_flight
            waiting = len(state.waiters)
        return (
            "Timed out while waiting for the per-origin connection quota "
            f"for {_origin_display(key)} after {timeout} seconds; "
            f"in_flight={in_flight}, waiting={waiting}, max_connections={cap}."
        )

    def _enforce_idle_reserve(self, key: OriginKey, pool: typing.Any) -> None:
        quota = self._quotas.get(key)
        if quota is None or quota.max_keepalive_connections is None:
            return
        reserve = quota.max_keepalive_connections
        idle = _pool_idle_connections(pool, key)
        if idle is None:
            return
        surplus = idle[:-reserve] if reserve else list(idle)
        if not surplus:
            return

        to_close = surplus
        pool_lock = getattr(pool, "_optional_thread_lock", None)
        if pool_lock is not None and hasattr(pool, "_connections"):
            surplus_set = set(surplus)
            with pool_lock:
                # Re-check and detach under the pool's own assignment lock,
                # mirroring the pool's internal eviction bookkeeping.
                to_close = [
                    connection
                    for connection in pool._connections
                    if connection in surplus_set
                    and connection.is_idle()
                    and not connection.is_closed()
                ]
                closing = set(to_close)
                pool._connections = [
                    connection
                    for connection in pool._connections
                    if connection not in closing
                ]

        for connection in to_close:
            with contextlib.suppress(Exception):
                connection.close()


class _AsyncOriginQuotaGate:
    """
    Async (anyio) per-origin in-flight quota gate. See `_SyncOriginQuotaGate`.
    """

    def __init__(self, quotas: typing.Mapping[OriginKey, OriginLimits]) -> None:
        self._quotas = dict(quotas)
        self._lock = anyio.Lock()
        self._states: dict[OriginKey, _OriginState] = {}
        self._generation = 0

    def wait_timeout(self, key: OriginKey) -> float | None:
        quota = self._quotas.get(key)
        return None if quota is None else quota.pool_timeout

    async def acquire(self, key: OriginKey) -> bool:
        """Acquire one quota token. Returns `True` if the request had to wait."""
        async with self._lock:
            state = self._states.setdefault(key, _OriginState())
            cap = self._cap(key)
            if cap is None or (state.in_flight < cap and not state.waiters):
                state.in_flight += 1
                return False
            event = anyio.Event()
            state.waiters.append(event)
            generation = self._generation

        timeout = self.wait_timeout(key)
        try:
            if timeout is None:
                await event.wait()
            else:
                with anyio.fail_after(timeout):
                    await event.wait()
        except TimeoutError:
            await self._abandon_waiter(key, state, event)
            raise OriginPoolTimeout(
                self._timeout_message(key, state, cap, timeout), origin=key
            )
        except BaseException:
            # Cancellation must never hold a token.
            await self._abandon_waiter(key, state, event)
            raise

        async with self._lock:
            if self._generation != generation:
                current = self._states.get(key)
                if current is not None:
                    self._handoff_or_release_locked(current)
                raise PoolTimeout(
                    "The connection pool was closed while waiting for the "
                    f"per-origin quota for {_origin_display(key)}."
                )
        return True

    async def release(self, key: OriginKey, pool: typing.Any) -> None:
        await self._enforce_idle_reserve(key, pool)
        with anyio.CancelScope(shield=True):
            async with self._lock:
                state = self._states.get(key)
                if state is not None:
                    self._handoff_or_release_locked(state)

    async def close(self) -> None:
        async with self._lock:
            self._generation += 1
            events = [
                event for state in self._states.values() for event in state.waiters
            ]
            self._states.clear()
        for event in events:
            event.set()

    async def snapshot(self, pool: typing.Any) -> dict[OriginKey, OriginPoolStats]:
        async with self._lock:
            states = list(self._states)
        idle_counts = {
            key: len(connections)
            for key in states
            if (connections := _pool_idle_connections(pool, key)) is not None
        }

        result: dict[OriginKey, OriginPoolStats] = {}
        async with self._lock:
            for key in list(self._states):
                state = self._states[key]
                idle = idle_counts.get(key, 0)
                if state.in_flight == 0 and not state.waiters and idle == 0:
                    del self._states[key]
                    continue
                quota = self._quotas.get(key)
                result[key] = OriginPoolStats(
                    origin=key,
                    in_flight=state.in_flight,
                    waiting=len(state.waiters),
                    idle=idle,
                    max_connections=None if quota is None else quota.max_connections,
                    max_keepalive_connections=(
                        None if quota is None else quota.max_keepalive_connections
                    ),
                )
        return result

    def _cap(self, key: OriginKey) -> int | None:
        quota = self._quotas.get(key)
        return None if quota is None else quota.max_connections

    async def _abandon_waiter(
        self, key: OriginKey, state: _OriginState, event: anyio.Event
    ) -> None:
        with anyio.CancelScope(shield=True):
            async with self._lock:
                try:
                    state.waiters.remove(event)
                except ValueError:
                    # A token had already been handed to this waiter.
                    self._handoff_or_release_locked(state)

    @staticmethod
    def _handoff_or_release_locked(state: _OriginState) -> None:
        if state.waiters:
            state.waiters.popleft().set()
        else:
            state.in_flight -= 1

    def _timeout_message(
        self,
        key: OriginKey,
        state: _OriginState,
        cap: int | None,
        timeout: float | None,
    ) -> str:
        return (
            "Timed out while waiting for the per-origin connection quota "
            f"for {_origin_display(key)} after {timeout} seconds; "
            f"in_flight={state.in_flight}, waiting={len(state.waiters)}, "
            f"max_connections={cap}."
        )

    async def _enforce_idle_reserve(self, key: OriginKey, pool: typing.Any) -> None:
        quota = self._quotas.get(key)
        if quota is None or quota.max_keepalive_connections is None:
            return
        reserve = quota.max_keepalive_connections
        idle = _pool_idle_connections(pool, key)
        if idle is None:
            return
        surplus = idle[:-reserve] if reserve else list(idle)
        if not surplus:
            return

        to_close = surplus
        pool_lock = getattr(pool, "_optional_thread_lock", None)
        if pool_lock is not None and hasattr(pool, "_connections"):
            surplus_set = set(surplus)
            with pool_lock:
                # No awaits in this block; no other event-loop task can assign
                # a connection while we detach the surplus.
                to_close = [
                    connection
                    for connection in pool._connections
                    if connection in surplus_set
                    and connection.is_idle()
                    and not connection.is_closed()
                ]
                closing = set(to_close)
                pool._connections = [
                    connection
                    for connection in pool._connections
                    if connection not in closing
                ]

        for connection in to_close:
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(Exception):
                    await connection.aclose()


class ResponseStream(SyncByteStream):
    def __init__(self, httpcore_stream: typing.Iterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    def __iter__(self) -> typing.Iterator[bytes]:
        with map_httpcore_exceptions():
            for part in self._httpcore_stream:
                yield part

    def close(self) -> None:
        if hasattr(self._httpcore_stream, "close"):
            self._httpcore_stream.close()


class _QuotaReleaseStream(SyncByteStream):
    """
    Wraps the httpcore response stream so the per-origin quota token is
    released exactly once, when the response is fully closed - even if
    closing the underlying stream raises.
    """

    def __init__(
        self,
        httpcore_stream: typing.Iterable[bytes],
        *,
        gate: _SyncOriginQuotaGate,
        origin: OriginKey,
        pool: typing.Any,
    ) -> None:
        self._httpcore_stream = httpcore_stream
        self._gate = gate
        self._origin = origin
        self._pool = pool
        self._closed = False

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from self._httpcore_stream

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self._httpcore_stream, "close"):
                self._httpcore_stream.close()
        finally:
            self._gate.release(self._origin, self._pool)


class HTTPTransport(BaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        import httpcore

        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
        ssl_context = create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)

        if proxy is None:
            self._pool = httpcore.ConnectionPool(
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                uds=uds,
                local_address=local_address,
                retries=retries,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("http", "https"):
            self._pool = httpcore.HTTPProxy(
                proxy_url=httpcore.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                proxy_headers=proxy.headers.raw,
                ssl_context=ssl_context,
                proxy_ssl_context=proxy.ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                ) from None

            self._pool = httpcore.SOCKSProxy(
                proxy_url=httpcore.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
            )
        else:  # pragma: no cover
            raise ValueError(
                "Proxy protocol must be either 'http', 'https', 'socks5', or 'socks5h',"
                f" but got {proxy.url.scheme!r}."
            )

        self._origin_gate = _SyncOriginQuotaGate(limits.per_origin)

    def __enter__(self: T) -> T:  # Use generics for subclass support.
        self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with map_httpcore_exceptions():
            self._origin_gate.close()
            self._pool.__exit__(exc_type, exc_value, traceback)

    def handle_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, SyncByteStream)
        import httpcore

        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )

        origin = origin_key(request.url)
        waited = self._origin_gate.acquire(origin)
        try:
            with map_httpcore_exceptions():
                resp = self._pool.handle_request(req)
        except BaseException:
            # Never leak a quota token if the pool raised (or was interrupted).
            self._origin_gate.release(origin, self._pool)
            raise

        assert isinstance(resp.stream, typing.Iterable)

        response = Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=ResponseStream(
                _QuotaReleaseStream(
                    resp.stream,
                    gate=self._origin_gate,
                    origin=origin,
                    pool=self._pool,
                )
            ),
            extensions=resp.extensions,
        )
        # Whether this request had to wait for the per-origin quota.
        response.extensions["origin_waited"] = (waited,)
        return response

    def get_origin_stats(self) -> dict[OriginKey, OriginPoolStats]:
        """
        Return a read-only snapshot of the per-origin in-flight, waiting and
        idle connection counts.
        """
        return self._origin_gate.snapshot(self._pool)

    def close(self) -> None:
        self._origin_gate.close()
        self._pool.close()


class AsyncResponseStream(AsyncByteStream):
    def __init__(self, httpcore_stream: typing.AsyncIterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        with map_httpcore_exceptions():
            async for part in self._httpcore_stream:
                yield part

    async def aclose(self) -> None:
        if hasattr(self._httpcore_stream, "aclose"):
            await self._httpcore_stream.aclose()


class _AsyncQuotaReleaseStream(AsyncByteStream):
    """
    Async counterpart of `_QuotaReleaseStream`: releases the per-origin quota
    token exactly once when the response stream is closed.
    """

    def __init__(
        self,
        httpcore_stream: typing.AsyncIterable[bytes],
        *,
        gate: _AsyncOriginQuotaGate,
        origin: OriginKey,
        pool: typing.Any,
    ) -> None:
        self._httpcore_stream = httpcore_stream
        self._gate = gate
        self._origin = origin
        self._pool = pool
        self._closed = False

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for part in self._httpcore_stream:
            yield part

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self._httpcore_stream, "aclose"):
                await self._httpcore_stream.aclose()
        finally:
            await self._gate.release(self._origin, self._pool)


class AsyncHTTPTransport(AsyncBaseTransport):
    def __init__(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        proxy: ProxyTypes | None = None,
        uds: str | None = None,
        local_address: str | None = None,
        retries: int = 0,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> None:
        import httpcore

        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
        ssl_context = create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)

        if proxy is None:
            self._pool = httpcore.AsyncConnectionPool(
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                uds=uds,
                local_address=local_address,
                retries=retries,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("http", "https"):
            self._pool = httpcore.AsyncHTTPProxy(
                proxy_url=httpcore.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                proxy_headers=proxy.headers.raw,
                proxy_ssl_context=proxy.ssl_context,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
                socket_options=socket_options,
            )
        elif proxy.url.scheme in ("socks5", "socks5h"):
            try:
                import socksio  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                ) from None

            self._pool = httpcore.AsyncSOCKSProxy(
                proxy_url=httpcore.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                ssl_context=ssl_context,
                max_connections=limits.max_connections,
                max_keepalive_connections=limits.max_keepalive_connections,
                keepalive_expiry=limits.keepalive_expiry,
                http1=http1,
                http2=http2,
            )
        else:  # pragma: no cover
            raise ValueError(
                "Proxy protocol must be either 'http', 'https', 'socks5', or 'socks5h',"
                f" but got {proxy.url.scheme!r}."
            )

        self._origin_gate = _AsyncOriginQuotaGate(limits.per_origin)

    async def __aenter__(self: A) -> A:  # Use generics for subclass support.
        await self._pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with map_httpcore_exceptions():
            await self._origin_gate.close()
            await self._pool.__aexit__(exc_type, exc_value, traceback)

    async def handle_async_request(
        self,
        request: Request,
    ) -> Response:
        assert isinstance(request.stream, AsyncByteStream)
        import httpcore

        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )

        origin = origin_key(request.url)
        waited = await self._origin_gate.acquire(origin)
        try:
            with map_httpcore_exceptions():
                resp = await self._pool.handle_async_request(req)
        except BaseException:
            # Never leak a quota token if the pool raised (or was cancelled).
            await self._origin_gate.release(origin, self._pool)
            raise

        assert isinstance(resp.stream, typing.AsyncIterable)

        response = Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=AsyncResponseStream(
                _AsyncQuotaReleaseStream(
                    resp.stream,
                    gate=self._origin_gate,
                    origin=origin,
                    pool=self._pool,
                )
            ),
            extensions=resp.extensions,
        )
        # Whether this request had to wait for the per-origin quota.
        response.extensions["origin_waited"] = (waited,)
        return response

    async def get_origin_stats(self) -> dict[OriginKey, OriginPoolStats]:
        """
        Return a read-only snapshot of the per-origin in-flight, waiting and
        idle connection counts.
        """
        return await self._origin_gate.snapshot(self._pool)

    async def aclose(self) -> None:
        await self._origin_gate.close()
        await self._pool.aclose()
