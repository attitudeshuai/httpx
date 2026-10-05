from __future__ import annotations

import os
import typing

from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes
from ._urls import URL

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

__all__ = ["Limits", "OriginLimits", "Proxy", "Timeout", "create_ssl_context"]


class UnsetType:
    pass  # pragma: no cover


UNSET = UnsetType()

# An origin is identified by the (scheme, host, port) triple. Host is the
# IDNA-encoded ASCII host, as found on `URL.raw_host`.
OriginKey = typing.Tuple[str, str, int]

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}


def origin_key(url: URL) -> OriginKey:
    """
    Return the normalized (scheme, host, port) origin key for a URL, filling
    in the scheme's default port when no explicit port is present - matching
    the origin semantics used by httpcore.
    """
    scheme = url.raw_scheme.decode("ascii")
    port = url.port
    if port is None:
        port = _DEFAULT_PORTS.get(scheme, 0)
    return (scheme, url.raw_host.decode("ascii"), port)


def _validate_int(
    value: typing.Any, name: str, *, allow_zero: bool = False
) -> typing.Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, but got {value!r}.")
    int_value: int = value
    if int_value < 0 or (int_value == 0 and not allow_zero):
        raise ValueError(f"{name} must be greater than zero, but got {value!r}.")
    return int_value


def _validate_timeout(value: typing.Any, name: str) -> typing.Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, but got {value!r}.")
    float_value = float(value)
    if float_value < 0:
        raise ValueError(
            f"{name} must be greater than or equal to zero, got {value!r}."
        )
    return float_value


class OriginLimits:
    """
    Per-origin connection quota configuration.

    Origins are distinguished by their scheme, host and port. A request to an
    origin that has no `OriginLimits` configured follows the global pool
    semantics exactly.

    **Parameters:**

    * **max_connections** - The maximum number of requests that may be in flight
            to the origin at the same time. Further requests wait in an ordered
            per-origin queue. `None` means no per-origin cap.
    * **max_keepalive_connections** - The number of idle connections to retain
            for the origin once in-flight requests complete. `None` means the
            global keepalive limit applies unchanged.
    * **pool_timeout** - The maximum time in seconds to wait for the per-origin
            quota before raising an `OriginPoolTimeout`. This is independent of
            the global pool timeout. `None` means wait indefinitely.
    """

    def __init__(
        self,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        pool_timeout: float | None = 5.0,
    ) -> None:
        self.max_connections = _validate_int(max_connections, "max_connections")
        self.max_keepalive_connections = _validate_int(
            max_keepalive_connections, "max_keepalive_connections", allow_zero=True
        )
        self.pool_timeout = _validate_timeout(pool_timeout, "pool_timeout")

        if (
            self.max_connections is not None
            and self.max_keepalive_connections is not None
            and self.max_keepalive_connections > self.max_connections
        ):
            raise ValueError(
                "max_keepalive_connections must be less than or equal to "
                "max_connections, but got "
                f"max_keepalive_connections={self.max_keepalive_connections} and "
                f"max_connections={self.max_connections}."
            )

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.max_connections == other.max_connections
            and self.max_keepalive_connections == other.max_keepalive_connections
            and self.pool_timeout == other.pool_timeout
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(max_connections={self.max_connections}, "
            f"max_keepalive_connections={self.max_keepalive_connections}, "
            f"pool_timeout={self.pool_timeout})"
        )


def create_ssl_context(
    verify: ssl.SSLContext | str | bool = True,
    cert: CertTypes | None = None,
    trust_env: bool = True,
) -> ssl.SSLContext:
    import ssl
    import warnings

    import certifi

    if verify is True:
        if trust_env and os.environ.get("SSL_CERT_FILE"):  # pragma: nocover
            ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
        elif trust_env and os.environ.get("SSL_CERT_DIR"):  # pragma: nocover
            ctx = ssl.create_default_context(capath=os.environ["SSL_CERT_DIR"])
        else:
            # Default case...
            ctx = ssl.create_default_context(cafile=certifi.where())
    elif verify is False:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif isinstance(verify, str):  # pragma: nocover
        message = (
            "`verify=<str>` is deprecated. "
            "Use `verify=ssl.create_default_context(cafile=...)` "
            "or `verify=ssl.create_default_context(capath=...)` instead."
        )
        warnings.warn(message, DeprecationWarning)
        if os.path.isdir(verify):
            return ssl.create_default_context(capath=verify)
        return ssl.create_default_context(cafile=verify)
    else:
        ctx = verify

    if cert:  # pragma: nocover
        message = (
            "`cert=...` is deprecated. Use `verify=<ssl_context>` instead,"
            "with `.load_cert_chain()` to configure the certificate chain."
        )
        warnings.warn(message, DeprecationWarning)
        if isinstance(cert, str):
            ctx.load_cert_chain(cert)
        else:
            ctx.load_cert_chain(*cert)

    return ctx


class Timeout:
    """
    Timeout configuration.

    **Usage**:

    Timeout(None)               # No timeouts.
    Timeout(5.0)                # 5s timeout on all operations.
    Timeout(None, connect=5.0)  # 5s timeout on connect, no other timeouts.
    Timeout(5.0, connect=10.0)  # 10s timeout on connect. 5s timeout elsewhere.
    Timeout(5.0, pool=None)     # No timeout on acquiring connection from pool.
                                # 5s timeout elsewhere.
    """

    def __init__(
        self,
        timeout: TimeoutTypes | UnsetType = UNSET,
        *,
        connect: None | float | UnsetType = UNSET,
        read: None | float | UnsetType = UNSET,
        write: None | float | UnsetType = UNSET,
        pool: None | float | UnsetType = UNSET,
    ) -> None:
        if isinstance(timeout, Timeout):
            # Passed as a single explicit Timeout.
            assert connect is UNSET
            assert read is UNSET
            assert write is UNSET
            assert pool is UNSET
            self.connect = timeout.connect  # type: typing.Optional[float]
            self.read = timeout.read  # type: typing.Optional[float]
            self.write = timeout.write  # type: typing.Optional[float]
            self.pool = timeout.pool  # type: typing.Optional[float]
        elif isinstance(timeout, tuple):
            # Passed as a tuple.
            self.connect = timeout[0]
            self.read = timeout[1]
            self.write = None if len(timeout) < 3 else timeout[2]
            self.pool = None if len(timeout) < 4 else timeout[3]
        elif not (
            isinstance(connect, UnsetType)
            or isinstance(read, UnsetType)
            or isinstance(write, UnsetType)
            or isinstance(pool, UnsetType)
        ):
            self.connect = connect
            self.read = read
            self.write = write
            self.pool = pool
        else:
            if isinstance(timeout, UnsetType):
                raise ValueError(
                    "httpx.Timeout must either include a default, or set all "
                    "four parameters explicitly."
                )
            self.connect = timeout if isinstance(connect, UnsetType) else connect
            self.read = timeout if isinstance(read, UnsetType) else read
            self.write = timeout if isinstance(write, UnsetType) else write
            self.pool = timeout if isinstance(pool, UnsetType) else pool

    def as_dict(self) -> dict[str, float | None]:
        return {
            "connect": self.connect,
            "read": self.read,
            "write": self.write,
            "pool": self.pool,
        }

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.connect == other.connect
            and self.read == other.read
            and self.write == other.write
            and self.pool == other.pool
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if len({self.connect, self.read, self.write, self.pool}) == 1:
            return f"{class_name}(timeout={self.connect})"
        return (
            f"{class_name}(connect={self.connect}, "
            f"read={self.read}, write={self.write}, pool={self.pool})"
        )


class Limits:
    """
    Configuration for limits to various client behaviors.

    **Parameters:**

    * **max_connections** - The maximum number of concurrent connections that may be
            established.
    * **max_keepalive_connections** - Allow the connection pool to maintain
            keep-alive connections below this point. Should be less than or equal
            to `max_connections`.
    * **keepalive_expiry** - Time limit on idle keep-alive connections in seconds.
    * **per_origin** - A mapping of origins to `OriginLimits` instances, allowing
            in-flight caps and idle-connection reserves to be configured per
            origin. Origins may be given as URLs/strings (eg `"https://example.org"`)
            or as `(scheme, host, port)` triples. Origins without an entry use the
            global limits unchanged.
    """

    def __init__(
        self,
        *,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        keepalive_expiry: float | None = 5.0,
        per_origin: typing.Mapping[str | URL | OriginKey, OriginLimits] | None = None,
    ) -> None:
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.per_origin: dict[OriginKey, OriginLimits] = (
            self._build_per_origin(per_origin) if per_origin is not None else {}
        )

    @staticmethod
    def _build_per_origin(
        per_origin: typing.Mapping[typing.Any, OriginLimits],
    ) -> dict[OriginKey, OriginLimits]:
        if not isinstance(per_origin, typing.Mapping):
            raise ValueError(
                "per_origin must be a mapping of origins to OriginLimits, "
                f"but got {per_origin!r}."
            )

        origin_limits: dict[OriginKey, OriginLimits] = {}
        for key, value in per_origin.items():
            if not isinstance(value, OriginLimits):
                raise ValueError(
                    "per_origin values must be OriginLimits instances, "
                    f"but got {value!r} for origin {key!r}."
                )
            origin_limits[Limits._origin_key(key)] = value
        return origin_limits

    @staticmethod
    def _origin_key(key: str | URL | OriginKey) -> OriginKey:
        if isinstance(key, tuple):
            if (
                len(key) != 3
                or not isinstance(key[0], str)
                or not isinstance(key[1], str)
                or isinstance(key[2], bool)
                or not isinstance(key[2], int)
            ):
                raise ValueError(
                    "per_origin tuple keys must be (scheme, host, port), "
                    f"but got {key!r}."
                )
            try:
                url = URL(f"{key[0]}://{key[1]}:{key[2]}")
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"per_origin origin {key!r} is not a valid origin."
                ) from exc
        elif isinstance(key, (str, URL)):
            url = URL(key)
        else:
            raise ValueError(
                "per_origin keys must be an origin URL/string or a "
                f"(scheme, host, port) tuple, but got {key!r}."
            )

        if not url.scheme or not url.host:
            raise ValueError(
                f"per_origin origin {key!r} must include scheme, host and port."
            )
        if url.scheme not in ("http", "https", "ws", "wss"):
            raise ValueError(
                "per_origin origins must use the http, https, ws or wss scheme, "
                f"but got {key!r}."
            )
        return origin_key(url)

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.max_connections == other.max_connections
            and self.max_keepalive_connections == other.max_keepalive_connections
            and self.keepalive_expiry == other.keepalive_expiry
            and self.per_origin == other.per_origin
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        repr_str = (
            f"{class_name}(max_connections={self.max_connections}, "
            f"max_keepalive_connections={self.max_keepalive_connections}, "
            f"keepalive_expiry={self.keepalive_expiry})"
        )
        if self.per_origin:
            repr_str = f"{repr_str[:-1]}, per_origin={self.per_origin!r})"
        return repr_str


class Proxy:
    def __init__(
        self,
        url: URL | str,
        *,
        ssl_context: ssl.SSLContext | None = None,
        auth: tuple[str, str] | None = None,
        headers: HeaderTypes | None = None,
    ) -> None:
        url = URL(url)
        headers = Headers(headers)

        if url.scheme not in ("http", "https", "socks5", "socks5h"):
            raise ValueError(f"Unknown scheme for proxy URL {url!r}")

        if url.username or url.password:
            # Remove any auth credentials from the URL.
            auth = (url.username, url.password)
            url = url.copy_with(username=None, password=None)

        self.url = url
        self.auth = auth
        self.headers = headers
        self.ssl_context = ssl_context

    @property
    def raw_auth(self) -> tuple[bytes, bytes] | None:
        # The proxy authentication as raw bytes.
        return (
            None
            if self.auth is None
            else (self.auth[0].encode("utf-8"), self.auth[1].encode("utf-8"))
        )

    def __repr__(self) -> str:
        # The authentication is represented with the password component masked.
        auth = (self.auth[0], "********") if self.auth else None

        # Build a nice concise representation.
        url_str = f"{str(self.url)!r}"
        auth_str = f", auth={auth!r}" if auth else ""
        headers_str = f", headers={dict(self.headers)!r}" if self.headers else ""
        return f"Proxy({url_str}{auth_str}{headers_str})"


DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)
DEFAULT_LIMITS = Limits(max_connections=100, max_keepalive_connections=20)
DEFAULT_MAX_REDIRECTS = 20
