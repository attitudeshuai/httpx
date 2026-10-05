from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import re
import signal
import sys
import tempfile
import threading
import time
import typing
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import click
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table

from ._client import Client
from ._config import Limits, Proxy
from ._exceptions import (
    DecodingError,
    InvalidURL,
    NetworkError,
    ProtocolError,
    ProxyError,
    RequestError,
    TimeoutException,
    TooManyRedirects,
    UnsupportedProtocol,
)
from ._models import Response
from ._status_codes import codes
from ._transports import HTTPTransport
from ._urls import URL

if typing.TYPE_CHECKING:
    import httpcore  # pragma: no cover


# Exit codes used by the batch runner.
EXIT_SUCCESS = 0  # Every target succeeded.
EXIT_ALL_FAILED = 1  # Every target failed (matches single-target behaviour).
EXIT_PARTIAL_FAILURE = 2  # Some targets failed, some succeeded.
EXIT_INTERRUPTED = 130  # 128 + SIGINT; the run was interrupted by a signal.

DEFAULT_CONCURRENCY = 4


def print_help() -> None:
    console = rich.console.Console()

    console.print("[bold]HTTPX :butterfly:", justify="center")
    console.print()
    console.print("A next generation HTTP client.", justify="center")
    console.print()
    console.print(
        "Usage: [bold]httpx[/bold] [cyan]<URL>... [OPTIONS][/cyan] ", justify="left"
    )
    console.print()

    table = rich.table.Table.grid(padding=1, pad_edge=True)
    table.add_column("Parameter", no_wrap=True, justify="left", style="bold")
    table.add_column("Description")
    table.add_row(
        "-m, --method [cyan]METHOD",
        "Request method, such as GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD.\n"
        "[Default: GET, or POST if a request body is included]",
    )
    table.add_row(
        "-p, --params [cyan]<NAME VALUE> ...",
        "Query parameters to include in the request URL.",
    )
    table.add_row(
        "-c, --content [cyan]TEXT", "Byte content to include in the request body."
    )
    table.add_row(
        "-d, --data [cyan]<NAME VALUE> ...", "Form data to include in the request body."
    )
    table.add_row(
        "-f, --files [cyan]<NAME FILENAME> ...",
        "Form files to include in the request body.",
    )
    table.add_row("-j, --json [cyan]TEXT", "JSON data to include in the request body.")
    table.add_row(
        "-h, --headers [cyan]<NAME VALUE> ...",
        "Include additional HTTP headers in the request.",
    )
    table.add_row(
        "--cookies [cyan]<NAME VALUE> ...", "Cookies to include in the request."
    )
    table.add_row(
        "--auth [cyan]<USER PASS>",
        "Username and password to include in the request. Specify '-' for the password"
        " to use a password prompt. Note that using --verbose/-v will expose"
        " the Authorization header, including the password encoding"
        " in a trivially reversible format.",
    )

    table.add_row(
        "--proxy [cyan]URL",
        "Send the request via a proxy. Should be the URL giving the proxy address.",
    )

    table.add_row(
        "--timeout [cyan]FLOAT",
        "Timeout value to use for network operations, such as establishing the"
        " connection, reading some data, etc... [Default: 5.0]",
    )

    table.add_row("--follow-redirects", "Automatically follow redirects.")
    table.add_row("--no-verify", "Disable SSL verification.")
    table.add_row(
        "--http2", "Send the request using HTTP/2, if the remote server supports it."
    )

    table.add_row(
        "--download [cyan]FILE",
        "Save the response content as a file, rather than displaying it.",
    )

    table.add_row("-v, --verbose", "Verbose output. Show request as well as response.")
    table.add_row(
        "--manifest [cyan]FILE",
        "Load a batch of targets from a JSON manifest. May be a JSON list of"
        " target objects, or an object with a 'targets' list. Each target may"
        " set its own 'url', 'id', 'method', 'params', 'headers', 'cookies',"
        " 'content', 'data', 'json', 'files', 'auth', 'proxy', 'timeout',"
        " 'follow_redirects' and 'output'. Target values override the matching"
        " command line options. Invalid values are reported before any request"
        " is sent.",
    )
    table.add_row(
        "--concurrency [cyan]INTEGER",
        f"Maximum number of targets to request in parallel when running a"
        f" batch. All requests share one client, its connection pool and"
        f" timeout configuration. [Default: {DEFAULT_CONCURRENCY}]",
    )
    table.add_row(
        "--download-dir [cyan]DIR",
        "In batch mode, save every response body into DIR, using the target id"
        " or target index as the file name. Per-target 'output' entries take"
        " precedence.",
    )
    table.add_row(
        "--report [cyan]FILE",
        "Write a machine-readable JSON report describing every target and its"
        " result to FILE.",
    )
    table.add_row("--help", "Show this message and exit.")
    console.print(table)


def get_lexer_for_response(response: Response) -> str:
    content_type = response.headers.get("Content-Type")
    if content_type is not None:
        mime_type, _, _ = content_type.partition(";")
        try:
            return typing.cast(
                str, pygments.lexers.get_lexer_for_mimetype(mime_type.strip()).name
            )
        except pygments.util.ClassNotFound:  # pragma: no cover
            pass
    return ""  # pragma: no cover


def format_request_headers(request: httpcore.Request, http2: bool = False) -> str:
    version = "HTTP/2" if http2 else "HTTP/1.1"
    headers = [
        (name.lower() if http2 else name, value) for name, value in request.headers
    ]
    method = request.method.decode("ascii")
    target = request.url.target.decode("ascii")
    lines = [f"{method} {target} {version}"] + [
        f"{name.decode('ascii')}: {value.decode('ascii')}" for name, value in headers
    ]
    return "\n".join(lines)


def format_response_headers(
    http_version: bytes,
    status: int,
    reason_phrase: bytes | None,
    headers: list[tuple[bytes, bytes]],
) -> str:
    version = http_version.decode("ascii")
    reason = (
        codes.get_reason_phrase(status)
        if reason_phrase is None
        else reason_phrase.decode("ascii")
    )
    lines = [f"{version} {status} {reason}"] + [
        f"{name.decode('ascii')}: {value.decode('ascii')}" for name, value in headers
    ]
    return "\n".join(lines)


def print_request_headers(request: httpcore.Request, http2: bool = False) -> None:
    console = rich.console.Console()
    http_text = format_request_headers(request, http2=http2)
    syntax = rich.syntax.Syntax(http_text, "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax("", "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)


def print_response_headers(
    http_version: bytes,
    status: int,
    reason_phrase: bytes | None,
    headers: list[tuple[bytes, bytes]],
) -> None:
    console = rich.console.Console()
    http_text = format_response_headers(http_version, status, reason_phrase, headers)
    syntax = rich.syntax.Syntax(http_text, "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax("", "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)


def get_lexer_for_headers(headers: list[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name.lower() == b"content-type":
            mime_type, _, _ = value.decode("ascii", "ignore").partition(";")
            try:
                return typing.cast(
                    str, pygments.lexers.get_lexer_for_mimetype(mime_type.strip()).name
                )
            except pygments.util.ClassNotFound:  # pragma: no cover
                return ""
    return ""


def render_body_text(headers: list[tuple[bytes, bytes]], content: bytes) -> str | None:
    """
    Return displayable body text, or None if the content is binary.
    An empty body renders as None as well, mirroring single-target output.
    """
    if not content:
        return None
    lexer_name = get_lexer_for_headers(headers)
    if not lexer_name:
        return None
    if lexer_name.lower() == "json":
        try:
            data = json.loads(content)
            return json.dumps(data, indent=4)
        except ValueError:  # pragma: no cover
            pass
    return content.decode("utf-8", errors="replace")


def print_response(response: Response) -> None:
    console = rich.console.Console()
    lexer_name = get_lexer_for_response(response)
    if lexer_name:
        if lexer_name.lower() == "json":
            try:
                data = response.json()
                text = json.dumps(data, indent=4)
            except ValueError:  # pragma: no cover
                text = response.text
        else:
            text = response.text

        syntax = rich.syntax.Syntax(text, lexer_name, theme="ansi_dark", word_wrap=True)
        console.print(syntax)
    else:
        console.print(f"<{len(response.content)} bytes of binary data>")


_PCTRTT = typing.Tuple[typing.Tuple[str, str], ...]
_PCTRTTT = typing.Tuple[_PCTRTT, ...]
_PeerCertRetDictType = typing.Dict[str, typing.Union[str, _PCTRTTT, _PCTRTT]]


def format_certificate(cert: _PeerCertRetDictType) -> str:  # pragma: no cover
    lines = []
    for key, value in cert.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"*   {key}:")
            for item in value:
                if key in ("subject", "issuer"):
                    for sub_item in item:
                        lines.append(f"*     {sub_item[0]}: {sub_item[1]!r}")
                elif isinstance(item, tuple) and len(item) == 2:
                    lines.append(f"*     {item[0]}: {item[1]!r}")
                else:
                    lines.append(f"*     {item!r}")
        else:
            lines.append(f"*   {key}: {value!r}")
    return "\n".join(lines)


def trace(
    name: str, info: typing.Mapping[str, typing.Any], verbose: bool = False
) -> None:
    console = rich.console.Console()
    if name == "connection.connect_tcp.started" and verbose:
        host = info["host"]
        console.print(f"* Connecting to {host!r}")
    elif name == "connection.connect_tcp.complete" and verbose:
        stream = info["return_value"]
        server_addr = stream.get_extra_info("server_addr")
        console.print(f"* Connected to {server_addr[0]!r} on port {server_addr[1]}")
    elif name == "connection.start_tls.complete" and verbose:  # pragma: no cover
        stream = info["return_value"]
        ssl_object = stream.get_extra_info("ssl_object")
        version = ssl_object.version()
        cipher = ssl_object.cipher()
        server_cert = ssl_object.getpeercert()
        alpn = ssl_object.selected_alpn_protocol()
        console.print(f"* SSL established using {version!r} / {cipher[0]!r}")
        console.print(f"* Selected ALPN protocol: {alpn!r}")
        if server_cert:
            console.print("* Server certificate:")
            console.print(format_certificate(server_cert))
    elif name == "http11.send_request_headers.started" and verbose:
        request = info["request"]
        print_request_headers(request, http2=False)
    elif name == "http2.send_request_headers.started" and verbose:  # pragma: no cover
        request = info["request"]
        print_request_headers(request, http2=True)
    elif name == "http11.receive_response_headers.complete":
        http_version, status, reason_phrase, headers = info["return_value"]
        print_response_headers(http_version, status, reason_phrase, headers)
    elif name == "http2.receive_response_headers.complete":  # pragma: no cover
        status, headers = info["return_value"]
        http_version = b"HTTP/2"
        reason_phrase = None
        print_response_headers(http_version, status, reason_phrase, headers)


def download_response(
    response: Response,
    download: typing.BinaryIO,
    download_name: str,
) -> None:
    console = rich.console.Console()
    console.print()
    content_length = response.headers.get("Content-Length")
    with rich.progress.Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.BarColumn(bar_width=None),
        rich.progress.DownloadColumn(),
        rich.progress.TransferSpeedColumn(),
    ) as progress:
        description = f"Downloading [bold]{rich.markup.escape(download_name)}"
        download_task = progress.add_task(
            description,
            total=int(content_length or 0),
            start=content_length is not None,
        )
        for chunk in response.iter_bytes():
            download.write(chunk)
            progress.update(download_task, completed=response.num_bytes_downloaded)


@contextlib.contextmanager
def atomic_open(path: str) -> typing.Iterator[typing.BinaryIO]:
    """
    Open *path* for an atomic download.

    Content is written to a unique temporary file in the same directory and is
    only moved into place with ``os.replace`` once the write completes. If an
    exception is raised the temporary file is removed and any pre-existing
    destination is left untouched. Each call creates its own temporary file, so
    concurrent targets can never write into each other's output.
    """
    final_path = pathlib.Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=str(final_path.parent)
    )
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as download:
            yield download
            download.flush()
            os.fsync(download.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def validate_json(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> typing.Any:
    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover
        raise click.BadParameter("Not valid JSON")


def validate_auth(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> typing.Any:
    if value == (None, None):
        return None

    username, password = value
    if password == "-":  # pragma: no cover
        password = click.prompt("Password", hide_input=True)
    return (username, password)


def handle_help(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> None:
    if not value or ctx.resilient_parsing:
        return

    print_help()
    ctx.exit()


# ---------------------------------------------------------------------------
# Batch support
# ---------------------------------------------------------------------------


class BatchConfigError(ValueError):
    """Raised for invalid command line / manifest configuration."""


@dataclasses.dataclass
class GlobalOptions:
    method: str | None
    params: list[tuple[str, str]]
    content: str | None
    data: list[tuple[str, str]]
    files: list[tuple[str, typing.BinaryIO]]
    json: typing.Any
    headers: list[tuple[str, str]]
    cookies: list[tuple[str, str]]
    auth: tuple[str, str] | None
    proxy: str | None
    timeout: float
    follow_redirects: bool
    verify: bool
    http2: bool
    download: str | None
    verbose: bool


@dataclasses.dataclass
class ResolvedTarget:
    index: int
    id: str | None
    url: str
    method: str
    params: list[tuple[str, str]]
    content: str | None
    data: dict[str, str] | None
    files: list[tuple[str, str]]
    json: typing.Any
    headers: list[tuple[str, str]]
    cookies: list[tuple[str, str]]
    auth: tuple[str, str] | None
    proxy: str | None
    timeout: float | None
    follow_redirects: bool
    output: pathlib.Path | None
    has_body: bool


@dataclasses.dataclass
class TargetResult:
    index: int
    id: str | None
    method: str
    url: str
    success: bool
    status_code: int | None = None
    reason_phrase: str | None = None
    http_version: str | None = None
    headers: list[tuple[bytes, bytes]] | None = None
    content: bytes = b""
    num_bytes: int = 0
    output: str | None = None
    elapsed_ms: float = 0.0
    error_category: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    skipped: bool = False
    request_text: str | None = None


_MANIFEST_TARGET_FIELDS = {
    "id",
    "url",
    "method",
    "params",
    "headers",
    "cookies",
    "content",
    "data",
    "json",
    "files",
    "auth",
    "proxy",
    "timeout",
    "follow_redirects",
    "output",
}
_MANIFEST_TOP_LEVEL_FIELDS = {"targets"}


def _label(index: int, target_id: str | None) -> str:
    if target_id is None:
        return f"target #{index}"
    return f"target #{index} (id={target_id!r})"


def _pairs(value: typing.Any, field: str, label: str) -> list[tuple[str, str]]:
    """Normalise a manifest mapping / pair list into a list of string pairs."""
    if isinstance(value, dict):
        items: typing.Iterable[typing.Any] = value.items()
    elif isinstance(value, list):
        items = value
    else:
        raise BatchConfigError(f"{label}: '{field}' must be a mapping or a list")

    result: list[tuple[str, str]] = []
    for item in items:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
        ):
            raise BatchConfigError(
                f"{label}: '{field}' entries must be [name, value] string pairs"
            )
        result.append((item[0], item[1]))
    return result


def _merge_pairs(
    base: list[tuple[str, str]],
    override: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Merge header/cookie style pairs. Values in *override* take precedence over
    values in *base* on a case-insensitive per-name basis; first-seen ordering
    is preserved.
    """
    merged: list[tuple[str, str]] = []
    index_by_name: dict[str, int] = {}
    for name, value in list(base) + list(override):
        key = name.lower()
        if key in index_by_name:
            merged[index_by_name[key]] = (name, value)
        else:
            index_by_name[key] = len(merged)
            merged.append((name, value))
    return merged


def load_manifest(path: str) -> list[dict[str, typing.Any]]:
    try:
        with open(path, "r", encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
    except OSError as exc:
        raise BatchConfigError(f"Could not read manifest {path!r}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise BatchConfigError(
            f"Manifest {path!r} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from None

    if isinstance(data, list):
        raw_targets: typing.Any = data
    elif isinstance(data, dict) and "targets" in data:
        unknown = set(data) - _MANIFEST_TOP_LEVEL_FIELDS
        if unknown:
            raise BatchConfigError(
                f"Manifest {path!r}: unknown top-level field(s): "
                f"{', '.join(sorted(unknown))}"
            )
        raw_targets = data["targets"]
    else:
        raise BatchConfigError(
            f"Manifest {path!r} must be a list of targets or an object with a"
            f" 'targets' list"
        )

    if not isinstance(raw_targets, list) or not raw_targets:
        raise BatchConfigError(f"Manifest {path!r}: 'targets' must be a non-empty list")

    targets: list[dict[str, typing.Any]] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise BatchConfigError(f"Manifest {path!r}: each target must be an object")
        targets.append(raw)
    return targets


def _resolve_output(
    raw_output: typing.Any,
    download_dir: str | None,
    index: int,
    target_id: str | None,
) -> pathlib.Path:
    if not isinstance(raw_output, str) or not raw_output:
        raise BatchConfigError(f"{_label(index, target_id)}: 'output' must be a path")
    output_path = pathlib.Path(raw_output)
    if not output_path.is_absolute() and download_dir is not None:
        output_path = pathlib.Path(download_dir) / output_path
    return output_path


def _default_output(
    download_dir: str, index: int, target_id: str | None
) -> pathlib.Path:
    if target_id is not None:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in target_id)
        safe = safe.strip("._") or f"target-{index:03d}"
        name = safe
    else:
        name = f"target-{index:03d}.body"
    return pathlib.Path(download_dir) / name


def _validate_proxy(value: typing.Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BatchConfigError(f"{label}: 'proxy' must be a URL")
    try:
        proxy_url = URL(value)
    except InvalidURL as exc:
        raise BatchConfigError(f"{label}: invalid proxy URL {value!r}: {exc}") from None
    if proxy_url.scheme not in ("http", "https", "socks5"):
        raise BatchConfigError(
            f"{label}: proxy URL must use the http, https or socks5 scheme,"
            f" got {proxy_url.scheme!r}"
        )
    return value


def _resolve_target(
    index: int,
    raw: dict[str, typing.Any],
    global_options: GlobalOptions,
    download_dir: str | None,
    *,
    from_manifest: bool,
) -> ResolvedTarget:
    """Resolve one raw target into validated, effective request settings."""
    target_id = raw.get("id", None)
    if target_id is not None and (not isinstance(target_id, str) or not target_id):
        raise BatchConfigError(f"target #{index}: 'id' must be a non-empty string")
    label = _label(index, target_id)

    def field(name: str, default: typing.Any = None) -> typing.Any:
        if from_manifest:
            if name not in raw:
                return default
            return raw[name]
        # Targets synthesised from positional URLs carry every global option.
        return raw.get(name, default)

    if from_manifest:
        unknown = set(raw) - _MANIFEST_TARGET_FIELDS
        if unknown:
            raise BatchConfigError(
                f"{label}: unknown field(s): {', '.join(sorted(unknown))}"
            )

    url = field("url")
    if not isinstance(url, str) or not url:
        raise BatchConfigError(f"{label}: 'url' is required and must be a string")
    try:
        parsed_url = URL(url)
    except InvalidURL as exc:
        raise BatchConfigError(f"{label}: invalid URL {url!r}: {exc}") from None
    if parsed_url.scheme not in ("http", "https"):
        raise BatchConfigError(
            f"{label}: URL must use the http or https scheme, got {parsed_url.scheme!r}"
        )
    if not parsed_url.host:
        raise BatchConfigError(f"{label}: URL {url!r} is missing a host")

    method = field("method", global_options.method)
    if method is not None:
        if (
            not isinstance(method, str)
            or not method.strip()
            or not re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", method)
        ):
            raise BatchConfigError(
                f"{label}: 'method' must be a valid HTTP token, got {method!r}"
            )
    method = method.upper() if method is not None else None

    params = list(global_options.params)
    if "params" in raw:
        params = _merge_pairs(params, _pairs(raw["params"], "params", label))

    headers = list(global_options.headers)
    if "headers" in raw:
        headers = _merge_pairs(headers, _pairs(raw["headers"], "headers", label))

    cookies = list(global_options.cookies)
    if "cookies" in raw:
        cookies = _merge_pairs(cookies, _pairs(raw["cookies"], "cookies", label))

    content = field("content", global_options.content)
    if content is not None and not isinstance(content, str):
        raise BatchConfigError(f"{label}: 'content' must be a string")

    data: dict[str, str] | None = None
    raw_data = field("data", global_options.data)
    if raw_data:
        if isinstance(raw_data, dict):
            data = {str(key): str(value) for key, value in raw_data.items()}
        else:
            data = dict(_pairs(raw_data, "data", label))

    json_value = field("json", global_options.json)

    # File uploads can only be configured per target through the manifest.
    # The paths are validated now and opened lazily by the worker thread.
    files: list[tuple[str, str]] = []
    if from_manifest and "files" in raw:
        raw_files = raw["files"]
        if not isinstance(raw_files, list):
            raise BatchConfigError(f"{label}: 'files' must be a list of pairs")
        for item in raw_files:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
            ):
                raise BatchConfigError(
                    f"{label}: 'files' entries must be [field, path] string pairs"
                )
            field_name, file_path = item
            if not os.path.isfile(file_path):
                raise BatchConfigError(
                    f"{label}: upload file {file_path!r} does not exist"
                )
            files.append((field_name, file_path))

    auth = global_options.auth
    if "auth" in raw and raw["auth"] is not None:
        raw_auth = raw["auth"]
        if (
            not isinstance(raw_auth, (list, tuple))
            or len(raw_auth) != 2
            or not all(isinstance(part, str) for part in raw_auth)
        ):
            raise BatchConfigError(
                f"{label}: 'auth' must be [username, password] strings"
            )
        if raw_auth[1] == "-":
            raise BatchConfigError(
                f"{label}: password prompts are not supported in manifests;"
                f" provide the password directly"
            )
        auth = (raw_auth[0], raw_auth[1])

    proxy = global_options.proxy
    if "proxy" in raw and raw["proxy"] is not None:
        proxy = _validate_proxy(raw["proxy"], label)

    timeout = field("timeout", None)
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise BatchConfigError(f"{label}: 'timeout' must be a number")
        timeout = float(timeout)
        if timeout < 0:
            raise BatchConfigError(f"{label}: 'timeout' must not be negative")

    follow_redirects = global_options.follow_redirects
    if "follow_redirects" in raw:
        if not isinstance(raw["follow_redirects"], bool):
            raise BatchConfigError(f"{label}: 'follow_redirects' must be a boolean")
        follow_redirects = raw["follow_redirects"]

    has_body = bool(content) or bool(data) or bool(files) or json_value is not None
    if method is None:
        method = "POST" if has_body else "GET"

    output: pathlib.Path | None = None
    if "output" in raw and raw["output"] is not None:
        output = _resolve_output(raw["output"], download_dir, index, target_id)
    elif download_dir is not None:
        output = _default_output(download_dir, index, target_id)

    return ResolvedTarget(
        index=index,
        id=target_id,
        url=url,
        method=method,
        params=params,
        content=content,
        data=data,
        files=files,
        json=json_value,
        headers=headers,
        cookies=cookies,
        auth=auth,
        proxy=proxy,
        timeout=timeout,
        follow_redirects=follow_redirects,
        output=output,
        has_body=has_body,
    )


def resolve_targets(
    urls: tuple[str, ...],
    manifest_path: str | None,
    global_options: GlobalOptions,
    download_dir: str | None,
) -> list[ResolvedTarget]:
    if global_options.proxy is not None:
        # The global proxy is shared by every target; validate it alongside
        # the rest of the manifest, before any request is sent.
        _validate_proxy(global_options.proxy, "global --proxy")

    raw_targets: list[tuple[dict[str, typing.Any], bool]] = []
    for url in urls:
        raw_targets.append(({"url": url}, False))
    if manifest_path is not None:
        for raw in load_manifest(manifest_path):
            raw_targets.append((raw, True))

    if not raw_targets:
        raise BatchConfigError("No targets given. Provide a URL or --manifest.")

    resolved = [
        _resolve_target(
            index, raw, global_options, download_dir, from_manifest=from_manifest
        )
        for index, (raw, from_manifest) in enumerate(raw_targets)
    ]

    # Output paths must be unique: two targets replacing the same file would
    # make concurrent delivery ambiguous.
    seen_outputs: dict[str, int] = {}
    for target in resolved:
        if target.output is None:
            continue
        if target.output.is_dir():
            raise BatchConfigError(
                f"{_label(target.index, target.id)}: output path"
                f" {str(target.output)!r} is a directory"
            )
        key = str(target.output.resolve())
        if key in seen_outputs:
            raise BatchConfigError(
                f"{_label(target.index, target.id)} and target #{seen_outputs[key]}"
                f" share the same output file {key!r}"
            )
        seen_outputs[key] = target.index

    # Proxies are mounted per scheme/host/port. Two targets whose origin matches
    # but which use different proxies would be ambiguous, so reject that up front.
    seen_proxies: dict[str, tuple[str, str]] = {}
    for target in resolved:
        if target.proxy is None:
            continue
        origin = (
            f"{URL(target.url).scheme}://{URL(target.url).host}:{URL(target.url).port}"
        )
        previous = seen_proxies.get(origin)
        if previous is not None and previous[0] != target.proxy:
            raise BatchConfigError(
                f"{_label(target.index, target.id)}: conflicting proxy overrides"
                f" for origin {origin!r} ({target.proxy!r} vs {previous[0]!r})"
            )
        seen_proxies[origin] = (target.proxy, target.url)

    return resolved


def build_proxy_mounts(
    targets: typing.Iterable[ResolvedTarget],
    *,
    global_proxy: str | None,
    verify: bool,
    http2: bool,
    limits: Limits,
) -> dict[str, HTTPTransport]:
    """
    One mount per origin that has a target-level proxy override. Targets that
    simply inherit the global proxy rely on the client's own proxy setting.
    """
    mounts: dict[str, HTTPTransport] = {}
    for target in targets:
        if target.proxy is None or target.proxy == global_proxy:
            continue
        target_url = URL(target.url)
        key = f"{target_url.scheme}://{target_url.host}:{target_url.port}"
        if key in mounts:
            continue
        mounts[key] = HTTPTransport(
            verify=verify,
            http2=http2,
            limits=limits,
            proxy=Proxy(url=target.proxy),
        )
    return mounts


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Map an exception onto a stable, machine-readable failure category."""
    if isinstance(exc, UnsupportedProtocol):
        category = "unsupported_protocol"
    elif isinstance(exc, TimeoutException):
        category = "timeout"
    elif isinstance(exc, ProxyError):
        category = "proxy"
    elif isinstance(exc, ProtocolError):
        category = "protocol"
    elif isinstance(exc, NetworkError):
        category = "network"
    elif isinstance(exc, TooManyRedirects):
        category = "too_many_redirects"
    elif isinstance(exc, DecodingError):
        category = "decoding"
    elif isinstance(exc, InvalidURL):
        category = "invalid_url"
    elif isinstance(exc, RequestError):
        category = "request_error"
    else:
        category = "error"
    return category, type(exc).__name__


class _SignalState:
    """First signal drains in-flight work; a second signal exits immediately."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.signal_name: str | None = None

    def __call__(self, signum: int, frame: typing.Any) -> None:
        if self.event.is_set():
            os._exit(EXIT_INTERRUPTED)
        self.signal_name = signal.Signals(signum).name
        self.event.set()


def _format_request_text(
    method: str, target: str, headers: list[tuple[bytes, bytes]]
) -> str:
    lines = [f"{method} {target} HTTP/1.1"] + [
        f"{name.decode('ascii', 'ignore')}: {value.decode('ascii', 'ignore')}"
        for name, value in headers
    ]
    return "\n".join(lines)


class BatchRunner:
    def __init__(
        self,
        targets: list[ResolvedTarget],
        *,
        global_options: GlobalOptions,
        concurrency: int,
        report_path: str | None,
        verbose: bool,
    ) -> None:
        self.targets = targets
        self.global_options = global_options
        self.concurrency = concurrency
        self.report_path = report_path
        self.verbose = verbose
        self.signal_state = _SignalState()
        self.temp_files: set[pathlib.Path] = set()
        self.temp_lock = threading.Lock()
        self.results: list[TargetResult | None] = [None] * len(targets)

    # -- worker -------------------------------------------------------------

    def _run_target(self, target: ResolvedTarget) -> TargetResult:
        start = time.perf_counter()

        def finish(**kwargs: typing.Any) -> TargetResult:
            kwargs.setdefault("elapsed_ms", (time.perf_counter() - start) * 1000.0)
            result = TargetResult(
                index=target.index,
                id=target.id,
                method=target.method,
                url=target.url,
                **kwargs,
            )
            self.results[target.index] = result
            return result

        opened_files: list[typing.IO[bytes]] = []
        try:
            files: list[tuple[str, typing.IO[bytes]]] = []
            for field_name, file_path in target.files:
                file_handle = open(file_path, "rb")
                opened_files.append(file_handle)
                files.append((field_name, file_handle))

            request_kwargs: dict[str, typing.Any] = dict(
                params=target.params or None,
                content=target.content,
                data=target.data,
                files=files or None,
                json=target.json,
                headers=target.headers or None,
                cookies=dict(target.cookies) if target.cookies else None,
            )
            if target.timeout is not None:
                request_kwargs["timeout"] = target.timeout

            client = self.client
            request = client.build_request(target.method, target.url, **request_kwargs)
            response = client.send(
                request,
                stream=True,
                auth=target.auth,
                follow_redirects=target.follow_redirects,
            )
            with contextlib.closing(response):
                headers_raw = list(response.headers.raw)
                status = response.status_code
                reason = response.reason_phrase
                http_version = response.extensions.get("http_version", b"HTTP/1.1")
                version_text = (
                    http_version.decode("ascii", "ignore")
                    if isinstance(http_version, bytes)
                    else "HTTP/1.1"
                )
                request_text = (
                    _format_request_text(
                        target.method,
                        request.url.raw_path.decode("ascii", "ignore"),
                        list(request.headers.raw),
                    )
                    if self.verbose
                    else None
                )

                output_name: str | None = None
                content = b""
                if target.output is not None:
                    output_name = str(target.output)
                    with self._atomic_registered(target.output) as download:
                        for chunk in response.iter_bytes():
                            download.write(chunk)
                else:
                    response.read()
                    content = response.content

            return finish(
                success=response.is_success,
                status_code=status,
                reason_phrase=reason,
                http_version=version_text,
                headers=headers_raw,
                content=content,
                num_bytes=(
                    len(content)
                    if output_name is None
                    else response.num_bytes_downloaded
                ),
                output=output_name,
                request_text=request_text,
                error_category=None if response.is_success else "http_status",
                error_type=None if response.is_success else f"HTTP {status}",
                error_message=(None if response.is_success else f"{status} {reason}"),
            )
        except (RequestError, InvalidURL, ValueError) as exc:
            category, error_type = (
                classify_exception(exc)
                if isinstance(exc, (RequestError, InvalidURL))
                else ("request_error", type(exc).__name__)
            )
            return finish(
                success=False,
                error_category=category,
                error_type=error_type,
                error_message=str(exc) or error_type,
            )
        except OSError as exc:
            # Filesystem failure while opening an upload or writing a download.
            return finish(
                success=False,
                error_category="file_error",
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )
        finally:
            for open_file in opened_files:
                with contextlib.suppress(OSError):
                    open_file.close()

    @contextlib.contextmanager
    def _atomic_registered(
        self, final_path: pathlib.Path
    ) -> typing.Iterator[typing.BinaryIO]:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=str(final_path.parent),
        )
        tmp_path = pathlib.Path(tmp_name)
        with self.temp_lock:
            self.temp_files.add(tmp_path)
        try:
            with os.fdopen(fd, "wb") as download:
                yield download
                download.flush()
                os.fsync(download.fileno())
            os.replace(tmp_path, final_path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
        finally:
            with self.temp_lock:
                self.temp_files.discard(tmp_path)

    # -- scheduling ---------------------------------------------------------

    def run(self) -> int:
        limits = Limits(
            max_connections=self.concurrency,
            max_keepalive_connections=min(self.concurrency, 20),
        )
        mounts = build_proxy_mounts(
            self.targets,
            global_proxy=self.global_options.proxy,
            verify=self.global_options.verify,
            http2=self.global_options.http2,
            limits=limits,
        )

        previous_handlers = self._install_signal_handlers()
        interrupted = False
        stderr_console = rich.console.Console(stderr=True)
        try:
            with (
                Client(
                    proxy=self.global_options.proxy,
                    timeout=self.global_options.timeout,
                    http2=self.global_options.http2,
                    verify=self.global_options.verify,
                    limits=limits,
                    mounts=mounts,
                ) as self.client,
                ThreadPoolExecutor(max_workers=self.concurrency) as executor,
            ):
                progress = rich.progress.Progress(
                    "[progress.description]{task.description}",
                    rich.progress.BarColumn(bar_width=None),
                    rich.progress.DownloadColumn(),
                    rich.progress.TransferSpeedColumn(),
                    console=stderr_console,
                    transient=True,
                )
                with progress:
                    self._schedule(executor, progress)
        finally:
            self._restore_signal_handlers(previous_handlers)
            self._cleanup_temp_files()
            interrupted = self.signal_state.event.is_set()

        results = typing.cast(list[TargetResult], self.results)
        self._print_results(results, stderr_console, interrupted=interrupted)
        self._write_report(results, interrupted=interrupted)

        if interrupted:
            return EXIT_INTERRUPTED
        failed = [result for result in results if not result.success]
        if not failed:
            return EXIT_SUCCESS
        if len(failed) == len(results):
            return EXIT_ALL_FAILED
        return EXIT_PARTIAL_FAILURE

    def _schedule(
        self, executor: ThreadPoolExecutor, progress: rich.progress.Progress
    ) -> None:
        next_index = 0
        inflight: set[Future[TargetResult]] = set()
        future_targets: dict[Future[TargetResult], ResolvedTarget] = {}
        total = len(self.targets)

        def submit_next() -> None:
            nonlocal next_index
            target = self.targets[next_index]
            next_index += 1
            host = URL(target.url).host or ""
            task_id = progress.add_task(
                f"#{target.index} {target.method} {host}", total=None
            )
            future = executor.submit(self._run_target, target)
            future.add_done_callback(
                lambda _: progress.update(task_id, total=1, completed=1)
            )
            inflight.add(future)
            future_targets[future] = target

        # Bounded in-flight set. The first wave is always dispatched so that
        # a signal landing while the client is still warming up still has
        # in-flight work to drain deterministically; after that, a signal
        # prevents any further targets from being handed to workers.
        while next_index < total or inflight:
            while next_index < total and len(inflight) < self.concurrency:
                if next_index > 0 and self.signal_state.event.is_set():
                    break
                submit_next()
            if not inflight:
                break
            done, _ = wait(inflight, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                inflight.discard(future)
                target = future_targets.pop(future)
                # _run_target handles request failures itself; this only covers
                # truly unexpected worker crashes.
                if (exc := future.exception()) is not None:
                    category, error_type = classify_exception(exc)
                    self.results[target.index] = TargetResult(
                        index=target.index,
                        id=target.id,
                        method=target.method,
                        url=target.url,
                        success=False,
                        error_category=category,
                        error_type=error_type,
                        error_message=str(exc) or error_type,
                    )

        # Targets never handed to a worker once interrupted.
        for target in self.targets[next_index:]:
            self.results[target.index] = TargetResult(
                index=target.index,
                id=target.id,
                method=target.method,
                url=target.url,
                success=False,
                skipped=True,
                error_category="interrupted",
                error_type="Interrupted",
                error_message="Run interrupted before the request was sent",
            )

    # -- signals / cleanup --------------------------------------------------

    def _install_signal_handlers(
        self,
    ) -> list[tuple[int, typing.Any]]:
        previous: list[tuple[int, typing.Any]] = []
        for signame in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                previous.append((sig, signal.getsignal(sig)))
                signal.signal(sig, self.signal_state)
            except (ValueError, OSError):  # pragma: no cover - not main thread
                pass
        return previous

    def _restore_signal_handlers(self, previous: list[tuple[int, typing.Any]]) -> None:
        for sig, handler in previous:
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover
                signal.signal(sig, handler)

    def _cleanup_temp_files(self) -> None:
        with self.temp_lock:
            temp_files = list(self.temp_files)
            self.temp_files.clear()
        for tmp_path in temp_files:
            with contextlib.suppress(OSError):
                tmp_path.unlink()

    # -- output -------------------------------------------------------------

    def _target_heading(self, result: TargetResult) -> str:
        identifier = f" {result.id}" if result.id is not None else ""
        return f"### [{result.index}]{identifier} {result.method} {result.url}"

    def _print_results(
        self,
        results: list[TargetResult],
        stderr_console: rich.console.Console,
        *,
        interrupted: bool,
    ) -> None:
        console = rich.console.Console()
        for result in results:
            console.print(self._target_heading(result))
            if result.request_text is not None:
                console.print(
                    rich.syntax.Syntax(
                        result.request_text, "http", theme="ansi_dark", word_wrap=True
                    )
                )
            if result.status_code is None:
                status = "SKIPPED" if result.skipped else "FAILED"
                console.print(
                    f"[{status}] {result.error_category}: {result.error_type}:"
                    f" {result.error_message}",
                    style="red",
                )
            else:
                reason = result.reason_phrase or codes.get_reason_phrase(
                    result.status_code
                )
                header_text = format_response_headers(
                    result.http_version.encode("ascii")
                    if result.http_version
                    else b"HTTP/1.1",
                    result.status_code,
                    reason.encode("ascii", "ignore"),
                    result.headers or [],
                )
                console.print(
                    rich.syntax.Syntax(
                        header_text, "http", theme="ansi_dark", word_wrap=True
                    )
                )
                if result.output is not None:
                    console.print(
                        f"Saved response body to {result.output}"
                        f" ({result.num_bytes} bytes)"
                    )
                else:
                    body_text = render_body_text(result.headers or [], result.content)
                    if body_text is not None:
                        lexer_name = (
                            get_lexer_for_headers(result.headers or []) or "text"
                        )
                        console.print(
                            rich.syntax.Syntax(
                                body_text,
                                lexer_name,
                                theme="ansi_dark",
                                word_wrap=True,
                            )
                        )
                    elif result.num_bytes:
                        console.print(f"<{result.num_bytes} bytes of binary data>")
                if not result.success:
                    console.print(
                        f"[FAILED] {result.error_category}: {result.error_message}",
                        style="red",
                    )
            console.print()

        succeeded = sum(1 for result in results if result.success)
        failed = len(results) - succeeded
        summary = f"{len(results)} target(s): {succeeded} succeeded, {failed} failed"
        if interrupted:
            summary += f"; interrupted by {self.signal_state.signal_name or 'signal'}"
        stderr_console.print(summary)

    def _write_report(self, results: list[TargetResult], *, interrupted: bool) -> None:
        if self.report_path is None:
            return
        payload = {
            "interrupted": interrupted,
            "signal": self.signal_state.signal_name,
            "total": len(results),
            "succeeded": sum(1 for result in results if result.success),
            "failed": sum(1 for result in results if not result.success),
            "results": [
                {
                    "index": result.index,
                    "id": result.id,
                    "method": result.method,
                    "url": result.url,
                    "success": result.success,
                    "skipped": result.skipped,
                    "status": result.status_code,
                    "reason": result.reason_phrase,
                    "http_version": result.http_version,
                    "headers": [
                        [
                            name.decode("ascii", "ignore"),
                            value.decode("ascii", "ignore"),
                        ]
                        for name, value in (result.headers or [])
                    ],
                    "bytes": result.num_bytes,
                    "output": result.output,
                    "elapsed_ms": round(result.elapsed_ms, 3),
                    "error_category": result.error_category,
                    "error_type": result.error_type,
                    "error": result.error_message,
                }
                for result in results
            ],
        }
        try:
            with atomic_open(self.report_path) as report_file:
                report_file.write(
                    json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
                )
        except OSError as exc:
            # Delivered results stay on stdout even if the report cannot be
            # written; surface the problem on the progress/error channel.
            rich.console.Console(stderr=True).print(
                f"[red]Could not write report to {self.report_path!r}: {exc}[/red]"
            )


def _legacy_single_request(
    *,
    url: str,
    global_options: GlobalOptions,
    download: str | None,
) -> None:
    """The original single-target code path, preserved byte for byte."""
    # Local aliases so the request body below is identical to the historical
    # single-target implementation, including its output formatting.
    params = global_options.params
    content = global_options.content
    data = global_options.data
    files = global_options.files
    json = global_options.json
    headers = global_options.headers
    cookies = global_options.cookies
    auth = global_options.auth
    proxy = global_options.proxy
    timeout = global_options.timeout
    follow_redirects = global_options.follow_redirects
    verify = global_options.verify
    http2 = global_options.http2
    verbose = global_options.verbose

    method = global_options.method
    if not method:
        method = "POST" if content or data or files or json else "GET"

    try:
        with Client(proxy=proxy, timeout=timeout, http2=http2, verify=verify) as client:
            with client.stream(
                method,
                url,
                params=list(params),
                content=content,
                data=dict(data),
                files=files,
                json=json,
                headers=headers,
                cookies=dict(cookies),
                auth=auth,
                follow_redirects=follow_redirects,
                extensions={"trace": functools.partial(trace, verbose=verbose)},
            ) as response:
                if download is not None:
                    with atomic_open(download) as download_file:
                        download_response(response, download_file, download)
                else:
                    response.read()
                    if response.content:
                        print_response(response)

    except RequestError as exc:
        console = rich.console.Console()
        console.print(f"[red]{type(exc).__name__}[/red]: {exc}")
        sys.exit(1)

    sys.exit(0 if response.is_success else 1)


@click.command(add_help_option=False)
@click.argument("urls", nargs=-1, required=False, type=str)
@click.option(
    "--method",
    "-m",
    "method",
    type=str,
    help=(
        "Request method, such as GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD. "
        "[Default: GET, or POST if a request body is included]"
    ),
)
@click.option(
    "--params",
    "-p",
    "params",
    type=(str, str),
    multiple=True,
    help="Query parameters to include in the request URL.",
)
@click.option(
    "--content",
    "-c",
    "content",
    type=str,
    help="Byte content to include in the request body.",
)
@click.option(
    "--data",
    "-d",
    "data",
    type=(str, str),
    multiple=True,
    help="Form data to include in the request body.",
)
@click.option(
    "--files",
    "-f",
    "files",
    type=(str, click.File(mode="rb")),
    multiple=True,
    help="Form files to include in the request body.",
)
@click.option(
    "--json",
    "-j",
    "json",
    type=str,
    callback=validate_json,
    help="JSON data to include in the request body.",
)
@click.option(
    "--headers",
    "-h",
    "headers",
    type=(str, str),
    multiple=True,
    help="Include additional HTTP headers in the request.",
)
@click.option(
    "--cookies",
    "cookies",
    type=(str, str),
    multiple=True,
    help="Cookies to include in the request.",
)
@click.option(
    "--auth",
    "auth",
    type=(str, str),
    default=(None, None),
    callback=validate_auth,
    help=(
        "Username and password to include in the request. "
        "Specify '-' for the password to use a password prompt. "
        "Note that using --verbose/-v will expose the Authorization header, "
        "including the password encoding in a trivially reversible format."
    ),
)
@click.option(
    "--proxy",
    "proxy",
    type=str,
    default=None,
    help="Send the request via a proxy. Should be the URL giving the proxy address.",
)
@click.option(
    "--timeout",
    "timeout",
    type=float,
    default=5.0,
    help=(
        "Timeout value to use for network operations, such as establishing the "
        "connection, reading some data, etc... [Default: 5.0]"
    ),
)
@click.option(
    "--follow-redirects",
    "follow_redirects",
    is_flag=True,
    default=False,
    help="Automatically follow redirects.",
)
@click.option(
    "--no-verify",
    "verify",
    is_flag=True,
    default=True,
    help="Disable SSL verification.",
)
@click.option(
    "--http2",
    "http2",
    type=bool,
    is_flag=True,
    default=False,
    help="Send the request using HTTP/2, if the remote server supports it.",
)
@click.option(
    "--download",
    "download",
    type=str,
    default=None,
    help="Save the response content as a file, rather than displaying it.",
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    type=bool,
    is_flag=True,
    default=False,
    help="Verbose. Show request as well as response.",
)
@click.option(
    "--manifest",
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Load a batch of targets from a JSON manifest file.",
)
@click.option(
    "--concurrency",
    "concurrency",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Maximum number of parallel requests in batch mode. "
        f"[Default: {DEFAULT_CONCURRENCY}]"
    ),
)
@click.option(
    "--download-dir",
    "download_dir",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Directory to save batch response bodies into.",
)
@click.option(
    "--report",
    "report",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write a machine-readable JSON report for a batch to this file.",
)
@click.option(
    "--help",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=handle_help,
    help="Show this message and exit.",
)
def main(
    urls: tuple[str, ...],
    method: str | None,
    params: tuple[tuple[str, str], ...],
    content: str | None,
    data: tuple[tuple[str, str], ...],
    files: tuple[tuple[str, typing.BinaryIO], ...],
    json: typing.Any,
    headers: tuple[tuple[str, str], ...],
    cookies: tuple[tuple[str, str], ...],
    auth: tuple[str, str] | None,
    proxy: str | None,
    timeout: float,
    follow_redirects: bool,
    verify: bool,
    http2: bool,
    download: str | None,
    verbose: bool,
    manifest: str | None,
    concurrency: int | None,
    download_dir: str | None,
    report: str | None,
) -> None:
    """
    An HTTP command line client.

    With a single URL it sends one request and displays the response. With
    multiple URLs (or a --manifest file) it runs the targets as a batch,
    sharing one client with a configurable concurrency limit, preserving input
    ordering for results, and exiting 0 for all success, 1 when every target
    failed and 2 for a partial failure.
    """
    if not urls and not manifest:
        raise click.UsageError("Missing argument 'URLS...'.")

    global_options = GlobalOptions(
        method=method,
        params=list(params),
        content=content,
        data=list(data),
        files=list(files),
        json=json,
        headers=list(headers),
        cookies=list(cookies),
        auth=auth,
        proxy=proxy,
        timeout=timeout,
        follow_redirects=follow_redirects,
        verify=verify,
        http2=http2,
        download=download,
        verbose=verbose,
    )

    batch_mode = (
        bool(manifest)
        or len(urls) > 1
        or concurrency is not None
        or (download_dir is not None or report is not None)
    )

    if not batch_mode:
        _legacy_single_request(
            url=urls[0],
            global_options=global_options,
            download=download,
        )
        return  # pragma: no cover - _legacy_single_request always exits

    # Batch-only configuration checks, reported before any request is sent.
    if download is not None and (len(urls) > 1 or manifest is not None):
        raise click.UsageError(
            "--download names a single file and can only be used with one URL."
            " Use a manifest 'output' per target, or --download-dir."
        )
    if files:
        raise click.UsageError(
            "--files can only be used with a single URL, or configured per"
            " target via a manifest 'files' entry."
        )

    try:
        targets = resolve_targets(urls, manifest, global_options, download_dir)
    except BatchConfigError as exc:
        raise click.UsageError(str(exc)) from None

    # A single positional URL in batch mode (e.g. with --report) may still use
    # --download to name its output file.
    if download is not None and len(targets) == 1 and targets[0].output is None:
        targets[0].output = pathlib.Path(download)

    runner = BatchRunner(
        targets,
        global_options=global_options,
        concurrency=concurrency or DEFAULT_CONCURRENCY,
        report_path=report,
        verbose=verbose,
    )
    sys.exit(runner.run())
