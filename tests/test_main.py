import json
import os
import signal
import socket
import threading
import time
import typing

from click.testing import CliRunner

import httpx
from httpx import _main


def splitlines(output: str) -> typing.Iterable[str]:
    return [line.strip() for line in output.splitlines()]


def remove_date_header(lines: typing.Iterable[str]) -> typing.Iterable[str]:
    return [line for line in lines if not line.startswith("date:")]


def test_help():
    runner = CliRunner()
    result = runner.invoke(httpx.main, ["--help"])
    assert result.exit_code == 0
    assert "A next generation HTTP client." in result.output


def test_get(server):
    url = str(server.url)
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: text/plain",
        "Transfer-Encoding: chunked",
        "",
        "Hello, world!",
    ]


def test_json(server):
    url = str(server.url.copy_with(path="/json"))
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: application/json",
        "Transfer-Encoding: chunked",
        "",
        "{",
        '"Hello": "world!"',
        "}",
    ]


def test_binary(server):
    url = str(server.url.copy_with(path="/echo_binary"))
    runner = CliRunner()
    content = "Hello, world!"
    result = runner.invoke(httpx.main, [url, "-c", content])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: application/octet-stream",
        "Transfer-Encoding: chunked",
        "",
        f"<{len(content)} bytes of binary data>",
    ]


def test_redirects(server):
    url = str(server.url.copy_with(path="/redirect_301"))
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url])
    assert result.exit_code == 1
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 301 Moved Permanently",
        "server: uvicorn",
        "location: /",
        "Transfer-Encoding: chunked",
        "",
    ]


def test_follow_redirects(server):
    url = str(server.url.copy_with(path="/redirect_301"))
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, "--follow-redirects"])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 301 Moved Permanently",
        "server: uvicorn",
        "location: /",
        "Transfer-Encoding: chunked",
        "",
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: text/plain",
        "Transfer-Encoding: chunked",
        "",
        "Hello, world!",
    ]


def test_post(server):
    url = str(server.url.copy_with(path="/echo_body"))
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, "-m", "POST", "-j", '{"hello": "world"}'])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: text/plain",
        "Transfer-Encoding: chunked",
        "",
        '{"hello":"world"}',
    ]


def test_verbose(server):
    url = str(server.url)
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, "-v"])
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "* Connecting to '127.0.0.1'",
        "* Connected to '127.0.0.1' on port 8000",
        "GET / HTTP/1.1",
        f"Host: {server.url.netloc.decode('ascii')}",
        "Accept: */*",
        "Accept-Encoding: gzip, deflate, br, zstd",
        "Connection: keep-alive",
        f"User-Agent: python-httpx/{httpx.__version__}",
        "",
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: text/plain",
        "Transfer-Encoding: chunked",
        "",
        "Hello, world!",
    ]


def test_auth(server):
    url = str(server.url)
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, "-v", "--auth", "username", "password"])
    print(result.output)
    assert result.exit_code == 0
    assert remove_date_header(splitlines(result.output)) == [
        "* Connecting to '127.0.0.1'",
        "* Connected to '127.0.0.1' on port 8000",
        "GET / HTTP/1.1",
        f"Host: {server.url.netloc.decode('ascii')}",
        "Accept: */*",
        "Accept-Encoding: gzip, deflate, br, zstd",
        "Connection: keep-alive",
        f"User-Agent: python-httpx/{httpx.__version__}",
        "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
        "",
        "HTTP/1.1 200 OK",
        "server: uvicorn",
        "content-type: text/plain",
        "Transfer-Encoding: chunked",
        "",
        "Hello, world!",
    ]


def test_download(server):
    url = str(server.url)
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(httpx.main, [url, "--download", "index.txt"])
        assert os.path.exists("index.txt")
        with open("index.txt", "r") as input_file:
            assert input_file.read() == "Hello, world!"


def test_errors():
    runner = CliRunner()
    result = runner.invoke(httpx.main, ["invalid://example.org"])
    assert result.exit_code == 1
    assert splitlines(result.output) == [
        "UnsupportedProtocol: Request URL has an unsupported protocol 'invalid://'.",
    ]


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def batch_global_options(**overrides: typing.Any) -> _main.GlobalOptions:
    defaults: dict[str, typing.Any] = dict(
        method=None,
        params=[],
        content=None,
        data=[],
        files=[],
        json=None,
        headers=[],
        cookies=[],
        auth=None,
        proxy=None,
        timeout=5.0,
        follow_redirects=False,
        verify=True,
        http2=False,
        download=None,
        verbose=False,
    )
    defaults.update(overrides)
    return _main.GlobalOptions(**defaults)


def test_batch_multiple_urls_partial_failure(server):
    url_ok = str(server.url)
    url_missing = str(server.url.copy_with(path="/status/404"))
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url_ok, url_missing], catch_exceptions=False)
    assert result.exit_code == 2  # Partial failure.
    # Results are presented in input order regardless of completion order.
    assert f"### [0] GET {url_ok}" in result.output
    assert f"### [1] GET {url_missing}" in result.output
    assert result.output.index("### [0]") < result.output.index("### [1]")
    assert "[FAILED] http_status: 404 Not Found" in result.output
    assert "2 target(s): 1 succeeded, 1 failed" in result.output


def test_batch_all_failed_exit_code_one():
    refused = f"http://127.0.0.1:{unused_port()}/"
    runner = CliRunner()
    result = runner.invoke(httpx.main, [refused, refused], catch_exceptions=False)
    assert result.exit_code == 1  # Every target failed.
    assert "[FAILED] network: ConnectError" in result.output


def test_batch_all_success_exit_code_zero(server):
    url = str(server.url)
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, url], catch_exceptions=False)
    assert result.exit_code == 0
    assert "2 target(s): 2 succeeded, 0 failed" in result.output


def test_batch_results_go_to_stdout_and_progress_to_stderr(server):
    url = str(server.url)
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(httpx.main, [url, url], catch_exceptions=False)
    assert result.exit_code == 0
    assert "### [0] GET" in result.stdout
    assert "Hello, world!" in result.stdout
    assert "succeeded" in result.stderr
    assert "###" not in result.stderr


def test_batch_timeout_category(server, tmp_path):
    url = str(server.url.copy_with(path="/slow_response"))
    runner = CliRunner()
    result = runner.invoke(
        httpx.main,
        [url, "--timeout", "0.01", "--report", str(tmp_path / "report.json")],
        catch_exceptions=False,
    )
    # A batch-only flag triggers batch mode.
    assert result.exit_code == 1
    assert "[FAILED] timeout: ReadTimeout" in result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["results"][0]["error_category"] == "timeout"


def test_batch_invalid_url_reported_before_running():
    runner = CliRunner()
    result = runner.invoke(httpx.main, ["http://example.org", "ftp://example.org"])
    assert result.exit_code == 2
    assert "URL must use the http or https scheme" in result.output
    assert "###" not in result.output  # Nothing was executed.


def test_batch_missing_urls_is_a_usage_error():
    runner = CliRunner()
    result = runner.invoke(httpx.main, [])
    assert result.exit_code == 2
    assert "Missing argument 'URLS...'." in result.output


def test_batch_global_proxy_validated_before_running(server):
    url = str(server.url)
    result = CliRunner().invoke(httpx.main, [url, url, "--proxy", "ftp://proxy:8080"])
    assert result.exit_code == 2
    assert "global --proxy" in result.output
    assert "###" not in result.output


def test_batch_invalid_method_validated_before_running(server):
    url = str(server.url)
    result = CliRunner().invoke(httpx.main, [url, url, "-m", "GET BAD"])
    assert result.exit_code == 2
    assert "'method'" in result.output
    assert "###" not in result.output


def test_batch_concurrency_must_be_positive(server):
    url = str(server.url)
    runner = CliRunner()
    result = runner.invoke(httpx.main, [url, url, "--concurrency", "0"])
    assert result.exit_code == 2
    assert "###" not in result.output


def test_batch_single_download_cannot_target_multiple_urls(tmp_path, server):
    url = str(server.url)
    result = CliRunner().invoke(
        httpx.main, [url, url, "--download", str(tmp_path / "x.txt")]
    )
    assert result.exit_code == 2
    assert "--download" in result.output


def test_batch_shared_client_limits_and_timeout(server, tmp_path, monkeypatch):
    url = str(server.url)
    captured: dict[str, typing.Any] = {}

    class RecordingClient(httpx.Client):
        def __init__(self, **kwargs: typing.Any) -> None:
            captured["max_connections"] = kwargs["limits"].max_connections
            captured["max_keepalive"] = kwargs["limits"].max_keepalive_connections
            captured["timeout"] = kwargs["timeout"]
            super().__init__(**kwargs)

    monkeypatch.setattr(_main, "Client", RecordingClient)
    result = CliRunner().invoke(
        httpx.main,
        [
            url,
            url,
            "--concurrency",
            "3",
            "--timeout",
            "7.5",
            "--report",
            str(tmp_path / "report.json"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert captured["max_connections"] == 3
    assert captured["max_keepalive"] == 3
    assert captured["timeout"] == 7.5


def test_batch_manifest_with_report_and_downloads(server, tmp_path):
    root = str(server.url)
    missing = str(server.url.copy_with(path="/status/500"))
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {"id": "root", "url": root},
                    {"id": "broken", "url": missing, "output": "broken.txt"},
                ]
            }
        )
    )
    report_path = tmp_path / "report.json"

    result = CliRunner().invoke(
        httpx.main,
        [
            "--manifest",
            str(manifest_path),
            "--concurrency",
            "2",
            "--download-dir",
            str(tmp_path / "out"),
            "--report",
            str(report_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "### [0] root GET" in result.output
    assert "### [1] broken GET" in result.output

    # Bodies were saved atomically, one file per target, without leftovers.
    assert (tmp_path / "out" / "root").read_text() == "Hello, world!"
    assert (tmp_path / "out" / "broken.txt").read_text() == "Hello, world!"
    leftovers = [
        p.name for p in (tmp_path / "out").iterdir() if p.name.endswith(".tmp")
    ]
    assert leftovers == []

    report = json.loads(report_path.read_text())
    assert report["total"] == 2
    assert report["succeeded"] == 1
    assert report["failed"] == 1
    assert report["interrupted"] is False
    assert [item["index"] for item in report["results"]] == [0, 1]
    assert report["results"][0]["id"] == "root"
    assert report["results"][0]["success"] is True
    assert report["results"][0]["status"] == 200
    assert report["results"][0]["error_category"] is None
    assert report["results"][1]["success"] is False
    assert report["results"][1]["error_category"] == "http_status"
    assert report["results"][1]["status"] == 500


def test_batch_manifest_accepts_top_level_list(server, tmp_path):
    url = str(server.url)
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(json.dumps([{"url": url}, {"url": url}]))
    result = CliRunner().invoke(
        httpx.main, ["--manifest", str(manifest_path)], catch_exceptions=False
    )
    assert result.exit_code == 0


def test_batch_manifest_invalid_json(tmp_path, server):
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("{not json")
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "not valid JSON" in result.output


def test_batch_manifest_unknown_field(tmp_path, server):
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps({"targets": [{"url": str(server.url), "bogus": 1}]})
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "unknown field" in result.output and "bogus" in result.output


def test_batch_manifest_bad_auth(tmp_path, server):
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps({"targets": [{"url": str(server.url), "auth": ["only-username"]}]})
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "'auth'" in result.output


def test_batch_manifest_bad_proxy(tmp_path, server):
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps({"targets": [{"url": str(server.url), "proxy": "ftp://proxy:8080"}]})
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "proxy URL must use" in result.output


def test_batch_duplicate_outputs_rejected(tmp_path, server):
    url = str(server.url)
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {"url": url, "output": "same.txt"},
                    {"url": url, "output": "same.txt"},
                ]
            }
        )
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "same output file" in result.output


def test_batch_conflicting_proxy_overrides(server, tmp_path):
    url = str(server.url)
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {"url": url, "proxy": "http://a-proxy:8080"},
                    {"url": url, "proxy": "http://b-proxy:8080"},
                ]
            }
        )
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "conflicting proxy overrides" in result.output


def test_batch_target_headers_override_global_per_key(server, tmp_path):
    url = str(server.url.copy_with(path="/echo_headers"))
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "url": url,
                        "headers": [["X-Target", "target"], ["X-Global", "target"]],
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(
        httpx.main,
        [
            "-h",
            "X-Global",
            "global",
            "-h",
            "X-Only-Global",
            "global",
            "--manifest",
            str(manifest_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # The response echoes received headers back as JSON.
    assert '"X-target": "target"' in result.output
    assert '"X-global": "target"' in result.output  # Target value wins.
    assert '"X-only-global": "global"' in result.output  # Global inherited.


def test_batch_target_method_and_auth_override_global(server, tmp_path):
    url = str(server.url.copy_with(path="/echo_body"))
    result = CliRunner().invoke(
        httpx.main,
        ["-m", "GET", "-v", url, "--report", str(tmp_path / "report-a.json")],
        catch_exceptions=False,
    )
    # Global GET is inherited by the single batch target.
    assert result.exit_code == 0
    assert "GET /echo_body HTTP/1.1" in result.output

    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "url": url,
                        "method": "POST",
                        "json": {"hello": "world"},
                        "auth": ["username", "password"],
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(
        httpx.main,
        [
            "--manifest",
            str(manifest_path),
            "-v",
            "--report",
            str(tmp_path / "report-b.json"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "POST /echo_body HTTP/1.1" in result.output
    assert "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=" in result.output
    assert '{"hello":"world"}' in result.output


def test_batch_manifest_file_upload(server, tmp_path):
    upload = tmp_path / "upload.txt"
    upload.write_text("file-content")
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "url": str(server.url.copy_with(path="/echo_body")),
                        "method": "POST",
                        "files": [["upload", str(upload)]],
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(
        httpx.main, ["--manifest", str(manifest_path)], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert "file-content" in result.output


def test_batch_manifest_missing_upload_file(server, tmp_path):
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "url": str(server.url),
                        "files": [["upload", str(tmp_path / "missing.txt")]],
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(httpx.main, ["--manifest", str(manifest_path)])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_batch_failed_download_keeps_existing_file_and_cleans_tmp(server, tmp_path):
    final = tmp_path / "keep.txt"
    final.write_text("original")
    refused_url = f"http://127.0.0.1:{unused_port()}/"
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps({"targets": [{"url": refused_url, "output": str(final)}]})
    )
    result = CliRunner().invoke(
        httpx.main, ["--manifest", str(manifest_path)], catch_exceptions=False
    )
    assert result.exit_code == 1
    # The previous file is untouched and no half-written temp remains.
    assert final.read_text() == "original"
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_batch_concurrent_outputs_do_not_mix(server, tmp_path):
    url = str(server.url)
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {"id": f"target-{i}", "url": url, "output": f"file-{i}.txt"}
                    for i in range(6)
                ]
            }
        )
    )
    result = CliRunner().invoke(
        httpx.main,
        [
            "--manifest",
            str(manifest_path),
            "--concurrency",
            "4",
            "--download-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    for i in range(6):
        assert (tmp_path / f"file-{i}.txt").read_text() == "Hello, world!"


def test_batch_report_on_network_failure(server, tmp_path):
    refused_url = f"http://127.0.0.1:{unused_port()}/"
    report_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        httpx.main,
        [str(server.url), refused_url, "--report", str(report_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    report = json.loads(report_path.read_text())
    categories = {item["error_category"] for item in report["results"]}
    assert categories == {None, "network"}
    failed = report["results"][1]
    assert failed["error_type"] == "ConnectError"
    assert failed["success"] is False
    assert failed["output"] is None


def test_batch_interrupt_drains_inflight_and_skips_the_rest(server, tmp_path):
    slow = str(server.url.copy_with(path="/slow_response"))
    options = batch_global_options(timeout=10.0)
    targets = _main.resolve_targets((slow, slow, slow), None, options, str(tmp_path))
    runner = _main.BatchRunner(
        targets,
        global_options=options,
        concurrency=1,
        report_path=str(tmp_path / "report.json"),
        verbose=False,
    )

    def fire_signal_after_delay() -> None:
        time.sleep(0.2)
        runner.signal_state.signal_name = "SIGINT"
        runner.signal_state.event.set()

    thread = threading.Thread(target=fire_signal_after_delay)
    thread.start()
    exit_code = runner.run()
    thread.join()

    assert exit_code == 130
    # The in-flight target finished and its body was delivered atomically.
    assert (tmp_path / "target-000.body").read_text() == "Hello, world!"
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["interrupted"] is True
    assert report["results"][0]["success"] is True
    assert report["results"][1]["skipped"] is True
    assert report["results"][1]["error_category"] == "interrupted"
    assert report["results"][2]["error_category"] == "interrupted"


def test_batch_signal_handler_maps_to_event(server):
    options = batch_global_options()
    targets = _main.resolve_targets((str(server.url),), None, options, None)
    runner = _main.BatchRunner(
        targets,
        global_options=options,
        concurrency=1,
        report_path=None,
        verbose=False,
    )
    # Invoke the installed callable directly, exactly as the interpreter does
    # when delivering the signal to the main thread.
    assert runner.signal_state.event.is_set() is False
    runner.signal_state(signal.SIGINT, None)
    assert runner.signal_state.event.is_set() is True
    assert runner.signal_state.signal_name == "SIGINT"
