import pytest

import httpx


def app(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/redirect":
        return httpx.Response(303, headers={"server": "testserver", "location": "/"})
    elif request.url.path.startswith("/status/"):
        status_code = int(request.url.path[-3:])
        return httpx.Response(status_code, headers={"server": "testserver"})

    return httpx.Response(200, headers={"server": "testserver"})


def test_event_hooks():
    events = []

    def on_request(request):
        events.append({"event": "request", "headers": dict(request.headers)})

    def on_response(response):
        events.append({"event": "response", "headers": dict(response.headers)})

    event_hooks = {"request": [on_request], "response": [on_response]}

    with httpx.Client(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        http.get("http://127.0.0.1:8000/", auth=("username", "password"))

    assert events == [
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"server": "testserver"},
        },
    ]


def test_event_hooks_raising_exception(server):
    def raise_on_4xx_5xx(response):
        response.raise_for_status()

    event_hooks = {"response": [raise_on_4xx_5xx]}

    with httpx.Client(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        try:
            http.get("http://127.0.0.1:8000/status/400")
        except httpx.HTTPStatusError as exc:
            assert exc.response.is_closed


@pytest.mark.anyio
async def test_async_event_hooks():
    events = []

    async def on_request(request):
        events.append({"event": "request", "headers": dict(request.headers)})

    async def on_response(response):
        events.append({"event": "response", "headers": dict(response.headers)})

    event_hooks = {"request": [on_request], "response": [on_response]}

    async with httpx.AsyncClient(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        await http.get("http://127.0.0.1:8000/", auth=("username", "password"))

    assert events == [
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"server": "testserver"},
        },
    ]


@pytest.mark.anyio
async def test_async_event_hooks_raising_exception():
    async def raise_on_4xx_5xx(response):
        response.raise_for_status()

    event_hooks = {"response": [raise_on_4xx_5xx]}

    async with httpx.AsyncClient(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        try:
            await http.get("http://127.0.0.1:8000/status/400")
        except httpx.HTTPStatusError as exc:
            assert exc.response.is_closed


def test_event_hooks_with_redirect():
    """
    A redirect request should trigger additional 'request' and 'response' event hooks.
    """

    events = []

    def on_request(request):
        events.append({"event": "request", "headers": dict(request.headers)})

    def on_response(response):
        events.append({"event": "response", "headers": dict(response.headers)})

    event_hooks = {"request": [on_request], "response": [on_response]}

    with httpx.Client(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(app),
        follow_redirects=True,
    ) as http:
        http.get("http://127.0.0.1:8000/redirect", auth=("username", "password"))

    assert events == [
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"location": "/", "server": "testserver"},
        },
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"server": "testserver"},
        },
    ]


@pytest.mark.anyio
async def test_async_event_hooks_with_redirect():
    """
    A redirect request should trigger additional 'request' and 'response' event hooks.
    """

    events = []

    async def on_request(request):
        events.append({"event": "request", "headers": dict(request.headers)})

    async def on_response(response):
        events.append({"event": "response", "headers": dict(response.headers)})

    event_hooks = {"request": [on_request], "response": [on_response]}

    async with httpx.AsyncClient(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(app),
        follow_redirects=True,
    ) as http:
        await http.get("http://127.0.0.1:8000/redirect", auth=("username", "password"))

    assert events == [
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"location": "/", "server": "testserver"},
        },
        {
            "event": "request",
            "headers": {
                "host": "127.0.0.1:8000",
                "user-agent": f"python-httpx/{httpx.__version__}",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "connection": "keep-alive",
                "authorization": "Basic dXNlcm5hbWU6cGFzc3dvcmQ=",
            },
        },
        {
            "event": "response",
            "headers": {"server": "testserver"},
        },
    ]


# ---------------------------------------------------------------------------
# Layered hook stages
# ---------------------------------------------------------------------------


def test_hook_stage_order():
    """
    The five stages fire in a fixed order. For a simple non-stream request:
    request -> response -> response_complete -> hop_end.
    """
    events = []

    def on_request(request):
        events.append("request")
        return "request-value"

    def on_response(response):
        events.append("response")

    def on_response_complete(response):
        events.append("response_complete")

    def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    with httpx.Client(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        response = http.get("http://127.0.0.1:8000/")

    assert events == ["request", "response", "response_complete", "hop_end"]

    execution = response.request.hook_execution
    assert [result.stage for result in execution.results] == events
    assert execution.failures == []
    request_result = execution.results_for("request")[0]
    assert request_result.ok
    assert request_result.value == "request-value"
    assert request_result.request is response.request
    assert execution.results_for("response")[0].response is response


@pytest.mark.anyio
async def test_async_hook_stage_order():
    events = []

    async def on_request(request):
        events.append("request")
        return "request-value"

    async def on_response(response):
        events.append("response")

    async def on_response_complete(response):
        events.append("response_complete")

    async def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    async with httpx.AsyncClient(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        response = await http.get("http://127.0.0.1:8000/")

    assert events == ["request", "response", "response_complete", "hop_end"]

    execution = response.request.hook_execution
    assert [result.stage for result in execution.results] == events
    assert execution.failures == []
    assert execution.results_for("request")[0].value == "request-value"


def test_hook_stage_order_with_redirect():
    """
    Each followed redirect is a full hop, so the body-complete and hop-end
    stages fire before the next request stage.
    """
    events = []

    def on_request(request):
        events.append("request")

    def on_response(response):
        events.append("response")

    def on_response_complete(response):
        events.append("response_complete")

    def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    with httpx.Client(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(app),
        follow_redirects=True,
    ) as http:
        http.get("http://127.0.0.1:8000/redirect")

    assert events == [
        "request",
        "response",
        "response_complete",
        "hop_end",
        "request",
        "response",
        "response_complete",
        "hop_end",
    ]


@pytest.mark.anyio
async def test_async_hook_stage_order_with_redirect():
    events = []

    async def on_request(request):
        events.append("request")

    async def on_response(response):
        events.append("response")

    async def on_response_complete(response):
        events.append("response_complete")

    async def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    async with httpx.AsyncClient(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(app),
        follow_redirects=True,
    ) as http:
        await http.get("http://127.0.0.1:8000/redirect")

    assert events == [
        "request",
        "response",
        "response_complete",
        "hop_end",
        "request",
        "response",
        "response_complete",
        "hop_end",
    ]


# ---------------------------------------------------------------------------
# The error stage
# ---------------------------------------------------------------------------


def failing_app(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("boom", request=request)


def test_error_hook_fires_on_chain_failure():
    seen = []

    def on_error(request, exception):
        seen.append((request, exception))

    with httpx.Client(
        event_hooks={"error": [on_error]},
        transport=httpx.MockTransport(failing_app),
    ) as http:
        with pytest.raises(httpx.ConnectError) as exc_info:
            http.get("http://127.0.0.1:8000/")

    assert len(seen) == 1
    error_request, error_exception = seen[0]
    assert error_exception is exc_info.value
    assert error_request is exc_info.value.request
    execution = error_request.hook_execution
    assert execution.error is exc_info.value


@pytest.mark.anyio
async def test_async_error_hook_fires_on_chain_failure():
    seen = []

    async def on_error(request, exception):
        seen.append((request, exception))

    async with httpx.AsyncClient(
        event_hooks={"error": [on_error]},
        transport=httpx.MockTransport(failing_app),
    ) as http:
        with pytest.raises(httpx.ConnectError) as exc_info:
            await http.get("http://127.0.0.1:8000/")

    assert len(seen) == 1
    error_request, error_exception = seen[0]
    assert error_exception is exc_info.value
    assert error_request is exc_info.value.request
    execution = error_request.hook_execution
    assert execution.error is exc_info.value


def test_error_hook_failure_does_not_mask_original():
    """
    A failure inside an error hook is recorded but the original exception
    always propagates unchanged.
    """

    def failing_error_hook(request, exception):
        raise RuntimeError("error hook failure")

    with httpx.Client(
        event_hooks={"error": [failing_error_hook]},
        transport=httpx.MockTransport(failing_app),
    ) as http:
        with pytest.raises(httpx.ConnectError) as exc_info:
            http.get("http://127.0.0.1:8000/")

    assert str(exc_info.value) == "boom"
    execution = exc_info.value.request.hook_execution
    error_hook_failure = execution.results_for("error")[0]
    assert isinstance(error_hook_failure.exception, RuntimeError)
    assert execution.error is exc_info.value


@pytest.mark.anyio
async def test_async_error_hook_failure_does_not_mask_original():
    async def failing_error_hook(request, exception):
        raise RuntimeError("error hook failure")

    async with httpx.AsyncClient(
        event_hooks={"error": [failing_error_hook]},
        transport=httpx.MockTransport(failing_app),
    ) as http:
        with pytest.raises(httpx.ConnectError) as exc_info:
            await http.get("http://127.0.0.1:8000/")

    assert str(exc_info.value) == "boom"
    execution = exc_info.value.request.hook_execution
    error_hook_failure = execution.results_for("error")[0]
    assert isinstance(error_hook_failure.exception, RuntimeError)
    assert execution.error is exc_info.value


# ---------------------------------------------------------------------------
# Hook failure policies
# ---------------------------------------------------------------------------


def test_continue_policy_records_and_proceeds():
    def fail(response):
        raise RuntimeError("hook failure")

    def succeed(response):
        return "succeed-value"

    with httpx.Client(
        event_hooks={"response": [fail, succeed]},
        hook_error_policy="continue",
        transport=httpx.MockTransport(app),
    ) as http:
        response = http.get("http://127.0.0.1:8000/")

    execution = response.request.hook_execution
    assert len(execution.failures) == 1
    failure = execution.failures[0]
    assert isinstance(failure.exception, RuntimeError)
    assert failure.request is response.request
    assert failure.response is response

    results = execution.results_for("response")
    assert [result.index for result in results] == [0, 1]
    assert not results[0].ok
    assert results[1].ok and results[1].value == "succeed-value"


@pytest.mark.anyio
async def test_async_continue_policy_records_and_proceeds():
    async def fail(response):
        raise RuntimeError("hook failure")

    async def succeed(response):
        return "succeed-value"

    async with httpx.AsyncClient(
        event_hooks={"response": [fail, succeed]},
        hook_error_policy="continue",
        transport=httpx.MockTransport(app),
    ) as http:
        response = await http.get("http://127.0.0.1:8000/")

    execution = response.request.hook_execution
    assert len(execution.failures) == 1
    results = execution.results_for("response")
    assert not results[0].ok
    assert results[1].ok and results[1].value == "succeed-value"


def test_default_policy_preserves_original_exception():
    request = httpx.Request("GET", "http://127.0.0.1:8000/")
    original = RuntimeError("hook failure")

    def fail(response):
        raise original

    def never_called(response):
        raise AssertionError("should not have run")  # pragma: no cover

    with httpx.Client(
        event_hooks={"response": [fail, never_called]},
        transport=httpx.MockTransport(app),
    ) as http:
        with pytest.raises(RuntimeError) as exc_info:
            http.send(request)

    # The original exception is re-raised unchanged...
    assert exc_info.value is original
    # ...later hooks did not run...
    execution = request.hook_execution
    assert len(execution.results_for("response")) == 1
    # ...and the failure was recorded against this request.
    assert execution.failures[0].exception is original


@pytest.mark.anyio
async def test_async_default_policy_preserves_original_exception():
    request = httpx.Request("GET", "http://127.0.0.1:8000/")
    original = RuntimeError("hook failure")

    async def fail(response):
        raise original

    async def never_called(response):
        raise AssertionError("should not have run")  # pragma: no cover

    async with httpx.AsyncClient(
        event_hooks={"response": [fail, never_called]},
        transport=httpx.MockTransport(app),
    ) as http:
        with pytest.raises(RuntimeError) as exc_info:
            await http.send(request)

    assert exc_info.value is original
    execution = request.hook_execution
    assert len(execution.results_for("response")) == 1
    assert execution.failures[0].exception is original


def test_per_stage_policy_mapping_recorded():
    def fail_request(request):
        raise RuntimeError("request failure")

    def fail_response(response):
        raise ValueError("response failure")

    client = httpx.Client(
        event_hooks={"request": [fail_request], "response": [fail_response]},
        hook_error_policy={"request": "continue", "response": "raise"},
        transport=httpx.MockTransport(app),
    )
    request = client.build_request("GET", "http://127.0.0.1:8000/")
    with pytest.raises(ValueError) as exc_info:
        client.send(request)

    assert str(exc_info.value) == "response failure"
    execution = request.hook_execution
    request_failure = execution.results_for("request")[0]
    assert isinstance(request_failure.exception, RuntimeError)


@pytest.mark.anyio
async def test_async_per_stage_policy_mapping_recorded():
    async def fail_request(request):
        raise RuntimeError("request failure")

    async def fail_response(response):
        raise ValueError("response failure")

    client = httpx.AsyncClient(
        event_hooks={"request": [fail_request], "response": [fail_response]},
        hook_error_policy={"request": "continue", "response": "raise"},
        transport=httpx.MockTransport(app),
    )
    request = client.build_request("GET", "http://127.0.0.1:8000/")
    with pytest.raises(ValueError) as exc_info:
        await client.send(request)

    assert str(exc_info.value) == "response failure"
    execution = request.hook_execution
    request_failure = execution.results_for("request")[0]
    assert isinstance(request_failure.exception, RuntimeError)


def test_hook_wrapper_individual_policy():
    def fail(request):
        raise RuntimeError("hook failure")

    def succeed(request):
        return "succeed-value"

    # Client-wide policy remains the default "raise"; only the wrapped hook
    # opts into record-and-continue.
    with httpx.Client(
        event_hooks={"request": [httpx.Hook(fail, on_error="continue"), succeed]},
        transport=httpx.MockTransport(app),
    ) as http:
        response = http.get("http://127.0.0.1:8000/")

    execution = response.request.hook_execution
    assert isinstance(execution.failures[0].exception, RuntimeError)
    assert execution.results_for("request")[1].value == "succeed-value"


@pytest.mark.anyio
async def test_async_hook_wrapper_individual_policy():
    async def fail(request):
        raise RuntimeError("hook failure")

    async def succeed(request):
        return "succeed-value"

    async with httpx.AsyncClient(
        event_hooks={"request": [httpx.Hook(fail, on_error="continue"), succeed]},
        transport=httpx.MockTransport(app),
    ) as http:
        response = await http.get("http://127.0.0.1:8000/")

    execution = response.request.hook_execution
    assert isinstance(execution.failures[0].exception, RuntimeError)
    assert execution.results_for("request")[1].value == "succeed-value"


# ---------------------------------------------------------------------------
# Snapshot isolation
# ---------------------------------------------------------------------------


def test_snapshot_isolation_during_send():
    calls = []

    def hook_b(request):
        calls.append("b")

    def hook_a(request):
        calls.append("a")
        # Mutating the registry mid-send must not disturb this traversal.
        client.event_hooks.add("request", hook_b)

    client = httpx.Client(
        event_hooks={"request": [hook_a]}, transport=httpx.MockTransport(app)
    )
    client.get("http://127.0.0.1:8000/")
    assert calls == ["a"]

    # Later requests observe the updated table: the snapshot is [a, b], and
    # 'a' appends yet another 'b' that this traversal does not see.
    client.get("http://127.0.0.1:8000/")
    assert calls == ["a", "a", "b"]


@pytest.mark.anyio
async def test_async_snapshot_isolation_during_send():
    calls = []

    async def hook_b(request):
        calls.append("b")

    async def hook_a(request):
        calls.append("a")
        client.event_hooks.add("request", hook_b)

    client = httpx.AsyncClient(
        event_hooks={"request": [hook_a]}, transport=httpx.MockTransport(app)
    )
    await client.get("http://127.0.0.1:8000/")
    assert calls == ["a"]

    await client.get("http://127.0.0.1:8000/")
    assert calls == ["a", "a", "b"]


def test_snapshot_isolation_on_whole_table_replace():
    def old_hook(request):
        calls.append("old")

    def new_hook(request):
        calls.append("new")

    calls = []
    client = httpx.Client(
        event_hooks={"request": [old_hook]}, transport=httpx.MockTransport(app)
    )

    def replace_hooks(request):
        client.event_hooks = {"request": [new_hook]}

    client.event_hooks.add("request", replace_hooks)
    client.get("http://127.0.0.1:8000/")
    assert calls == ["old"]

    client.get("http://127.0.0.1:8000/")
    assert calls == ["old", "new"]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_skips_response_complete():
    events = []

    def on_response(response):
        events.append("response")

    def on_response_complete(response):
        events.append("response_complete")

    def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    with httpx.Client(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        with http.stream("GET", "http://127.0.0.1:8000/") as response:
            assert events == ["response", "hop_end"]
            response.read()

    # The body read happens outside the pipeline; no extra stage fires.
    assert events == ["response", "hop_end"]


@pytest.mark.anyio
async def test_async_streaming_skips_response_complete():
    events = []

    async def on_response(response):
        events.append("response")

    async def on_response_complete(response):
        events.append("response_complete")

    async def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    async with httpx.AsyncClient(
        event_hooks=event_hooks, transport=httpx.MockTransport(app)
    ) as http:
        async with http.stream("GET", "http://127.0.0.1:8000/") as response:
            assert events == ["response", "hop_end"]
            await response.aread()

    assert events == ["response", "hop_end"]


# ---------------------------------------------------------------------------
# No-hooks behaviour stays untouched
# ---------------------------------------------------------------------------


def test_no_hooks_observable_behaviour_unchanged():
    with httpx.Client(transport=httpx.MockTransport(app)) as http:
        response = http.get("http://127.0.0.1:8000/")

    assert not hasattr(response.request, "hook_execution")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_async_no_hooks_observable_behaviour_unchanged():
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as http:
        response = await http.get("http://127.0.0.1:8000/")

    assert not hasattr(response.request, "hook_execution")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Authentication round-trips
# ---------------------------------------------------------------------------


def auth_challenge_app(request: httpx.Request) -> httpx.Response:
    if "x-authed" not in request.headers:
        return httpx.Response(401, headers={"server": "testserver"})
    return httpx.Response(200, headers={"server": "testserver"})


class ChallengeAuth(httpx.Auth):
    def sync_auth_flow(self, request):
        response = yield request
        assert response.status_code == 401
        request.headers["x-authed"] = "1"
        yield request

    async def async_auth_flow(self, request):
        response = yield request
        assert response.status_code == 401
        request.headers["x-authed"] = "1"
        yield request


def test_hooks_fire_on_auth_round_trip():
    events = []

    def on_request(request):
        events.append("request")

    def on_response(response):
        events.append("response")

    def on_response_complete(response):
        events.append("response_complete")

    def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    with httpx.Client(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(auth_challenge_app),
    ) as http:
        http.get("http://127.0.0.1:8000/", auth=ChallengeAuth())

    assert events == [
        # 401 challenge hop, finalized inside the auth flow...
        "request",
        "response",
        "response_complete",
        "hop_end",
        # ...authenticated hop, finalized by send().
        "request",
        "response",
        "response_complete",
        "hop_end",
    ]


@pytest.mark.anyio
async def test_async_hooks_fire_on_auth_round_trip():
    events = []

    async def on_request(request):
        events.append("request")

    async def on_response(response):
        events.append("response")

    async def on_response_complete(response):
        events.append("response_complete")

    async def on_hop_end(response):
        events.append("hop_end")

    event_hooks = {
        "request": [on_request],
        "response": [on_response],
        "response_complete": [on_response_complete],
        "hop_end": [on_hop_end],
    }

    async with httpx.AsyncClient(
        event_hooks=event_hooks,
        transport=httpx.MockTransport(auth_challenge_app),
    ) as http:
        await http.get("http://127.0.0.1:8000/", auth=ChallengeAuth())

    assert events == [
        "request",
        "response",
        "response_complete",
        "hop_end",
        "request",
        "response",
        "response_complete",
        "hop_end",
    ]


# ---------------------------------------------------------------------------
# Concurrent mutation
# ---------------------------------------------------------------------------


def test_concurrent_mutation_never_breaks_iteration():
    import threading

    client = httpx.Client(transport=httpx.MockTransport(app))
    stop = threading.Event()
    errors = []

    def worker_send() -> None:
        while not stop.is_set():
            try:
                client.get("http://127.0.0.1:8000/")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
                stop.set()

    def hook_one(request):
        pass  # pragma: no cover

    def hook_two(request):
        pass  # pragma: no cover

    def worker_mutate() -> None:
        toggle = False
        while not stop.is_set():
            toggle = not toggle
            if toggle:
                client.event_hooks.add("request", hook_one)
                client.event_hooks.add("response", hook_two)
            else:
                client.event_hooks.remove("request", hook_one)
                client.event_hooks.remove("response", hook_two)

    send_thread = threading.Thread(target=worker_send)
    mutate_thread = threading.Thread(target=worker_mutate)
    send_thread.start()
    mutate_thread.start()
    stop.wait(0.2)
    stop.set()
    send_thread.join()
    mutate_thread.join()

    assert errors == []
    client.close()
