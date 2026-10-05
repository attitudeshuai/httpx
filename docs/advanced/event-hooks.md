HTTPX allows you to register "event hooks" with the client, that are called
every time a particular type of event takes place.

There are five event hooks, which occur in this fixed order during the
lifetime of a request:

* `request` - Called after a request is fully prepared, but before it is sent to the network. Passed the `request` instance.
* `response` - Called after the response headers have been fetched from the network, but before the body has been read. Passed the `response` instance.
* `response_complete` - Called once the client pipeline has finished reading the response body. Passed the `response` instance.
* `hop_end` - Called when a single request/response hop has been fully handled, including followed redirects and authentication round-trips. Passed the `response` instance.
* `error` - Called when the whole request chain has failed. Passed the `request` instance and the original exception.

These allow you to install client-wide functionality such as logging, monitoring or tracing.

```python
def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")

def log_response(response):
    request = response.request
    print(f"Response event hook: {request.method} {request.url} - Status {response.status_code}")

client = httpx.Client(event_hooks={'request': [log_request], 'response': [log_response]})
```

You can also use these hooks to install response processing code, such as this
example, which creates a client instance that always raises `httpx.HTTPStatusError`
on 4xx and 5xx responses.

```python
def raise_on_4xx_5xx(response):
    response.raise_for_status()

client = httpx.Client(event_hooks={'response': [raise_on_4xx_5xx]})
```

!!! note
    Response event hooks are called before the response body has been read.

    If you need access to the response body inside a `response` hook, you'll
    need to call `response.read()`, or for AsyncClients, `response.aread()`.
    Use the `response_complete` hook to observe the body once the client
    pipeline has read it.

The hooks are also allowed to modify `request` and `response` objects.

```python
def add_timestamp(request):
    request.headers['x-request-timestamp'] = datetime.now(tz=datetime.utc).isoformat()

client = httpx.Client(event_hooks={'request': [add_timestamp]})
```

Event hooks must always be set as a **list of callables**, and you may register
multiple event hooks for each type of event. Hooks registered for the same
event are called in registration order.

As well as being able to set event hooks on instantiating the client, there
is also an `.event_hooks` property, that allows you to inspect and modify
the installed hooks.

```python
client = httpx.Client()
client.event_hooks['request'] = [log_request]
client.event_hooks['response'] = [log_response, raise_on_4xx_5xx]
```

You can also use the registry methods for explicit, ordered updates:

```python
client.event_hooks.add('hop_end', on_hop_end)
client.event_hooks.remove('request', log_request)
```

!!! note
    Every request runs against an immutable snapshot of the hooks that was
    taken when the request was sent. Adding, removing or replacing hooks while
    requests are in flight never affects those requests, and a traversal can
    never observe a partially modified table. Subsequent requests use the
    updated hooks.

## Hook failure policy

By default, when a hook raises an exception the original exception is
propagated immediately, and any hooks later in registration order are not
called. You can change this with the `hook_error_policy` argument, setting it
to `'continue'` to record the failure and keep running later hooks:

```python
client = httpx.Client(
    event_hooks={'request': [hook_a, hook_b]},
    hook_error_policy='continue',
)
```

The policy can also be configured per stage:

```python
client.hook_error_policy = {'request': 'continue', 'response': 'raise'}
```

An individual hook can override the client-wide policy by wrapping it in
`httpx.Hook(...)`:

```python
client.event_hooks['request'].append(
    httpx.Hook(log_request, on_error='continue')
)
```

The outcome of every hook invocation is recorded on a per-request execution
context, available as `request.hook_execution`. It exposes:

* `.results` - Every `HookResult`, in execution order.
* `.failures` - Only the results that captured an exception.
* `.results_for(stage)` - Results for a particular stage.
* `.error` - The exception that failed the chain, if any.

Each `HookResult` has `.stage`, `.index`, `.hook`, `.request`, `.response`,
`.value`, `.exception` and an `.ok` property, so recorded failures can always
be matched to the request that produced them. Exceptions are never wrapped:
the original exception always propagates unchanged, and failures of `error`
hooks are recorded but never mask it.

!!! note
    If you are using HTTPX's async support, then you need to be aware that
    hooks registered with `httpx.AsyncClient` MUST be async functions,
    rather than plain functions.

!!! note
    When streaming a request the client does not read the response body, so
    `response_complete` does not fire; `hop_end` still fires once the headers
    have been handled.
