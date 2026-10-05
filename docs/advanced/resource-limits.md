You can control the connection pool size using the `limits` keyword
argument on the client. It takes instances of `httpx.Limits` which define:

- `max_keepalive_connections`, number of allowable keep-alive connections, or `None` to always
allow. (Defaults 20)
- `max_connections`, maximum number of allowable connections, or `None` for no limits.
(Default 100)
- `keepalive_expiry`, time limit on idle keep-alive connections in seconds, or `None` for no limits. (Default 5)

```python
limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
client = httpx.Client(limits=limits)
```

## Per-origin limits

Origins are distinguished by their scheme, host and port. You can give an
individual origin its own connection quota with `per_origin`, which maps
origin URLs/strings or `(scheme, host, port)` triples to `httpx.OriginLimits`:

- `max_connections`, the maximum number of requests that may be in flight to
the origin at the same time. Further requests wait in an ordered per-origin
queue. `None` applies no per-origin cap.
- `max_keepalive_connections`, the number of idle connections retained for the
origin once requests complete. `None` leaves the global keepalive limit in
charge.
- `pool_timeout`, the maximum time (in seconds) to wait for the per-origin
quota before an `httpx.OriginPoolTimeout` is raised. This timeout is
independent of the global pool timeout. `None` waits indefinitely.

Origins without an entry follow the global limits exactly.

```python
limits = httpx.Limits(
    per_origin={
        "https://slow-api.example.com": httpx.OriginLimits(
            max_connections=4,
            max_keepalive_connections=2,
            pool_timeout=10.0,
        ),
    }
)
client = httpx.Client(limits=limits)
```

A read-only snapshot of the in-flight, waiting and idle connection counts per
origin is available with `client.get_origin_stats()`, and each response
reports whether the request had to wait in `response.extensions["origin_waited"]`.
