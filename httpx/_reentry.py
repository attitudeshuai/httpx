"""
Reentrancy tracking for requests that are issued while the client is already
sending another request on the same thread/task.

Callbacks that run as part of the send chain may themselves call back into the
client:

* request/response event hooks,
* authentication flow generators (the user code between `yield` points),
* transport wrappers (a custom transport that uses the same client while
  handling a request).

Such *nested* requests share the client of the outer request, and therefore
also share its connection pool and client level state. This module keeps a
stack of "frames" that records which callbacks are currently executing, and
how deeply requests are nested.

The stack is stored in a `ContextVar`, which means:

* each thread keeps its own stack,
* each asyncio task / trio task keeps its own stack,
* concurrent top-level sends never share nesting state.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    from ._models import Cookies

__all__ = [
    "DEFAULT_MAX_NESTED_DEPTH",
    "NESTED_POLICIES",
    "NestedRequestPolicy",
    "NestedRequestWarning",
    "ReentryConfig",
    "ReentrySource",
    "ReentryState",
]


class ReentrySource(str, enum.Enum):
    """
    Identifies the send-chain callback from which a nested request originated.
    """

    REQUEST_HOOK = "request_hook"
    RESPONSE_HOOK = "response_hook"
    AUTH = "auth"
    TRANSPORT = "transport"


class NestedRequestPolicy(str, enum.Enum):
    """
    What the client should do once the configured nesting depth is reached:

    * `ALLOW` - keep issuing nested requests regardless of the depth.
    * `WARN`  - issue a `NestedRequestWarning`, but still send the request.
    * `REJECT`- raise a `NestedDepthExceeded` exception without sending.
    """

    ALLOW = "allow"
    WARN = "warn"
    REJECT = "reject"


NESTED_POLICIES = frozenset(policy.value for policy in NestedRequestPolicy)

#: The default nesting depth cap. The cap only ever affects code that calls
#: back into the client while another send is already in progress.
DEFAULT_MAX_NESTED_DEPTH = 4


class NestedRequestWarning(RuntimeWarning):
    """
    Emitted when the configured reentrant nesting depth is reached and the
    client policy is `NestedRequestPolicy.WARN`.
    """


@dataclasses.dataclass(frozen=True)
class ReentryConfig:
    """
    Validated nesting configuration for a client instance.
    """

    max_depth: int | None
    policy: NestedRequestPolicy

    @staticmethod
    def build(
        max_depth: int | None, policy: str | NestedRequestPolicy
    ) -> ReentryConfig:
        if max_depth is not None:
            # `bool` is a subclass of `int`, so reject it explicitly.
            if isinstance(max_depth, bool) or not isinstance(max_depth, int):
                raise TypeError(
                    "max_nested_depth must be an int or None, "
                    f"got {type(max_depth).__name__}."
                )
            if max_depth < 1:
                raise ValueError(
                    "max_nested_depth must be greater than or equal to 1, "
                    f"got {max_depth}."
                )

        if isinstance(policy, NestedRequestPolicy):
            resolved_policy = policy
        elif isinstance(policy, str):
            try:
                resolved_policy = NestedRequestPolicy(policy)
            except ValueError:
                allowed = ", ".join(sorted(NESTED_POLICIES))
                raise ValueError(
                    f"on_nested_depth must be one of {{{allowed}}}, got {policy!r}."
                ) from None
        else:
            raise TypeError(
                "on_nested_depth must be a str or NestedRequestPolicy, "
                f"got {type(policy).__name__}."
            )

        return ReentryConfig(max_depth=max_depth, policy=resolved_policy)


@dataclasses.dataclass
class _Frame:
    # Either a full `client.send()` ("send") or a callback boundary that may
    # run user code ("callback").
    kind: str
    # Callback frames record which part of the chain is executing.
    source: ReentrySource | None = None
    # Nested send frames own an isolated cookie jar. Top-level send frames
    # leave this as `None` and use the client's regular cookie jar.
    cookies: Cookies | None = None


class ReentryState:
    """
    Per execution-unit (thread/task) reentrancy state for a single client.

    A fresh, empty stack is used outside of any send. The stack is replaced
    functionally (a new tuple is set through a `ContextVar` token) on push and
    restored through the token on pop, so concurrent execution units always
    observe their own state.
    """

    def __init__(self) -> None:
        self._stack: contextvars.ContextVar[tuple[_Frame, ...]] = (
            contextvars.ContextVar("httpx_reentry_stack", default=())
        )

    # --- introspection -------------------------------------------------

    @property
    def send_count(self) -> int:
        return sum(1 for frame in self._stack.get() if frame.kind == "send")

    @property
    def depth(self) -> int:
        """
        Nesting depth of the request currently being sent. `0` at the top
        level, `1` for a request nested one level deep, and so on.
        """
        return max(0, self.send_count - 1)

    @property
    def is_nested(self) -> bool:
        """
        `True` while a send frame *other than the current one* is active,
        i.e. when request processing runs inside another in-flight send.
        """
        return self.depth >= 1

    def has_active_send(self) -> bool:
        """
        `True` if any send frame is active, including the current one.
        Used at `send()` entry, before this send's own frame is pushed.
        """
        return self.send_count >= 1

    @property
    def origin(self) -> ReentrySource | None:
        """
        Origin of the innermost nested send, per the current execution unit.

        This is the callback frame closest below the innermost `send` frame.
        Returns `None` for top-level sends or outside of any send.
        """
        stack = self._stack.get()
        send_index = -1
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].kind == "send":
                send_index = index
                break
        for frame in reversed(stack[:send_index]):
            if frame.kind == "callback":
                return frame.source
        return None

    def current_send_cookies(self) -> Cookies | None:
        """
        The isolated cookie jar owned by the innermost nested send, or `None`
        when the regular client jar should be used.
        """
        for frame in reversed(self._stack.get()):
            if frame.kind == "send":
                return frame.cookies
        return None

    # --- frame boundaries ----------------------------------------------

    @contextlib.contextmanager
    def send_frame(
        self, *, cookies: Cookies | None = None
    ) -> typing.Iterator[None]:
        frame = _Frame(kind="send", cookies=cookies)
        token = self._stack.set(self._stack.get() + (frame,))
        try:
            yield
        finally:
            self._stack.reset(token)

    @contextlib.contextmanager
    def callback_frame(self, source: ReentrySource) -> typing.Iterator[None]:
        token = self._stack.set(
            self._stack.get() + (_Frame(kind="callback", source=source),)
        )
        try:
            yield
        finally:
            self._stack.reset(token)
