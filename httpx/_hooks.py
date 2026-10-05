"""
Layered event hook machinery.

Hooks can be registered for five distinct stages, which occur in this fixed
order during the lifetime of a request:

* ``request``          - Called after the request has been prepared, but before
                         it is sent to the network. ``hook(request)``
* ``response``         - Called once the response headers have arrived, before
                         the body has been read. ``hook(response)``
* ``response_complete``- Called once the client pipeline has finished reading
                         the response body. ``hook(response)``
* ``hop_end``          - Called when a single request/response hop has been
                         fully handled. ``hook(response)``
* ``error``            - Called when the whole request chain has failed.
                         ``hook(request, exception)``

Multiple hooks registered for the same stage are called in registration order.
Each request sends against an immutable snapshot of the registry, so hooks can
be added or removed on the client while requests are in flight without any
iteration ever observing a partially modified table.

When a hook fails the configured error policy determines whether the exception
is recorded and the remaining hooks continue to run (``"continue"``), or whether
it is re-raised immediately, preserving the original exception
(``"raise"`` - the default and historical behaviour).
"""

from __future__ import annotations

import enum
import threading
import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    from ._models import Request, Response

#: A hook is any plain callable...
EventHook = typing.Callable[..., typing.Any]

#: The five hook stages, in the fixed order in which they occur.
HOOK_STAGES = ("request", "response", "response_complete", "hop_end", "error")

__all__ = [
    "EventHook",
    "EventHooks",
    "Hook",
    "HookErrorPolicy",
    "HookExecution",
    "HookResult",
    "normalize_hook_policy",
]


class HookErrorPolicy(enum.Enum):
    """
    What should happen when a hook raises an exception.

    * ``RAISE`` - Record the failure and re-raise the original exception
        immediately. Any hooks later in registration order are not called.
        This is the default, matching the historical behaviour.
    * ``CONTINUE`` - Record the failure and continue with the remaining hooks.
    """

    RAISE = "raise"
    CONTINUE = "continue"


def _coerce_policy(value: str | HookErrorPolicy) -> HookErrorPolicy:
    if isinstance(value, HookErrorPolicy):
        return value
    try:
        return HookErrorPolicy(value)
    except ValueError:  # pragma: no cover
        raise ValueError(
            f"Invalid hook error policy: {value!r}. "
            "Use 'raise' or 'continue'."
        ) from None


HookPolicyValue = typing.Union[str, HookErrorPolicy]
HookPolicyTypes = typing.Union[
    str, HookErrorPolicy, typing.Mapping[str, HookPolicyValue]
]


def normalize_hook_policy(
    policy: HookPolicyTypes,
) -> dict[str, HookErrorPolicy]:
    """
    Normalize a policy argument into a mapping of one policy per stage.

    Accepts either a single value applied to every stage, or a mapping of
    per-stage policies. Unknown stage names raise a ``ValueError``.

    Note that the ``error`` stage policy is never used to abort: failures of
    ``error`` hooks are always recorded and the original chain exception is
    always propagated unchanged.
    """
    if isinstance(policy, (str, HookErrorPolicy)):
        single = _coerce_policy(policy)
        return {stage: single for stage in HOOK_STAGES}

    result = {stage: HookErrorPolicy.RAISE for stage in HOOK_STAGES}
    for stage, value in policy.items():
        if stage not in result:
            raise ValueError(
                f"Unknown hook stage: {stage!r}. "
                f"Valid stages are: {', '.join(HOOK_STAGES)}."
            )
        result[stage] = _coerce_policy(value)
    return result


class Hook:
    """
    Wrap a hook callable with an individual error policy that overrides the
    client-wide policy for that hook.

    Usage:

    ```python
    def log_only(request):
        ...

    client = httpx.Client(
        event_hooks={"request": [httpx.Hook(log_only, on_error="continue")]}
    )
    ```
    """

    def __init__(
        self,
        callback: EventHook,
        *,
        on_error: HookPolicyValue = HookErrorPolicy.RAISE,
    ) -> None:
        if not callable(callback):
            raise TypeError(f"A hook must be callable, got {callback!r}.")
        self.callback = callback
        self.on_error = _coerce_policy(on_error)

    def __repr__(self) -> str:
        return f"<Hook({self.callback!r}, on_error={self.on_error.value!r})>"


def _unwrap_hook(hook: EventHook) -> tuple[EventHook, HookErrorPolicy | None]:
    if isinstance(hook, Hook):
        return hook.callback, hook.on_error
    return hook, None


class HookResult:
    """
    The outcome of invoking a single hook at a single stage.

    **Attributes:**

    * ``stage`` - The name of the stage the hook ran at.
    * ``index`` - The position of the hook within its stage at registration.
    * ``hook`` - The hook callable itself.
    * ``request`` - The request that was being handled.
    * ``response`` - The response, for response-related stages. Otherwise ``None``.
    * ``value`` - The hook's return value, or ``None``.
    * ``exception`` - The exception raised by the hook, or ``None`` on success.
    """

    def __init__(
        self,
        stage: str,
        index: int,
        hook: EventHook,
        request: Request,
        *,
        response: Response | None = None,
        value: typing.Any = None,
        exception: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.index = index
        self.hook = hook
        self.request = request
        self.response = response
        self.value = value
        self.exception = exception

    @property
    def ok(self) -> bool:
        "Return ``True`` if the hook completed without raising."
        return self.exception is None

    def __repr__(self) -> str:
        status = "ok" if self.ok else f"failed: {self.exception!r}"
        return f"<HookResult stage={self.stage!r} #{self.index} {status}>"


class EventHooks(typing.Mapping[str, list[EventHook]]):
    """
    The client-wide hook registry.

    Behaves as a mapping of stage name to a list of hooks, but performs all
    updates behind a lock and can produce immutable snapshots that in-flight
    requests iterate over, so adding/removing hooks can never result in a
    concurrent modification of a traversal.
    """

    def __init__(
        self,
        hooks: None | (typing.Mapping[str, typing.Iterable[EventHook]]) = None,
    ) -> None:
        self._lock = threading.RLock()
        self._hooks: dict[str, list[EventHook]] = {
            stage: [] for stage in HOOK_STAGES
        }
        if hooks is not None:
            self.replace(hooks)

    def _validate_stage(self, stage: str) -> None:
        if stage not in self._hooks:
            raise ValueError(
                f"Unknown hook stage: {stage!r}. "
                f"Valid stages are: {', '.join(HOOK_STAGES)}."
            )

    def replace(
        self, hooks: typing.Mapping[str, typing.Iterable[EventHook]]
    ) -> None:
        "Replace the contents of the registry wholesale."
        with self._lock:
            for stage in HOOK_STAGES:
                if stage in hooks:
                    self._hooks[stage] = list(hooks[stage])
                else:
                    self._hooks[stage] = []

    def add(self, stage: str, hook: EventHook) -> None:
        "Append a hook to the end of the given stage, preserving order."
        self._validate_stage(stage)
        if not callable(hook):
            raise TypeError(f"A hook must be callable, got {hook!r}.")
        with self._lock:
            self._hooks[stage].append(hook)

    def remove(self, stage: str, hook: EventHook) -> None:
        "Remove a previously registered hook. Raises ``ValueError`` if absent."
        self._validate_stage(stage)
        with self._lock:
            self._hooks[stage].remove(hook)

    def clear(self, stage: str | None = None) -> None:
        "Clear a single stage, or every stage when ``stage`` is ``None``."
        with self._lock:
            if stage is None:
                for hooks in self._hooks.values():
                    hooks.clear()
            else:
                self._validate_stage(stage)
                self._hooks[stage].clear()

    def snapshot(self) -> dict[str, tuple[EventHook, ...]]:
        """
        Return an immutable point-in-time copy of the registry.

        Requests iterate over this snapshot; later mutations of the registry
        have no effect on it.
        """
        with self._lock:
            return {
                stage: tuple(self._hooks[stage]) for stage in HOOK_STAGES
            }

    def is_empty(self) -> bool:
        with self._lock:
            return not any(self._hooks.values())

    def copy(self) -> EventHooks:
        return EventHooks(self.snapshot())

    # Mapping interface...

    def __getitem__(self, key: str) -> list[EventHook]:
        self._validate_stage(key)
        with self._lock:
            return self._hooks[key]

    def __setitem__(self, key: str, value: typing.Iterable[EventHook]) -> None:
        self._validate_stage(key)
        with self._lock:
            self._hooks[key] = list(value)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(HOOK_STAGES)

    def __len__(self) -> int:
        return len(HOOK_STAGES)

    def __repr__(self) -> str:
        with self._lock:
            content = {
                stage: list(hooks) for stage, hooks in self._hooks.items()
            }
        return f"EventHooks({content!r})"


class HookExecution:
    """
    The per-request hook execution context.

    Holds the immutable registry snapshot the request runs against, the
    configured policies, and the :class:`HookResult` records produced by every
    hook invocation. Results and failures are therefore always attributable to
    this specific request.
    """

    def __init__(
        self,
        request: Request,
        snapshot: dict[str, tuple[EventHook, ...]],
        policy: typing.Mapping[str, HookErrorPolicy],
    ) -> None:
        self.request = request
        self.snapshot = snapshot
        self.policy = dict(policy)
        self.results: list[HookResult] = []
        #: The exception that failed the chain, if any.
        self.error: BaseException | None = None
        self._index = {stage: 0 for stage in HOOK_STAGES}

    @property
    def failures(self) -> list[HookResult]:
        "All hook results that captured an exception."
        return [result for result in self.results if result.exception is not None]

    def results_for(self, stage: str) -> list[HookResult]:
        "All hook results recorded at the given stage."
        return [result for result in self.results if result.stage == stage]

    def _record(
        self,
        stage: str,
        hook: EventHook,
        request: Request,
        response: Response | None,
        value: typing.Any,
        exception: BaseException | None,
    ) -> HookResult:
        index = self._index[stage]
        self._index[stage] = index + 1
        result = HookResult(
            stage,
            index,
            hook,
            request,
            response=response,
            value=value,
            exception=exception,
        )
        self.results.append(result)
        return result

    def _should_abort(
        self,
        stage: str,
        override: HookErrorPolicy | None,
        exception: BaseException,
    ) -> bool:
        # Control-flow signals (task cancellation, KeyboardInterrupt, ...)
        # always propagate, even under an explicit "continue" policy.
        if not isinstance(exception, Exception):
            return True
        policy = self.policy[stage] if override is None else override
        return policy is HookErrorPolicy.RAISE

    @staticmethod
    def _args(
        stage: str,
        request: Request,
        response: Response | None,
        exception: BaseException | None,
    ) -> tuple[typing.Any, ...]:
        if stage == "request":
            return (request,)
        if stage == "error":
            return (request, exception)
        return (response,)

    def run(
        self,
        stage: str,
        request: Request,
        response: Response | None = None,
    ) -> None:
        """
        Synchronously run every hook registered for ``stage``, in order.
        """
        for registered in self.snapshot.get(stage, ()):
            hook, override = _unwrap_hook(registered)
            args = self._args(stage, request, response, None)
            try:
                value = hook(*args)
            except BaseException as exc:
                self._record(stage, hook, request, response, None, exc)
                if self._should_abort(stage, override, exc):
                    raise
            else:
                self._record(stage, hook, request, response, value, None)

    async def arun(
        self,
        stage: str,
        request: Request,
        response: Response | None = None,
    ) -> None:
        """
        Asynchronously run every hook registered for ``stage``, in order.
        """
        for registered in self.snapshot.get(stage, ()):
            hook, override = _unwrap_hook(registered)
            args = self._args(stage, request, response, None)
            try:
                value = await hook(*args)
            except BaseException as exc:
                self._record(stage, hook, request, response, None, exc)
                if self._should_abort(stage, override, exc):
                    raise
            else:
                self._record(stage, hook, request, response, value, None)

    def fire_error(
        self, request: Request, exception: BaseException
    ) -> None:
        """
        Run the ``error`` hooks after the request chain has failed.

        Exceptions raised by ``error`` hooks are always recorded, and never
        replace or mask the original chain exception, which is re-raised by
        the caller unchanged.
        """
        self.error = exception
        for registered in self.snapshot.get("error", ()):
            hook, _ = _unwrap_hook(registered)
            try:
                hook(request, exception)
            except BaseException as exc:
                self._record("error", hook, request, None, None, exc)

    async def afire_error(
        self, request: Request, exception: BaseException
    ) -> None:
        """
        Asynchronously run the ``error`` hooks after the request chain has
        failed. See :meth:`fire_error`.
        """
        self.error = exception
        for registered in self.snapshot.get("error", ()):
            hook, _ = _unwrap_hook(registered)
            try:
                await hook(request, exception)
            except BaseException as exc:
                self._record("error", hook, request, None, None, exc)
