from dataclasses import dataclass
from typing import Final

from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.agentic_followup_failure import ServedRound

_MARK_ATTRIBUTE: Final = "_litellm_unusable_tool_call_failure"
_UNBILLED_ROUND_ATTRIBUTE: Final = "_litellm_unusable_tool_call_unbilled_round"


@dataclass(frozen=True, slots=True)
class _UnbilledRound:
    served_round: ServedRound


def _provider_error(error: BaseException | None) -> BaseException | None:
    if isinstance(error, MidStreamFallbackError):
        return _provider_error(error.original_exception)
    return error


def mark_unusable_tool_call_failure(error: BaseException, unbilled_round: ServedRound | None = None) -> None:
    """Tag a response that ended on a tool call the model could not produce: the request and the model's
    output caused it, so the deployment that served it stays healthy. ``unbilled_round`` is what the response
    consumed, for a caller whose own logging object never saw it."""
    setattr(error, _MARK_ATTRIBUTE, True)
    setattr(error, _UNBILLED_ROUND_ATTRIBUTE, None if unbilled_round is None else _UnbilledRound(unbilled_round))


def is_unusable_tool_call_failure(error: BaseException | None) -> bool:
    return getattr(_provider_error(error), _MARK_ATTRIBUTE, False) is True


def _unbilled_round_of(error: BaseException | None) -> ServedRound | None:
    unbilled: Final[object] = getattr(error, _UNBILLED_ROUND_ATTRIBUTE, None)
    return unbilled.served_round if isinstance(unbilled, _UnbilledRound) else None


def take_unbilled_round(error: BaseException) -> ServedRound | None:
    """The failed round's usage and cost, handed out once so nested callers cannot bill it twice."""
    provider_error: Final = _provider_error(error)
    served_round: Final = _unbilled_round_of(provider_error)
    if provider_error is not None and served_round is not None:
        setattr(provider_error, _UNBILLED_ROUND_ATTRIBUTE, None)
    return served_round


def carry_unusable_tool_call_mark(source: BaseException, mapped: BaseException) -> None:
    if is_unusable_tool_call_failure(source):
        mark_unusable_tool_call_failure(mapped, _unbilled_round_of(_provider_error(source)))
