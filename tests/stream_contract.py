"""Shared assertions for canonical provider streaming error envelopes."""


def assert_canonical_stream_error_envelope(
    events: list[str], *, user_message_substr: str
) -> None:
    """Native transports emit message_start → text error → message_stop."""
    blob = "".join(events)
    assert "message_start" in blob
    assert user_message_substr in blob
    assert "message_stop" in blob
    assert "event: error\ndata:" not in blob
