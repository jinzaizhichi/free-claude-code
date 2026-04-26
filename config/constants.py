"""Shared defaults used by config models and provider adapters."""

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920

# Max bytes read from a non-200 native messages response when verbose error logging is on.
NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES = 4096
