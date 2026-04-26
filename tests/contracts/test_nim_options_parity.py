"""Keep config NimSettings and provider NimRequestOptions aligned."""

from config.nim import NimSettings
from providers.nvidia_nim.options import NimRequestOptions


def test_nim_settings_matches_nim_request_options_fields() -> None:
    assert set(NimSettings.model_fields) == set(NimRequestOptions.model_fields)
