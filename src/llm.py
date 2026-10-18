
import time

from google import genai

from src.config import (
    MAX_API_RETRIES,
    MIN_INTERVAL_S,
    MODEL,
    require_api_key,
)

_client = None
_last_call = 0.0


def _get_client():
    """Create the Gemini client on first use, not at import time."""
    global _client

    if _client is None:
        _client = genai.Client(api_key=require_api_key())

    return _client


def _wait_for_rate_limit() -> None:
    """Keep at least MIN_INTERVAL_S between consecutive API calls."""
    global _last_call

    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)

    _last_call = time.monotonic()


def ask_text(prompt: str) -> str:
    """Send a prompt to Gemini and return its text response."""

    last_error = None

    for attempt in range(MAX_API_RETRIES):
        _wait_for_rate_limit()

        try:
            response = _get_client().models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            return response.text or ""
        except SystemExit:
            raise
        except Exception as exc:  # transient API/network failure
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini request failed after {MAX_API_RETRIES} attempts: "
        f"{last_error}"
    )
