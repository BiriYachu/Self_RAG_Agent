
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Gemini model configuration
MODEL = "gemini-3.1-flash-lite"

# Maximum passages retrieved per search
TOP_K = 2

# Maximum query rewrites
MAX_REWRITES = 1

# Maximum answer regeneration attempts
MAX_REGENERATES = 1

# Delay between Gemini API calls
MIN_INTERVAL_S = 4.5


def require_api_key() -> str:
    """Read the Gemini API key or report a helpful error."""
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    return key


if __name__ == "__main__":
    print(f"Python: {sys.executable}")
    print(f"Model: {MODEL}")
    print(f"Top-k: {TOP_K}")
    print(
        "Gemini API key configured: "
        f"{bool(os.getenv('GEMINI_API_KEY'))}"
    )
