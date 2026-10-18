
import os

from dotenv import load_dotenv
from google import genai

from src.config import MODEL

# Load the API key from the project's .env file.
load_dotenv()

# Read the API key.
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

# Connect to Gemini.
client = genai.Client(api_key=API_KEY)


def ask_text(prompt: str) -> str:
    """Send a prompt to Gemini and return its text response."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return response.text or ""
