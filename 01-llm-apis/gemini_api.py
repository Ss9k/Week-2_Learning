import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Timeout is in milliseconds: 120 seconds
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=120_000)
)

print(f"Sending request to model: {MODEL}...")

start_time = time.time()

try:
    response = client.models.generate_content(
        model=MODEL,
        contents="Explain what an API is in simple language with an example.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),
            max_output_tokens=300
        )
    )

    elapsed_time = time.time() - start_time

    print("\nResponse:")
    print(response.text)

    print("\n--- Usage Information ---")
    if response.usage_metadata:
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Output tokens:", response.usage_metadata.candidates_token_count)
        print("Total tokens:", response.usage_metadata.total_token_count)

    print(f"Response time: {elapsed_time:.2f} seconds")

except Exception as e:
    print(f"\nRequest failed: {e}")
    print("Check the model availability, API status, and network.")
