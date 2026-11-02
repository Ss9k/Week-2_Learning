import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

# Initialize Groq client
client = Groq(
    api_key=API_KEY,
    timeout=120.0
)

print(f"Sending request to model: {MODEL}...")

start_time = time.time()

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI engineering tutor. Explain concepts in simple language with practical examples."
            },
            {
                "role": "user",
                "content": "Explain what an API is in simple language with an example."
            }
        ],
        max_completion_tokens=600,
        temperature=0.7
    )

    elapsed_time = time.time() - start_time

    print("\nResponse:")
    print(response.choices[0].message.content)

    print("\n--- Usage Information ---")

    if response.usage:
        print("Prompt tokens:", response.usage.prompt_tokens)
        print("Completion tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)

    print("Model used:", response.model)
    print("Finish reason:", response.choices[0].finish_reason)
    print(f"Response time: {elapsed_time:.2f} seconds")

except Exception as e:
    print(f"\nRequest failed: {e}")
