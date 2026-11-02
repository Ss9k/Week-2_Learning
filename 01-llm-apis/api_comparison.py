
import os
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types
from groq import Groq
import ollama


load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROMPT = (
    "Explain RAG using one simple real-world example. "
    "Keep your answer within 120 words."
)

SYSTEM_PROMPT = (
    "You are a helpful AI engineering tutor. "
    "Explain concepts clearly using practical examples."
)

MAX_OUTPUT_TOKENS = 800

results = []


# --------------------------------------------------
# Common result structure
# --------------------------------------------------

def save_result(
    provider,
    model,
    output="",
    input_tokens=None,
    output_tokens=None,
    latency=None,
    error=None,
    finish_reason=None
):

    total_tokens = None

    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    results.append({
        "provider": provider,
        "model": model,
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency": latency,
        "status": "Success" if error is None else "Failed",
        "error": error,
        "finish_reason": finish_reason
    })


# --------------------------------------------------
# Helper: Normalize finish reason
# --------------------------------------------------

def normalize_finish_reason(reason):

    if reason is None:
        return "Not available"

    return getattr(reason, "name", str(reason))


# --------------------------------------------------
# 1. Gemini
# --------------------------------------------------

def test_gemini():

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key or not model:
        save_result(
            "Gemini",
            model or "Not configured",
            error="Missing API key or model"
        )
        return

    print("\nTesting Gemini...")

    client = None
    start = time.perf_counter()

    try:

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=120_000)
        )

        response = client.models.generate_content(
            model=model,
            contents=f"{SYSTEM_PROMPT}\n\n{PROMPT}",
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        latency = time.perf_counter() - start

        usage = response.usage_metadata

        finish_reason = None

        if response.candidates:
            finish_reason = normalize_finish_reason(
                response.candidates[0].finish_reason
            )

        save_result(
            provider="Gemini",
            model=model,
            output=response.text or "",
            input_tokens=(
                usage.prompt_token_count if usage else None
            ),
            output_tokens=(
                usage.candidates_token_count if usage else None
            ),
            latency=latency,
            finish_reason=finish_reason
        )

    except Exception as e:

        save_result(
            "Gemini",
            model,
            error=str(e)
        )

    finally:

        if client:
            client.close()


# --------------------------------------------------
# 2. Groq
# --------------------------------------------------

def test_groq():

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not api_key or not model:
        save_result(
            "Groq",
            model or "Not configured",
            error="Missing API key or model"
        )
        return

    print("\nTesting Groq...")

    start = time.perf_counter()

    try:

        client = Groq(
            api_key=api_key,
            timeout=120.0
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            max_completion_tokens=MAX_OUTPUT_TOKENS
        )

        latency = time.perf_counter() - start

        usage = response.usage
        choice = response.choices[0]

        finish_reason = normalize_finish_reason(
            choice.finish_reason
        )

        save_result(
            provider="Groq",
            model=model,
            output=choice.message.content or "",
            input_tokens=(
                usage.prompt_tokens if usage else None
            ),
            output_tokens=(
                usage.completion_tokens if usage else None
            ),
            latency=latency,
            finish_reason=finish_reason
        )

        client.close()

    except Exception as e:

        save_result(
            "Groq",
            model,
            error=str(e)
        )


# --------------------------------------------------
# 3. Ollama
# --------------------------------------------------


def test_ollama():

    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    print("\nTesting Ollama...")

    start = time.perf_counter()

    try:

        stream = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            stream=True,
            think=False
        )

        output = ""
        input_tokens = None
        output_tokens = None
        finish_reason = None

        for chunk in stream:

            content = chunk.message.content or ""
            output += content

            if chunk.done:

                input_tokens = chunk.prompt_eval_count
                output_tokens = chunk.eval_count
                finish_reason = chunk.done_reason

        latency = time.perf_counter() - start

        save_result(
            provider="Ollama",
            model=model,
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
            finish_reason=normalize_finish_reason(finish_reason)
        )

    except Exception as e:

        save_result(
            "Ollama",
            model,
            error=str(e)
        )


# --------------------------------------------------
# Run all providers
# --------------------------------------------------

def main():

    print("=" * 65)
    print("             LLM PROVIDER COMPARISON")
    print("=" * 65)

    print("\nPrompt:", PROMPT)

    test_gemini()
    test_groq()
    test_ollama()

    # --------------------------------------------------
    # Display comparison results
    # --------------------------------------------------

    print("\n\n" + "=" * 65)
    print("                 COMPARISON RESULTS")
    print("=" * 65)

    for result in results:

        print(f"\nProvider      : {result['provider']}")
        print(f"Model         : {result['model']}")
        print(f"Status        : {result['status']}")

        if result["status"] == "Success":

            print(f"Latency       : {result['latency']:.2f} seconds")
            print(f"Input tokens  : {result['input_tokens']}")
            print(f"Output tokens : {result['output_tokens']}")
            print(f"Total tokens  : {result['total_tokens']}")
            print(f"Finish reason : {result['finish_reason']}")

            if result["finish_reason"] in ("MAX_TOKENS", "length"):
                print("WARNING: Response may have been truncated!")

            print("\nGenerated response:")
            print(result["output"])

        else:

            print("Error:", result["error"])

        print("-" * 65)

    # --------------------------------------------------
    # Performance summary
    # --------------------------------------------------

    successful_results = [
        result for result in results
        if result["status"] == "Success"
    ]

    if successful_results:

        fastest = min(
            successful_results,
            key=lambda result: result["latency"]
        )

        print("\n" + "=" * 65)
        print("                  PERFORMANCE SUMMARY")
        print("=" * 65)

        print(
            f"\nFastest provider: {fastest['provider']} "
            f"({fastest['latency']:.2f} seconds)"
        )

        print(
            f"Successful providers: "
            f"{len(successful_results)}/{len(results)}"
        )


if __name__ == "__main__":
    main()
