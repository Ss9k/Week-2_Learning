import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI engineering tutor."
        },
        {
            "role": "user",
            "content": "Explain RAG with a simple example."
        }
    ]
)

print(response.message.content)

print("\nUsage:")
print("Prompt tokens:", response.prompt_eval_count)
print("Generated tokens:", response.eval_count)
