import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


sentences = [
    "Hello, how are you?",
    "I am learning Artificial Intelligence.",
    "Large Language Models can understand and generate human language.",
]

for sentence in sentences:
    count = count_tokens(sentence)

    print(f"\nText: {sentence}")
    print(f"Tokens: {count}")
