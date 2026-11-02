# 01 - LLM APIs

## 1. Tokens
- Tokens are the units of text processed by an LLM.
- One word is not necessarily one token.
- API billing commonly depends on input and output tokens.
- Tokenization differs across models.

## 2. Context Window
- Maximum context an LLM can process in a request.
- Includes instructions, conversation, input and generated output.
- Large documents may require chunking and retrieval.

## 3. Temperature
- Controls randomness in generation.
- Lower temperature: generally more predictable.
- Higher temperature: generally more varied.

## 4. LLM APIs
- OpenAI: Responses API.
- Gemini: generate_content().
- Claude: messages.create().
- Ollama: local model inference.

## 5. API Cost
Cost =
(input_tokens * input_price +
output_tokens * output_price) / 1,000,000

## 6. Streaming
- Returns generated content incrementally.
- Improves perceived responsiveness.
- Useful for chat interfaces.

## 7. Timeout and Retries
- Timeout limits request waiting time.
- 429 indicates rate limiting.
- Exponential backoff helps manage retries.
- Do not blindly retry permanent errors.

## 8. Prompt Caching
- Reuses eligible prompt prefixes.
- Reduces repeated processing.
- Can reduce latency and input cost.
- Provider implementations differ.

## 9. Ollama
- Runs supported LLMs locally.
- Supports Qwen, Llama, Gemma and others.
- Does not require cloud inference for local models.

## 10. Production Considerations
- Never expose API keys.
- Track token usage.
- Monitor latency and failures.
- Handle rate limits.
- Set timeouts.
- Use appropriate models.
- Monitor costs.
- Log errors without logging sensitive user data.
