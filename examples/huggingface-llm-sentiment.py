import re

import torch
import transformers

# Initialize model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
)

# System prompt (constant)
system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful assistant trained to classify the sentiment of messages. "
        "You must explain your reasoning first, then at the end of your response, output exactly one of these labels: Positive, Negative, or Neutral. "
        "Use the format:\n\n"
        "Reasoning: <your explanation>\nSentiment: <Positive/Negative/Neutral>"
    ),
}


# Extract sentiment function
def extract_sentiment(text):
    match = re.search(
        r"Sentiment:\s*(Positive|Negative|Neutral)", text, re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()
    for word in ["positive", "negative", "neutral"]:
        if word in text.lower():
            return word.capitalize()
    return "Unknown"


def analyze_sentiment_batch(user_texts, pipeline):
    results = []
    for text in user_texts:
        messages = [system_prompt, {"role": "user", "content": text}]
        output = pipeline(messages, max_new_tokens=100)
        generated = output[0]["generated_text"][-1]["content"]
        sentiment = extract_sentiment(generated)
        results.append((text, generated, sentiment))
    return results


# List of user messages
user_texts = [
    "I didn’t get the promotion, but I’m happy for my colleague.",
    "I absolutely love the new update!",
    "This is the worst experience I've had with this service.",
    "I guess it's okay, nothing special.",
]

# Process each message
for user_text, output_text, sentiment in analyze_sentiment_batch(user_texts):
    print(
        f"Input: {user_text}\nOutput: {output_text}\nExtracted Sentiment: {sentiment}\n{'-' * 60}"
    )


"""
can we combine Cross-validation with Multiple Models with  Model Self-Validation
"""
