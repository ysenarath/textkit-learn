import re

import numpy as np
import torch
import transformers


class SentimentAnalyzer:
    def __init__(
        self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", num_runs=5
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful assistant trained to classify the sentiment of word definitions. "
                "For each definition, evaluate the tone of the meaning and categorize it as one of the following labels: "
                "Positive, Negative, or Neutral. "
                "Use the format:\n\n"
                "Reasoning: <your explanation>\nSentiment: <Positive/Negative/Neutral>"
            ),
        }
        self.num_runs = num_runs

    def extract_sentiment(self, text):
        match = re.search(
            r"Sentiment:\s*(Positive|Negative|Neutral)", text, re.IGNORECASE
        )
        if match:
            return match.group(1).capitalize()
        for word in ["positive", "negative", "neutral"]:
            if word in text.lower():
                return word.capitalize()
        return "Unknown"

    def analyze(self, word_definitions):
        results = []
        for item in word_definitions:
            word = item["word"]
            definition = item["definition"]
            predictions = []

            for seed in range(self.num_runs):
                generator = torch.Generator().manual_seed(seed)
                messages = [
                    self.system_prompt,
                    {
                        "role": "user",
                        "content": f"Word: {word}\nDefinition: {definition}",
                    },
                ]
                output = self.pipeline(
                    messages,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    generator=generator,
                )
                output_text = output[0]["generated_text"][-1]["content"]
                sentiment = self.extract_sentiment(output_text)
                predictions.append(sentiment)

            final_sentiment, confidence = self._compute_confidence(predictions)
            results.append({
                "word": word,
                "definition": definition,
                "sentiment": final_sentiment,
                "confidence": confidence,
                "all_predictions": predictions,
            })
        return results

    def _compute_confidence(self, predictions):
        unique_preds, counts = np.unique(predictions, return_counts=True)
        max_count_idx = np.argmax(counts)
        confidence = counts[max_count_idx] / len(predictions)
        return unique_preds[max_count_idx], round(float(confidence), 2)


if __name__ == "__main__":
    word_definitions = [
        {"word": "Good", "definition": "Competent or talented."},
        {"word": "Bad", "definition": "Inaccurate; incorrect."},
        {"word": "Neutral", "definition": "Not helping or hindering."},
        {"word": "Amazing", "definition": "Extremely impressive or good."},
        {"word": "Terrible", "definition": "Extremely bad or unpleasant."},
        {"word": "Sick", "definition": "Affected by illness or disease."},
        {
            "word": "Sick",
            "definition": "Extremely impressive or cool (informal usage).",
        },
    ]

    analyzer = SentimentAnalyzer(num_runs=5)
    results = analyzer.analyze(word_definitions)

    for result in results:
        print(
            f"Word: {result['word']}\nDefinition: {result['definition']}\n"
            f"Predicted Sentiment: {result['sentiment']}\nConfidence: {result['confidence']}\n"
            f"Votes: {result['all_predictions']}\n{'-' * 60}"
        )
