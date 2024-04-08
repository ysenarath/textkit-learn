import re
from typing import List, Mapping, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class RegexVerbalizer:
    labels_re: re.Pattern
    labels: set[str]
    multilabel: bool

    def __init__(self, labels: Mapping[str, str], multilabel: bool = False):
        self.labels = set(labels)
        labels = "|".join([
            f"(?P<{re.escape(key)}>{token})" for key, token in labels.items()
        ])
        self.labels_re = re.compile(labels, re.IGNORECASE)
        self.multilabel = multilabel

    def _verbalize_multilabel(self, input: str) -> list[str]:
        pos = 0
        m = self.labels_re.search(input, pos)
        labels = set()
        while m is not None:
            for label in self.labels:
                if m.group(label) is not None:
                    labels.add(label)
            m = self.labels_re.search(input, m.end())
        return list(labels)

    def _verbalize_multiclass(self, input: str) -> str:
        m = self.labels_re.search(input)
        if m is None:
            return None
        for label in self.labels:
            if m.group(label) is not None:
                return label
        return None

    def verbalize(self, input: str) -> Union[str, list[str]]:
        if self.multilabel:
            return self._verbalize_multilabel(input)
        return self._verbalize_multiclass(input)


class EmotionClassifier:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.verbalizer = RegexVerbalizer(
            {
                "anger": "anger",
                "disgust": "disgust",
                "fear": "fear",
                "happiness": "happiness",
                "sadness": "sadness",
                "surprise": "surprise",
                "neutral": "neutral",
            },
            multilabel=True,
        )
        # prompt-multi-v1
        self.prompt = """Question: Identify the most appropriate emotion classes from the provided options that is observed in the given input.

Options:
- anger
- disgust
- fear
- happiness
- sadness
- surprise
- neutral

Input: {input}

Output: The input is classified as """

    def predict_sample(self, input: str) -> str:
        model_inputs = self.tokenizer(
            self.prompt.format(input=input),
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"].to("cuda")
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=30,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=1e-5,
            do_sample=True,
        )
        outputs = []
        for output_text in self.tokenizer.batch_decode(output_ids):
            output_text = output_text.split(
                "Output: The input is classified as ", maxsplit=1
            )
            if len(output_text) == 1:
                output_text = ""
            else:
                output_text = output_text[1].strip()
            labels = self.verbalizer.verbalize(output_text)
            outputs.append({
                "labels": labels,
                "output_text": output_text,
            })
        return outputs[0]

    def predict(self, input: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str):
            return [self.predict_sample(input)]
        return [self.predict_sample(text) for text in input]
