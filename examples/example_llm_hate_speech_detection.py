import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Mapping, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DisjunctiveConstraint,
)


class Verbalizer:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def verbalize(self, input: str) -> str:
        raise NotImplementedError


class RegexVerbalizer(Verbalizer):
    labels_re: re.Pattern

    def __init__(self, **labels: str):
        self.labels = set(labels)
        labels = "|".join([
            f"(?P<{re.escape(key)}>{token})" for key, token in labels.items()
        ])
        self.labels_re = re.compile(labels)

    def verbalize(self, input: str) -> str:
        m = self.labels_re.search(input)
        if m is None:
            return None
        for label in self.labels:
            if m.group(label) is not None:
                return label
        return None


class HateSpeechClassifier:
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
        constraint_token_ids = [
            self.tokenizer("neutral").input_ids[1:],
            self.tokenizer("offensive").input_ids[1:],
            self.tokenizer("hateful").input_ids[1:],
        ]
        self.constraints = [
            DisjunctiveConstraint(constraint_token_ids),
        ]
        self.verbalizer = RegexVerbalizer(
            neutral="neutral",
            offensive="offensive",
            hateful="hateful|hate",
        )
        self.prompt = """Question: Is the given input offensive or hateful or neutral?

Options:
- offensive
- hateful
- neutral

Input: {input}

Output: The input is """

    def predict_sample(self, input: str) -> str:
        # content = "\n".join(self.prompt).format(input=input)
        # messages = [
        #     {"role": "user", "content": content},
        # ]
        # formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer(
            self.prompt.format(input=input),
            return_tensors="pt",
            # truncation=True,
            # padding="longest",
        )
        input_ids = model_inputs["input_ids"].to("cuda")
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=100,
            num_beams=10,
            do_sample=False,
            constraints=self.constraints,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        outputs = []
        for output_text in self.tokenizer.batch_decode(output_ids):
            output_text = output_text.split("Output: The input is ", maxsplit=1)
            if len(output_text) == 1:
                output_text = ""
            else:
                output_text = output_text[1].strip()
            label = self.verbalizer.verbalize(output_text)
            if label is None:
                label = "other"
            outputs.append({
                "label": label,
                "output_text": output_text,
            })
        return outputs[0]

    def predict(self, input: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str):
            return [self.predict_sample(input)]
        return [self.predict_sample(text) for text in input]


def load_dataset_split(base_path, split, aspect):
    split_df = pd.read_csv(base_path / split / (aspect + ".csv"))
    labels = split_df["pos_group"].isna()
    split_df = split_df["text"].to_frame()
    split_df["labels"] = labels
    return split_df


def labels_to_float_32(example):
    labels = [1.0 if tf else 0.0 for tf in example["labels"]]
    return {"label": labels}


def load_dataset() -> Mapping[str, DatasetDict]:
    base_data_path = Path("data/processed/qian-2021-lifelong-750")
    with open(base_data_path / "order.txt", "r") as fp:
        order = fp.read()
    order = order.strip().split("\n")
    datasets = OrderedDict()
    for step_id, filename in enumerate(order):
        aspect = filename[:-4]
        train_df = load_dataset_split(base_data_path, "train", aspect)
        valid_df = load_dataset_split(base_data_path, "valid", aspect)
        test_df = load_dataset_split(base_data_path, "test", aspect)
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "valid": Dataset.from_pandas(valid_df),
            "test": Dataset.from_pandas(test_df),
        })
        raw_datasets = raw_datasets.map(labels_to_float_32, batched=True)
        raw_datasets = raw_datasets.remove_columns("labels")
        raw_datasets = raw_datasets.rename_column("label", "labels")
        # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        # tokenized_datasets = tokenized_datasets.remove_columns([
        #     "text",
        # ])
        # tokenized_datasets.set_format("torch")
        datasets[aspect] = raw_datasets
    return datasets


def main():
    from tqdm import auto as tqdm

    model = HateSpeechClassifier()
    datasets = load_dataset()
    for aspect, dataset in datasets.items():
        dset = dataset["test"]
        y_pred = []
        for doc in tqdm.tqdm(dset, total=dset.num_rows):
            y_pred_sample = model.predict_sample(doc)
            y_pred += [{**doc, **y_pred_sample}]
        pd.DataFrame(y_pred).to_csv(
            f"logs/hate-speech-classification/llm-zero-shot/{aspect}.csv", index=False
        )

        from datasets import interleave_datasets

        interleave_datasets([
            dset.filter(lambda doc: doc["labels"] >= 0.5).take(100),
            dset.filter(lambda doc: doc["labels"] < 0.5).take(100),
        ])


def evaluate(df):
    y_true, y_pred = df["y_true"], df["y_pred"]
    target_names = ["Neutral", "Hateful/Offensive"]
    results = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    return pd.json_normalize(results, sep=">")


def labels_int2str(label_ints):
    return [
        datasets["train"].features["labels"].feature.int2str(int(label_int))
        for label_int in label_ints
    ]
