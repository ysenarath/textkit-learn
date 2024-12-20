from __future__ import annotations

import json
import pickle
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Union,
)

import dill.source
import torch
import xxhash
from tqdm import auto as tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from typing_extensions import Literal, Self


class FileCache:
    def __init__(self, path: Union[Path, str] = "cache.json"):
        self.path = Path(path)
        # Create cache file if it does not exist
        if not self.path.exists():
            with open(self.path, "w") as f:
                json.dump({}, f)

    def __getitem__(self, key: str) -> Optional[Dict[str, Any]]:
        key = str(key)
        if not self.path.exists():
            raise KeyError(f"{key}")
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)[key]

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        key = str(key)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        with open(self.path, "w", encoding="utf-8") as f:
            data[key] = value
            json.dump(data, f)

    def __contains__(self, key: str) -> bool:
        with open(self.path, encoding="utf-8") as f:
            return str(key) in json.load(f)


class VerbalizationError(Exception):
    pass


@dataclass
class PipelineConfig:
    """Configuration for the LLM model."""

    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 512
    load_in: Literal["8bit", "4bit", "32bit"] = "4bit"


class Pipeline:
    """A agent class for LLaMA model."""

    config: PipelineConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        try:
            # Configure quantization
            quantization_config = None
            load_in = str(self.config.load_in).lower()
            if load_in.startswith("8"):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in.startswith("4"):
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                quantization_config=quantization_config,
            )
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )
            # Set model to eval mode
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        num_beams: int = 4,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: int = 42,
    ) -> str:
        if do_sample:
            if temperature is None:
                temperature = 1.0
            if top_p is None:
                top_p = 1.0
        set_seed(seed)
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        # Set max tokens
        max_tokens = max_new_tokens or self.config.max_new_tokens
        # Generate response
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            output_scores=False,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )
        # Decode response
        prompt_size = len(input_ids[0])
        s = self.tokenizer.decode(output[0][prompt_size:])
        if s.endswith("<|eot_id|>"):
            s = s[:-10]
        return s

    def format(self, content: str, *, role: str, end_of_turn: bool) -> str:
        out = ""
        if role == "system":
            out = "<|start_header_id|>system<|end_header_id|>\n" + content
            if end_of_turn:
                out += "<|eot_id|>"
        elif role == "user":
            out = "<|start_header_id|>user<|end_header_id|>\n" + content
            if end_of_turn:
                out += (
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                )
        elif role == "assistant":
            out = "<|start_header_id|>assistant<|end_header_id|>\n" + content
            if end_of_turn:
                out += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        else:
            raise ValueError(f"invalid role: {role}")
        return out


class Agent:
    def __init__(
        self,
        pipeline: Pipeline,
        sytem_prompt: str | None = None,
    ):
        if sytem_prompt is None:
            sytem_prompt = "You are a helpful assistant providing useful information to the user."
        self._pipe = weakref.ref(pipeline)
        self.messages = []
        self.messages.append(
            dict(
                content=sytem_prompt,
                role="system",
                end_of_turn=True,
            )
        )

    @property
    def pipe(self) -> Pipeline:
        return self._pipe()

    def send(
        self,
        user: str,
        assistant: Optional[str] = None,
        end_of_turn: bool = False,
    ) -> Self:
        self.messages.append(dict(content=user, role="user", end_of_turn=True))
        if assistant:
            message = dict(
                content=assistant, role="assistant", end_of_turn=end_of_turn
            )
            self.messages.append(message)
        return self

    def get_prompt(self) -> str:
        out = ""
        for message in self.messages:
            out += self.pipe.format(**message)
        return out

    def get_prefix(self) -> str:
        last_message = self.messages[-1]
        if last_message["role"] == "assistant":
            return last_message["content"]
        return ""

    def receive(
        self, verbalizer: Optional[Callable[[str], str]] = None
    ) -> str:
        tries = 0
        while True:
            result = self.get_prefix() + self.pipe.generate(
                self.get_prompt(), 512
            )
            # if last message is assistant, then remove it
            if self.messages[-1]["role"] == "assistant":
                self.messages.pop()
            self.messages.append({
                "content": result,
                "role": "assistant",
                "end_of_turn": True,
            })
            if verbalizer:
                try:
                    return verbalizer(result)
                except VerbalizationError:
                    if tries > 3:
                        raise
            else:
                return result
            tries += 1
            src = dill.source.getsource(verbalizer.__class__).strip()
            if hasattr(verbalizer, "classes"):
                classes_str = f"with classes {verbalizer.classes}"
            else:
                classes_str = ""
            content = (
                f"I can't understand your output. Please response with one of the classes here: {classes_str}. "
                "I am using following code to extract the class from your response:\n\n"
                f"```python\n{src}\n```\n\n"
            )
            self.messages.append({
                "content": content,
                "role": "user",
                "end_of_turn": True,
            })
