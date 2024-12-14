from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from typing_extensions import Literal


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
