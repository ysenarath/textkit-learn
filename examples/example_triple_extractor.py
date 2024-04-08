import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT_TEMPLATE = """Instruction: Extract the list of all consistent facts [fact1, fact2, fact3, ...] from the given text. Each fact is represented as a triple ("subject", "relation", "object").

Input: {input}

Output: """


class TripletExtractor:
    def __init__(self) -> None:
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
        self.prompt_template = PROMPT_TEMPLATE

    def extract(self, input: str) -> tuple[str, str, str]:
        model_input = self.prompt_template.format(input=input)
        input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            max_length=200,
            temperature=1e-5,
            do_sample=True,
        )
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs
