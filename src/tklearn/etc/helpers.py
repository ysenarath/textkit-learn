from transformers import AutoTokenizer


class SubwordDetector:
    def __init__(self, tokenizer: AutoTokenizer):
        prefix = ""
        text = "thequickbrownfoxjumpsover thelazydog"
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for token in raw_tokens:
            if text.startswith(token):
                text = text[len(token) :].strip()
                continue
            for i in range(len(token)):
                if text.startswith(token[i:]):
                    prefix = token[:i]
                    break
        self.prefix = prefix
        # if true "word" else "subword"
        self.prefix_type = text[0].startswith("t")

    def is_prefix(self, token: str, is_start_token: bool = False) -> bool:
        if is_start_token:
            # start token is always a word
            return False
        if self.prefix_type:
            return not token.startswith(self.prefix)
        return token.startswith(self.prefix)
