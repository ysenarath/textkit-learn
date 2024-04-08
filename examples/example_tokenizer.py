from tklearn.base.field import Text
from tklearn.preprocessing.tokenizer import WhiteSpaceTokenizer

if __name__ == "__main__":
    text = Text("The quick brown fox jumps over the lazy dog")
    tokenizer = WhiteSpaceTokenizer()
    tokenizer.tokenize(text)
    for token in text.tokens:
        print(token, end=" ")
    print()
    print("Tokenizer:", text.tokens.tokenizer)
