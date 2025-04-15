from tklearn.kb.lexicon import Lexicon

# lexicon = Lexicon[str]()
# lexicon["cat"] = "animal"
# lexicon["catches"] = "verb"
# text = "The cat catches mice"
# for value, start, end in lexicon.extract(text):
#     print(f"Found {value} at {start}-{end}: {text[start:end]}")
# Found animal at 4-7: cat
# Found verb at 4-9: catch


# lexicon = Lexicon[str]()
# lexicon["py"] = "python"  # Add new keyword
# lexicon["python"] = "language"  # Add new keyword
# lexicon["python"] = "snake"  # Update existing keyword

# lexicon.display()
# py python
# python snake


lexicon = Lexicon[str]()
lexicon["cat"] = "animal"
lexicon["catch"] = "verb"
lexicon.display()
