# Knowledge base

Examples of `tklearn.kb`. Each one prints what it finds, then checks it
and prints a line per check.

| File | Shows |
|---|---|
| [`custom_store.py`](custom_store.py) | A small store of your own: `write_tables`, `TableStore` and `KnowledgeBase`, with relations read from either side. Runs offline |
| [`look_up_words.py`](look_up_words.py) | The senses of a word in Wiktionary and the relations of one sense: synonyms, hypernyms and the hyponyms they imply |
| [`augment_text.py`](augment_text.py) | `augment`: variants of a sentence with synonyms or hypernyms, and narrowed to one sense of a word |
| [`disambiguate_senses.py`](disambiguate_senses.py) | Picking the sense of "bank" that fits a sentence, with the gloss embeddings and `OpenRouterEmbedding` |

## Running

```sh
python examples/kb/custom_store.py
python examples/kb/look_up_words.py
python examples/kb/augment_text.py
OPENROUTER_API_KEY=... python examples/kb/disambiguate_senses.py
```

The Wiktionary examples download the tables from the Hugging Face Hub
dataset [`textkit-learn/wiktionary-v0.5`](https://huggingface.co/datasets/textkit-learn/wiktionary-v0.5)
the first time (about 170 MB, or 6.8 GB with the embeddings for
`disambiguate_senses.py`) into `~/.cache/tklearn/assets/wiktionary-v0.5`.
`custom_store.py` writes its tables to `examples/outputs/kb/`.
