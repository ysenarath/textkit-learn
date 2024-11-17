import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import FillMaskPipeline, pipeline

lemmatizer = WordNetLemmatizer()


def masked_lm_pipeline():
    return pipeline(model="google-bert/bert-base-uncased")


def create_masked_sentence(sentence: str, target_word: str, masked_token: str):
    tokens = word_tokenize(sentence)
    target_word = target_word.lower()
    for i, token in enumerate(tokens):
        if token.lower() == target_word:
            tokens[i] = masked_token
    return sentence


def generate_similar_examples(
    sentence, lemma, pipeline: FillMaskPipeline, top_k: int = 3
):
    masked_token = pipeline.tokenizer.mask_token
    masked_sentence = create_masked_sentence(sentence, lemma, masked_token)
    outputs = pipeline(masked_sentence, top_k=top_k)
    new_sentences = []
    for token in outputs:
        token_score = token["score"]
        predicted_word: str = token["token_str"]
    return new_sentences


def expand_lemmas(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    pipeline = masked_lm_pipeline()
    expanded_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lemma = row["lemma"]
        example = row["example"]
        pos = row["pos"]
        synset = row["synset"]
        new_examples = generate_similar_examples(
            example, lemma, pipeline, top_k=top_k
        )
        expanded_data.append(row)
        for new_example in new_examples:
            expanded_row = {
                "lemma": lemma,
                "example": new_example,
                "pos": pos,
                "synset": synset,
            }
            expanded_data.append(expanded_row)
    return pd.DataFrame(expanded_data)
