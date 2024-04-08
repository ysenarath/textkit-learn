import itertools
from typing import List

from tokenizers import Encoding


def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
    tokens = tokenized.tokens
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):
            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

        else:
            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
    return aligned_labels


class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            label = f"{s}-{label}"
            self.labels_to_id[label] = num
            self.ids_to_label[num] = label
        # Add the OUTSIDE label - no label for the token

    def get_aligned_label_ids_from_annotations(self, tokens, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokens, annotations)
        return list(map(self.labels_to_id.get, raw_labels))


EMOTION_LABELS = LabelSet([
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
])

SENTIMENT_LABELS = LabelSet(["positive", "negative", "neutral"])


from tqdm import auto as tqdm
from flashtext import KeywordProcessor

from tklearn.kb.nrc.emolex.io import EmoLexIO


class EmoLex:
    def __init__(self):
        io = EmoLexIO()
        io.download(exist_ok=True, unzip=True)
        self.emolex = list(io.load())
        self.emolex_processor = KeywordProcessor()
        for i, doc in enumerate(tqdm.tqdm(self.emolex)):
            self.emolex_processor.add_keyword(doc["label"], i)

    def annotate(self, text: str):
        return [
            {"start": ki, "end": ke, "label": self.emolex[ki]["emotions"]}
            for ki, ks, ke in self.emolex_processor.extract_keywords(
                text, span_info=True
            )
        ]
