from typing import List

from tklearn.utils.flashtext import KeywordProcessor

__all__ = [
    "KeywordAnnotator",
]


class KeywordAnnotator:
    def __init__(self):
        self.kp = KeywordProcessor()

    def annotate(
        self, texts: str | List[str]
    ) -> List[List[tuple[str, int, int]]]:
        if isinstance(texts, str):
            texts = [texts]
        annotations = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    "annotate input must be a string or a list of strings"
                )
            annotations.append(self.kp.extract_keywords(text, span_info=True))
        return annotations


# if not self.path.with_suffix(".kp.pkl").exists():
#     kp = KeywordProcessor()
#     label2id = defaultdict(set)
#     vocab = self.get_vocab()
#     for id, lbl in tqdm.tqdm(vocab):
#         label2id[lbl].add(id)
#         kp.add_keyword(lbl)
#         if "-" in lbl:
#             new_label = lbl.replace("-", " ")
#             label2id[new_label].add(id)
#             kp.add_keyword(new_label)
#         if "_" in lbl:
#             new_label = lbl.replace("_", " ")
#             label2id[new_label].add(id)
#             kp.add_keyword(new_label)
#     # save the keyword processor
#     kp.metadata.update({
#         "label2id": label2id,
#     })
#     kp.dump(self.path.with_suffix(".kp.pkl"))
# self.keyword_processor = KeywordProcessor.load(
#     self.path.with_suffix(".kp.pkl")
# )
