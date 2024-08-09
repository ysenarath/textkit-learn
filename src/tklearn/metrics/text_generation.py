import math
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction


def display(references, candidates, similarity_matrix):
    for idx_i, sentence1 in enumerate(references):
        print(sentence1)
        for idx_j, sentence2 in enumerate(candidates):
            print(f" - {sentence2: <30}: {similarity_matrix[idx_i][idx_j]:.4f}")
        print()


class RankedSentenceBERTScore:
    def __init__(self, model_name_or_path="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name_or_path)
        self.model.similarity_fn_name = SimilarityFunction.COSINE

    def __call__(
        self,
        references: List[str],
        candidates: List[str],
        scale: float = 1.0,
        return_scores: bool = False,  # weighted_similarity_matrix
    ) -> dict:
        weights = np.zeros((len(references), len(candidates)))
        for i in range(len(references)):
            for j in range(len(candidates)):
                # if i == 0 and j == 0 => (i - j) ** 2 = 0 ** 2 = 0
                # if i == 0 and j == 1 => (i - j) ** 2 = 1 ** 2 = 1
                # if i == 0 and j == 2 => (i - j) ** 2 = 2 ** 2 = 4
                # if i == 1 and j == 0 => (i - j) ** 2 = 1 ** 2 = 1
                # if i == 1 and j == 1 => (i - j) ** 2 = 0 ** 2 = 0
                # if i == 1 and j == 2 => (i - j) ** 2 = 1 ** 2 = 1
                # (i - j) ** 2 >= 0
                # zero iff if i == j
                # weights[i, j] = 1 / math.exp(abs(i - j))
                weights[i, j] = math.exp(-1 * ((i - j) ** 2) / scale) / scale
        ref_embeddings = self.model.encode(references, convert_to_tensor=True)
        cand_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        scores = self.model.similarity(ref_embeddings, cand_embeddings)
        scores = scores.cpu().numpy()
        scores = scores * weights
        row_max = scores.max(axis=1)
        assert len(references) == len(row_max)
        recall = row_max.mean()
        col_max = scores.max(axis=0)
        assert len(candidates) == len(col_max)
        precision = col_max.mean()
        f1 = 2 * precision * recall / (precision + recall)
        if return_scores:
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "scores": scores,
            }
        return {"precision": precision, "recall": recall, "f1": f1}


def test_sentence_bert_score():
    # ground truth
    references = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
    ]
    # prediction
    candidates = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
    ]
    # calculate sentence bert score
    sent_bert_score = RankedSentenceBERTScore()
    report = sent_bert_score(references, candidates)
    print(report)


if __name__ == "__main__":
    test_sentence_bert_score()
