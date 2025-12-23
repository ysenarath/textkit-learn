from __future__ import annotations

from collections import defaultdict
from itertools import tee
from typing import Generator

import numpy as np
from transformers import PreTrainedTokenizer

from tklearn.kb.base import KnowledgeBase
from tklearn.kb.models import Mention, Span, Triple
from tklearn.nn.models.kbert.helpers import Injection, inject
from tklearn.plotting.token_tree import TokenTree


def iter2dict(items: list[dict]) -> dict:
    batch = defaultdict(list)
    keys = set()
    it0, it1 = tee(items)
    for item in it0:
        keys.update(item.keys())
        break
    for item in it1:
        for k in keys:
            batch[k].append(item[k])
    return dict(batch)


def dict2iter(batch: dict) -> Generator[dict, None, None]:
    batch_size = len(next(iter(batch.values())))
    for i in range(batch_size):
        item = {}
        for k, v in batch.items():
            item[k] = v[i]
        yield item


def get_token_aligned_triples(item: dict) -> dict:
    if "triples" not in item or not item["triples"]:
        return []
    text = item["text"]
    offset_mapping = item["offset_mapping"]
    # build char to token index mapping
    char2token = np.zeros(len(text), dtype=np.int32) - 1
    previous_end_char = 0
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        for char_idx in range(previous_end_char, end_char):
            char2token[char_idx] = token_idx
        previous_end_char = end_char
    # normal tokens can see each other
    triples = []
    for triple in item["triples"]:
        # get the token indexes for the mention span
        mention_start_char = triple["mention.span.start"]
        mention_end_char = triple["mention.span.end"]
        mention_start_token = char2token[mention_start_char]
        mention_end_token = char2token[mention_end_char - 1] + 1
        # get the token indexes for the triple span
        triple_start_char = triple["triple.span.start"]
        triple_end_char = triple["triple.span.end"]
        triple_start_token = char2token[triple_start_char]
        triple_end_token = char2token[triple_end_char - 1] + 1
        triples.append({
            "triple.subject": triple["triple.subject"],
            "triple.predicate": triple["triple.predicate"],
            "triple.object": triple["triple.object"],
            "mention.span.start": mention_start_token.item(),
            "mention.span.end": mention_end_token.item(),
            "triple.span.start": triple_start_token.item(),
            "triple.span.end": triple_end_token.item(),
            "mention.span.start.char": mention_start_char,
            "mention.span.end.char": mention_end_char,
            "triple.span.start.char": triple_start_char,
            "triple.span.end.char": triple_end_char,
            "span.type": "token",
            "mention_str": triple["mention_str"],
            "triple_str": triple["triple_str"],
            "mention_tokens": item["tokens"][
                mention_start_token:mention_end_token
            ],
            "triple_tokens": item["tokens"][
                triple_start_token:triple_end_token
            ],
        })
    return triples


def extract_visibility_matrix(item: dict):
    input_ids = item["input_ids"]
    seq_len = len(input_ids)
    is_normal_token = np.ones(seq_len, dtype=bool)
    visibility_matrix = np.zeros((seq_len, seq_len), dtype=np.int8)
    for triple in item["triples"]:
        if triple["span.type"] != "token":
            raise ValueError(
                "Triples must be token-aligned to extract visibility matrix."
            )
        # mention is the parent of the triple tokens
        mention_start_token = triple["mention.span.start"]
        mention_end_token = triple["mention.span.end"]
        triple_start_token = triple["triple.span.start"]
        triple_end_token = triple["triple.span.end"]
        # set visibility between mention tokens and triple tokens
        for tt in range(triple_start_token, triple_end_token):
            is_normal_token[tt] = False
            for mt in range(mention_start_token, mention_end_token):
                visibility_matrix[mt, tt] = 1
                visibility_matrix[tt, mt] = 1
    # set visibility among normal tokens
    normal_token_indexes = np.where(is_normal_token)[0]
    for ti in normal_token_indexes:
        for tj in normal_token_indexes:
            visibility_matrix[ti, tj] = 1
    return visibility_matrix


def extract_token_tree(tokenizer: KnowledgeBaseTokenizer, item: dict):
    input_ids = item["input_ids"]
    tree = TokenTree()
    for input_id in input_ids:
        text = tokenizer.convert_ids_to_tokens(input_id)
        tree.append(text)
    # mark parent nodes for triples by setting parent for each token in tree
    tree.head = 0
    tree.tail = len(input_ids) - 1
    for triple in item["triples"]:
        if triple["span.type"] != "token":
            raise ValueError(
                "Triples must be token-aligned to extract token tree."
            )
        mention_end_token = triple["mention.span.end"] - 1
        tr_ = input_ids[mention_end_token]
        if mention_end_token is None:
            raise ValueError("Mention end token is None.")
        triple_start_token = triple["triple.span.start"]
        assert mention_end_token is not None, "mention_end_token is None"
        # triplet start should be the child of the mention end token
        tree.nodes[triple_start_token]["parent"] = mention_end_token
        # triplet end + 1 should be the child of the mention end token
        triple_end_token = triple["triple.span.end"]
        if triple_end_token < len(input_ids):
            tree.nodes[triple_end_token]["parent"] = mention_end_token
    tree.build()
    return tree


def extract_soft_position_index(tree: TokenTree) -> np.ndarray:
    soft_position_index = np.zeros(len(tree), dtype=np.int32)
    for idx, item in enumerate(tree):
        soft_position_index[idx] = item["depth"]
    return soft_position_index


class KnowledgeBaseTokenizer:
    tokenizer: PreTrainedTokenizer
    knowledge_base: KnowledgeBase
    predicates: set[str]

    def __init__(
        self, tokenizer: PreTrainedTokenizer, knowledge_base: KnowledgeBase
    ):
        self.tokenizer = tokenizer
        self.knowledge_base = knowledge_base
        self.k: int | None = 2  # top-k triples per mention
        self.predicates = {
            "synonym",
            "antonym",
            "hypernym",
            "hyponym",
            # "category",
        }

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def extract_mentions(self, *, text: str) -> list[Mention]:
        return self.knowledge_base.extract_mentions(text)

    def filter_triples(
        self,
        *,
        relations: list[Triple],
        text: str | None = None,
        mention: Mention | None = None,
    ) -> list[Triple]:
        # filter candidate triples based on some criteria
        return relations

    def extract_triples(
        self, *, text: str
    ) -> tuple[list[tuple[str, str, str]], list[Span]]:
        mentions = self.extract_mentions(text=text)
        collection = ([], [])  # triples, mention_spans
        for mention in mentions:
            curr_mention_triples = {}
            for candidate in mention.candidates:
                relations = self.knowledge_base.extract_relations(
                    candidate, predicates=self.predicates
                )
                candidate_triples = self.filter_triples(
                    relations=relations,
                    text=text,
                    mention=mention,
                )
                for triple in candidate_triples:
                    sw = triple.subject[0]
                    ow = triple.object[0]
                    pred = triple.predicate
                    # avoid duplicate triples for the same mention
                    curr_mention_triples[(sw, pred, ow)] = mention.span
            if self.k is not None:
                # keep only top-k triples by span length
                curr_mention_triples = sorted(
                    curr_mention_triples.items(),
                    key=lambda x: x[1].end - x[1].start,
                    reverse=True,
                )
                curr_mention_triples = dict(curr_mention_triples[: self.k])
            for triple, span in curr_mention_triples.items():
                collection[0].append(triple)
                collection[1].append(span)
        return collection

    def augment_text_with_triples(self, batch: dict) -> dict:
        batch_items = []
        for item in dict2iter(batch):
            item_text: str = item["text"]
            triple_spo, mention_spans = self.extract_triples(text=item_text)
            original_mention_strs = []
            for mention_span in mention_spans:
                original_mention_strs.append(
                    item_text[mention_span.start : mention_span.end]
                )
            injections = []
            for _spo, _span in zip(triple_spo, mention_spans):
                injection_text = " {} {} ".format(_spo[1], _spo[2])
                injection_position = _span.end
                injections.append(
                    Injection(injection_text, injection_position)
                )
            augmented_text, mention_spans, triple_spans = inject(
                item_text, injections, mention_spans
            )
            triples = []
            for i in range(len(mention_spans)):
                mention_span = mention_spans[i]
                triple_span = triple_spans[i]
                triple_str = augmented_text[
                    triple_span.start : triple_span.end
                ]
                original_mention_str_augmented = augmented_text[
                    mention_span.start : mention_span.end
                ]
                assert (
                    original_mention_strs[i] == original_mention_str_augmented
                ), "Original mention string does not match after augmentation."
                triples.append(({
                    "triple.subject": triple_spo[i][0],
                    "triple.predicate": triple_spo[i][1],
                    "triple.object": triple_spo[i][2],
                    "mention.span.start": mention_span.start,
                    "mention.span.end": mention_span.end,
                    "triple.span.start": triple_span.start,
                    "triple.span.end": triple_span.end,
                    "mention_str": original_mention_strs[i],
                    "triple_str": triple_str,
                    "span.type": "char",
                }))
            item["text"] = augmented_text
            item["original_text"] = item_text
            item["triples"] = triples
            batch_items.append(item)
        return batch_items

    def tokenize(self, batch: dict) -> dict:
        if not isinstance(batch, dict):
            batch = iter2dict(batch)
        data = self.tokenizer(batch["text"], return_offsets_mapping=True)
        batch.update(data)
        # update the augnmented triples to use token indexes (if any)
        triples_list = []
        for item in dict2iter(batch):
            item["tokens"] = [
                self.tokenizer.convert_ids_to_tokens(tid)
                for tid in item["input_ids"]
            ]
            triples = get_token_aligned_triples(item)
            triples_list.append(triples)
        batch["triples"] = triples_list
        return batch

    def extract_features(self, batch: dict) -> dict:
        visibility_matrices = []
        soft_position_indexes = []
        tree_data = []
        # we have to set the visibility matrix according to the injected triples
        for i, item in enumerate(dict2iter(batch)):
            visibility_matrix = extract_visibility_matrix(item)
            visibility_matrices.append(visibility_matrix)
            # plot visibility_matrices in here.parent / "outputs" / "visibility_matrices"
            # filename = here / "outputs" / "visibility_matrix_{}.png".format(i)
            # fig = plot_dot_matrix(visibility_matrix)
            # fig.savefig(filename)
            tree = extract_token_tree(self, item)
            # add tree to data -- you can use TokenTree.loads to load this back
            tree_data.append(tree.dumps())
            soft_position_index = extract_soft_position_index(tree)
            soft_position_indexes.append(soft_position_index)
            # save plot of the tree for debugging to here.parent / "outputs" / "token_trees"
            # filename = here / "outputs" / "token_tree_{}".format(i)
            # tree.graphviz().render(filename, view=False, format="png")
        batch["visibility_matrix"] = visibility_matrices
        batch["position_ids"] = soft_position_indexes
        batch["token_type_ids"] = batch.get("token_type_ids", None)
        batch["token_tree"] = tree_data
        return batch

    def __call__(self, batch: dict) -> dict:
        batch = self.augment_text_with_triples(batch)
        batch = self.tokenize(batch)
        batch = self.extract_features(batch)
        return batch
