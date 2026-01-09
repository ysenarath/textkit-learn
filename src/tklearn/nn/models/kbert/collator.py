from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tklearn.nn.models.kbert import KBertTokenizer

__all__ = [
    "KBertCollatorWithPadding",
]


@dataclass
class KBertCollatorWithPadding:
    tokenizer: KBertTokenizer

    def __call__(
        self, features: list[dict[str, list[int] | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_position_ids = []
        batch_visibility_matrix = []
        batch_labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            # Pad Input IDs
            input_ids = f["input_ids"].detach().clone().to(torch.long)
            if pad_len > 0:
                pad_seq = torch.full(
                    (pad_len,),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                input_ids = torch.cat([input_ids, pad_seq])

            # Pad Position IDs
            pos_ids = f["position_ids"].detach().clone().to(torch.long)
            if pad_len > 0:
                pos_ids = torch.cat([
                    pos_ids,
                    torch.zeros(pad_len, dtype=torch.long),
                ])

            # Pad Visibility Matrix (2D)
            vis_mat = f["visibility_matrix"].detach().clone().to(torch.long)
            if pad_len > 0:
                vis_mat = torch.nn.functional.pad(
                    vis_mat, (0, pad_len, 0, pad_len), value=0
                )

            # D. Collect Label
            batch_labels.append(f["label"])

            batch_input_ids.append(input_ids)
            batch_position_ids.append(pos_ids)
            batch_visibility_matrix.append(vis_mat)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "position_ids": torch.stack(batch_position_ids),
            "attention_mask": torch.stack(batch_visibility_matrix),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
