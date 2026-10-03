from __future__ import annotations

import adapters
from transformers import AutoModel, PreTrainedModel

from tklearn.nn.models.backbone.transformer import TransformerBackbone

__all__ = [
    "AdapterBackbone",
]


class AdapterBackbone(TransformerBackbone):
    """A transformer with a trainable adapter; the base weights are frozen.

    Parameters
    ----------
    model_name_or_path : str
        Hub id or local path of the base transformer.
    adapter : str or dict
        Adapter configuration accepted by ``adapters``' ``add_adapter``
        (e.g. ``"seq_bn"`` or a config dict).
    adapter_name : str, default="default"
        Name of the adapter inside the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter: str | dict,
        adapter_name: str = "default",
    ) -> None:
        self.adapter = adapter
        self.adapter_name = adapter_name
        super().__init__(model_name_or_path)

    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        model = AutoModel.from_pretrained(model_name_or_path)
        adapters.init(model)
        model.add_adapter(self.adapter_name, config=self.adapter)
        model.set_active_adapters(self.adapter_name)
        # train only the adapter; model.freeze_model(False) unfreezes the rest
        model.train_adapter(self.adapter_name, train_embeddings=False)
        return model
