from __future__ import annotations

import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Generic, Self

I, R = TypeVar("I"), TypeVar("R")  # noqa: E741


class Module(nn.Module, Generic[I, R]):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compile(
        self,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend: str | Callable[..., Any] = "inductor",
        mode: str | None = None,
        options: Dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ) -> Self:
        return torch.compile(
            self,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

    def freeze_layers(
        self, layers: Optional[List[str]] = None, prefix: str = ""
    ) -> int:
        """
        Freeze layers in the model that match the given patterns.

        Parameters
        ----------
        layers : list of str, optional
            A list of layer names or patterns to freeze. Supports wildcards (*)
            and dot notation for nested layers. If None, no layers will be frozen.
        prefix : str, default=""
            An optional prefix to apply to all layer patterns.

        Returns
        -------
        int
            The number of parameters frozen.

        Raises
        ------
        ValueError
            If an invalid regex pattern is provided.

        Examples
        --------
        >>> model.freeze_layers(['encoder.*', 'encoder.layer.[0-8].*'])
        >>> model.freeze_layers(['layer_[1-3]'], prefix='transformer')

        Notes
        -----
        This method uses regular expressions to match layer names. Dots in layer
        names are treated as literal dots, while asterisks are treated as wildcards.
        """
        if not layers:
            return 0  # no layers to freeze
        # escape dots and convert asterisks to regex wildcards
        layers = [p.replace(".", r"\.").replace("*", ".*") for p in layers]
        pattern_regex = "|".join(layers)
        if prefix:
            pattern_regex = f"{prefix}\.({pattern_regex})"
        # compile regex pattern
        try:
            pattern = re.compile(f"^{pattern_regex}$")
        except re.error as e:
            raise ValueError(str(e))
        # freeze parameters that match the pattern
        frozen_params = 0
        for name, param in self.named_parameters():
            if not pattern.match(name):
                continue
            param.requires_grad = False
            frozen_params += param.numel()
        # return number of frozen parameters
        return frozen_params

    def validation_step(
        self, batch: I, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Union[Tensor, Mapping[str, Any], None]:
        raise NotImplementedError

    def test_step(
        self, batch: I, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Union[Tensor, Mapping[str, Any], None]:
        raise NotImplementedError

    def training_step(
        self, batch: I, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Union[Tensor, Mapping[str, Any], None]:
        raise NotImplementedError

    def predict_step(
        self, batch: I, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> R:
        raise NotImplementedError

    def compute_loss(
        self, batch: I, output: R, **kwargs
    ) -> Union[Tensor, Mapping[str, Any], None]:
        raise NotImplementedError

    def compute_metric_inputs(self, batch: I, output: R, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
