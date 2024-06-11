from typing import Generator, List, Optional, Sequence, Union

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

__all__ = [
    "MultipleOptimizers",
    "OptimizerBuilder",
    "configure_optimizers",
    "MultipleLRSchedulers",
    "configure_lr_schedulers",
]

ParamLike = Union[torch.nn.Module, List[torch.nn.Parameter]]
OptimizerLike = Union[Optimizer, List[Optimizer]]


class MultipleOptimizers(Sequence[Optimizer]):
    def __init__(self, *optims: List[Optimizer]):
        self.optims: List[Optimizer] = optims

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def step(self):
        for optim in self.optims:
            optim.step()

    def __getitem__(self, index: int) -> Optimizer:
        return self.optims[index]

    def __len__(self) -> int:
        return len(self.optims)

    def __iter__(self) -> Generator[Optimizer, None, None]:
        return iter(self.optims)


class OptimizerBuilder:
    def __init__(self, config: Union[dict, str]):
        if isinstance(config, str):
            config = {"@type": config}
        elif isinstance(config, list):
            config = [OptimizerBuilder(c) for c in config]
        elif not isinstance(config, (dict, Optimizer)):
            msg = f"invalid optimizer config: {config}"
            raise ValueError(msg)
        self.config: Union[dict, List[OptimizerBuilder]] = config

    def build(self, params: ParamLike) -> OptimizerLike:
        if isinstance(self.config, list):
            optimizers = []
            for builder in self.config:
                optimizer = builder.build(params)
                optimizers.append(optimizer)
            return MultipleOptimizers(*optimizers)
        elif isinstance(self.config, Optimizer):
            return self.config
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        optimizer_type = getattr(torch.optim, self.config["@type"])
        optimizer_args = {k: v for k, v in self.config.items() if not k.startswith("@")}
        return optimizer_type(params, **optimizer_args)


def configure_optimizers(params: ParamLike, config: Union[dict, list]) -> OptimizerLike:
    if isinstance(params, torch.nn.Module):
        params = params.parameters()
    return OptimizerBuilder(config).build(params)


class MultipleLRSchedulers(Sequence[LRScheduler]):
    def __init__(self, *lrs: List[LRScheduler]):
        self.lrs: List[LRScheduler] = lrs

    def step(self):
        for lr in self.lrs:
            lr.step()

    def __getitem__(self, index: int) -> LRScheduler:
        return self.lrs[index]

    def __len__(self) -> int:
        return len(self.lrs)

    def __iter__(self) -> Generator[LRScheduler, None, None]:
        return iter(self.lrs)


def configure_lr_schedulers(
    optimizer: Optimizer,
    config: Union[dict, list, str, LRScheduler, None],
) -> Optional[LRScheduler]:
    if config is None:
        return None
    if isinstance(config, LRScheduler):
        return config
    if isinstance(config, str):
        lr_scheduler_type = getattr(torch.optim.lr_scheduler, config)
        return lr_scheduler_type(optimizer)
    if isinstance(config, list):
        lr_schedulers = []
        for c in config:
            lr_scheduler = configure_lr_schedulers(optimizer, c)
            lr_schedulers.append(lr_scheduler)
        return MultipleLRSchedulers(lr_schedulers)
    if isinstance(config, dict):
        lr_scheduler_type = getattr(torch.optim.lr_scheduler, config["@type"])
        lr_scheduler_args = {k: v for k, v in config.items() if not k.startswith("@")}
        return lr_scheduler_type(optimizer, **lr_scheduler_args)
    msg = f"invalid lr_scheduler config: {config}"
    raise ValueError(msg)
