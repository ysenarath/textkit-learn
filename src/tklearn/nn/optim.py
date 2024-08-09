from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from typing_extensions import Self

__all__ = [
    "OptimizerList",
    "LRSchedulerList",
]

I, R = TypeVar("I"), TypeVar("R")  # noqa: E741


class LRSchedulerConfigDict(TypedDict):
    scheduler: LRScheduler
    interval: str
    frequency: int
    monitor: str
    strict: bool
    name: Optional[str]


@dataclass
class LRSchedulerConfig(Mapping[str, Any]):
    scheduler: LRScheduler
    interval: str = "epoch"
    frequency: int = 1
    monitor: str = "val_loss"
    strict: bool = True
    name: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def from_dict(self, data: LRSchedulerConfigDict) -> Self:
        return LRSchedulerConfig(**data)


if TYPE_CHECKING:
    LRSchedulerConfig = LRSchedulerConfig | LRSchedulerConfigDict


class OptimizerAndLRSchedulerDict(TypedDict):
    optimizer: Optimizer
    lr_scheduler: Union[LRScheduler, LRSchedulerConfig, LRSchedulerConfigDict]


class OptimizerList(Sequence[Optimizer]):
    def __init__(self, *optims: List[Optimizer]):
        self.optims: List[Optimizer] = list(optims)

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


ParamLike = Union[torch.nn.Module, List[torch.nn.Parameter]]


class LRSchedulerList(Sequence[LRScheduler]):
    def __init__(self, *lrs: List[LRScheduler]):
        self.lrs: List[LRScheduler] = list(lrs)

    def step(self):
        for lr in self.lrs:
            lr.step()

    def __getitem__(self, index: int) -> LRScheduler:
        return self.lrs[index]

    def __len__(self) -> int:
        return len(self.lrs)

    def __iter__(self) -> Generator[LRScheduler, None, None]:
        return iter(self.lrs)


def extract_optimizer_and_lr_scheduler(
    optimizer_config: Union[
        OptimizerAndLRSchedulerDict,
        Tuple[
            Iterable[Optimizer],
            Iterable[Union[LRScheduler, LRSchedulerConfig]],
        ],
        Iterable[Optimizer],
        Optimizer,
        None,
    ],
) -> Tuple[Optimizer, LRSchedulerConfig]:
    if isinstance(optimizer_config, Mapping):
        optimizer = optimizer_config["optimizer"]
        lr_scheduler = optimizer_config["lr_scheduler"]
    elif isinstance(optimizer_config, tuple):
        optimizer, lr_scheduler = optimizer_config
    elif isinstance(optimizer_config, Iterable):
        optimizer, lr_scheduler = list(optimizer_config), None
    else:
        optimizer, lr_scheduler = optimizer_config, None
    if isinstance(optimizer, Iterable):
        optimizer = OptimizerList(optimizer)
    if lr_scheduler is None:
        return optimizer, None
    if not isinstance(lr_scheduler, LRSchedulerConfig):
        lr_scheduler_config = LRSchedulerConfig(scheduler=lr_scheduler)
    if isinstance(lr_scheduler_config.scheduler, Iterable):
        lr_scheduler.scheduler = LRSchedulerList(lr_scheduler)
    return optimizer, lr_scheduler_config
