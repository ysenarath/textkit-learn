from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import required
from typing_extensions import Self, TypeAlias

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


ParamLike = Union[torch.nn.Module, List[torch.nn.Parameter]]


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)


SCHEDULES = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
}

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


class BERTAdamW(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.

    Parameters
    ----------
    params : ParamsT
        Iterable of parameters to optimize or dicts defining parameter groups
    lr : float, optional
        Learning rate, by default required
    warmup : float, optional
        Portion of t_total for the warmup, -1 means no warmup, by default -1
    t_total : int, optional
        Total number of training steps for the learning rate schedule, -1 means
        constant learning rate, by default -1
    schedule : str, optional
        Schedule to use for the warmup, by default 'warmup_linear'
    b1 : float, optional
        Adams b1, by default 0.9
    b2 : float, optional
        Adams b2, by default 0.999
    e : float, optional
        Adams epsilon, by default 1e-6
    weight_decay : float, optional
        Weight decay, by default 0.01
    max_grad_norm : float, optional
        Maximum norm for the gradients (-1 means no clipping), by default 1.0
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = required,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = "warmup_linear",
        b1: float = 0.9,
        b2: float = 0.999,
        e: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        if lr is not required and lr < 0.0:
            msg = f"invalid learning rate: {lr} - should be >= 0.0"
            raise ValueError(msg)
        if schedule not in SCHEDULES:
            msg = f"invalid schedule parameter: {schedule}"
            raise ValueError(msg)
        if not 0.0 <= warmup < 1.0 and warmup != -1:
            msg = f"invalid warmup: {warmup} - should be in [0.0, 1.0[ or -1"
            raise ValueError(msg)
        if not 0.0 <= b1 < 1.0:
            msg = f"invalid b1 parameter: {b1} - should be in [0.0, 1.0)"
            raise ValueError(msg)
        if not 0.0 <= b2 < 1.0:
            msg = f"invalid b2 parameter: {b2} - should be in [0.0, 1.0)"
            raise ValueError(msg)
        if not e >= 0.0:
            msg = f"invalid epsilon value: {e} - should be >= 0.0"
            raise ValueError(msg)
        defaults = {
            "lr": lr,
            "schedule": schedule,
            "warmup": warmup,
            "t_total": t_total,
            "b1": b1,
            "b2": b2,
            "e": e,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        }
        super(BERTAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BERTAdamW, self).__setstate__(state)

    def step(self, closure=None, type=None, t=None, mask_back=None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss, by default None
        type : str, optional
            Type of gradient masking, by default None
        t : int, optional
            Current training step, by default None
        mask_back : dict, optional
            Masking for the backward pass, by default None

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p_id, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    msg = (
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["next_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["e"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.

                if type == "mask" and t > 0:
                    # adam may change the grad
                    # Restrict layer gradients in backprop
                    n = group["name"][p_id]
                    if n in mask_back:
                        update *= mask_back[n]

                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                # Update the step
                state["step"] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

    def plot_lr(self):
        """Plot the learning rate schedule."""
        data = []
        for i, group in enumerate(self.param_groups):
            t_total = group["t_total"]
            if t_total == -1:
                for x in range(1000):
                    y = group["lr"]
                    data.append({
                        "step": x / 1000,
                        "lr": y,
                        "group": i,
                    })
            else:
                lr = group["lr"]
                warmup = group["warmup"]
                schedule_fct = SCHEDULES[group["schedule"]]
                for x in range(t_total):
                    y = lr * schedule_fct(x / t_total, warmup)
                    data.append({
                        "step": x,
                        "lr": y,
                        "group": i,
                    })
        data = pd.DataFrame(data)
        sns.lineplot(data=data, x="step", y="lr", hue="group", palette="viridis")
        # rename the axis
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        # add a legend
        plt.legend(title="Group")
        # show the plot
        plt.show()
