import abc
import optuna
from optuna import Study
from typing import Callable

__all__ = [
    "Optimizer",
]


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def objective(self, *args, **kwargs):
        pass

    def optimize(self, n_trials: int = 100) -> None:
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        return study


class OptimizerFunction(Optimizer):
    def __init__(self, func: Callable) -> None:
        super(OptimizerFunction, self).__init__()
        self.func = func

    def objective(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def optimizer(func: Callable):
    return OptimizerFunction(func)
