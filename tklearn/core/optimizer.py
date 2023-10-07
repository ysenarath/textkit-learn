from typing import Callable
import optuna
from optuna import Study


class Optimizer(object):
    def __init__(self, study: Study) -> None:
        self._study = study

    def run(self, n_trials: int = 100) -> None:
        self._study.optimize(self._objective, n_trials=n_trials)
