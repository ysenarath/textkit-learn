from typing import Any, List, Tuple, Union

from tklearn.core.trainer import Predictor
from tklearn.utils.array import concat, move_to_device, to_numpy

InputDataType = Union[Tuple[List[Any], List[Any]], List[Any]]


def batch_generator(data: List[dict], batch_size: int = 32):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class Evaluator:
    def __init__(
        self,
        data: InputDataType,
        metric,
        batch_size: int = 32,
    ):
        if not isinstance(data, tuple):
            data = (data, None)
        self._input, self._y_true = data
        self.metric = metric
        self.batch_size = batch_size
        self._y_pred = None

    def evaluate(self, model: Predictor):
        if not isinstance(model, Predictor):
            msg = (
                f"model must be a predictor, but got '{type(model).__name__}'"
            )
            raise ValueError(msg)
        for batch in batch_generator(self._input, batch_size=self.batch_size):
            y_pred = model.predict(batch)
            # move y_pred to cpu and convert to numpy
            y_pred = to_numpy(move_to_device(y_pred, "cpu"))
            if self._y_pred is None:
                self._y_pred = y_pred
            else:
                self._y_pred = concat([self._y_pred, y_pred], axis=0)
