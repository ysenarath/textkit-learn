import functools
import warnings
from collections import UserList
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from typing_extensions import Self

__all__ = [
    "Callback",
    "CallbackList",
]


if TYPE_CHECKING:
    from tklearn.nn.module import Module
else:
    Module = TypeVar("Module")


class Callback:
    def __init__(self, *args, **kwargs) -> None:
        self._model = None
        self._params = {}
        super().__init__(*args, **kwargs)

    @property
    def model(self) -> Module:
        """
        Get the model associated with the callback.

        Returns
        -------
        Model
            The model associated with the callback.
        """
        return self._model

    def set_model(self, model: Module):
        """
        Set the model associated with the callback.

        Parameters
        ----------
        model : Model
            The model to be set.
        """
        if model is not None and self._model is not None and model is not self._model:
            msg = (
                f"the callback '{type(self).__name__}' is already "
                f"associated with a model '{type(self._model).__name__}'"
            )
            raise ValueError(msg)
        self._model = model

    @property
    def params(self) -> dict:
        """
        Get the parameters of the callback.

        Returns
        -------
        dict
            The parameters of the callback.
        """
        return self._params

    def set_params(self, params):
        """
        Set the parameters of the callback.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        if params is None:
            params = {}
        self._params = params

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of an epoch.

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method but
            that may change in the future.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict, optional
            Metric results for this training epoch, and for the validation
            epoch if validation is performed.
            Validation result keys are prefixed with `val_`.
            For the training epoch, the values of the `Model`'s metrics are
            returned.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        pass

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method
            but that may change in the future.
        """
        pass

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Aggregated metric results up until this batch.
        """
        pass

    def on_test_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method
            but that may change in the future.
        """
        pass

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Aggregated metric results up until this batch.
        """
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method
            but that may change in the future.
        """
        pass

    def on_predict_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict, optional
            Aggregated metric results up until this batch.
        """
        pass

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method
            but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently the output of the last call to `on_epoch_end()` is
            passed to this argument for this method but that may change
            in the future.
        """
        pass

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but
            that may change in the future.
        """
        pass

    def on_test_end(self, logs=None):
        """
        Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently the output of the last call to `on_test_batch_end()` is
            passed to this argument for this method but that may change in the
            future.
        """
        pass

    def on_predict_begin(self, logs=None):
        """
        Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but
            that may change in the future.
        """
        pass

    def on_predict_end(self, logs=None):
        """
        Called at the end of prediction.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but
            that may change in the future.
        """
        pass


def is_callback(name: str) -> bool:
    return name.startswith("set_") or name.startswith("on_")


class CallbackList(Callback, Sequence[Callback]):
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        super().__init__()
        self._callbacks: List[Callback] = []
        self.extend(callbacks)

    def extend(
        self,
        callbacks: Optional[List[Callback]],
        /,
        inplace: bool = True,
    ) -> Self:
        inst = self if inplace else self.copy()
        if callbacks is None:
            return
        for callback in callbacks:
            inst.append(callback)
        return inst

    def append(self, callback: Callback) -> None:
        callback.set_model(self.model)
        callback.set_params(self.params)
        if not isinstance(callback, Callback):
            msg = (
                "callback must be an instance of "
                f"'{Callback.__name__}', not "
                f"'{callback.__class__.__name__}'."
            )
            raise TypeError(msg)
        self._callbacks.append(callback)

    def __getitem__(self, item) -> Callback:
        return self._callbacks[item]

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self._callbacks)

    def __getattribute__(self, __name: str) -> Any:
        if is_callback(__name):
            return functools.partial(self.apply, __name)
        else:
            return super().__getattribute__(__name)

    def apply(self, __name: str, /, *args, **kwargs) -> UserList:
        # call on self
        super().__getattribute__(__name)(*args, **kwargs)
        # call on each callback
        outputs = UserList()
        outputs.errors = []
        for callback in self:
            output = None
            outputs.errors.append(None)
            try:
                # print(callback, func, args, kwargs)
                output = getattr(callback, __name)(*args, **kwargs)
            except NotImplementedError:
                pass
            except Exception as e:
                warnings.warn(
                    f"{e} in callback '{type(callback).__name__}' "
                    f"when calling '{__name}'",
                    RuntimeWarning,
                    stacklevel=0,
                    source=e,
                )
                outputs.errors[-1] = e
            outputs.append(output)
        return outputs

    def copy(self) -> Self:
        """Return a shallow copy of the list."""
        return self.__class__(self._callbacks.copy())
