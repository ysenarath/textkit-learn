from typing import TYPE_CHECKING, Generic, Type, TypeVar

from tklearn.core.callback import BaseCallback, BaseCallbackList

__all__ = [
    "TrainerCallback",
    "TrainerCallbackList",
]

if TYPE_CHECKING:
    from tklearn.nn.torch import TorchTrainer


class TrainerCallback(BaseCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trainer = None
        self._params = {}

    def set_trainer(self, trainer: "TorchTrainer"):
        """
        Set the trainer associated with the callback.

        Parameters
        ----------
        trainer : Trainer
            The model to be set.
        """
        self._trainer = trainer

    @property
    def trainer(self) -> "TorchTrainer":
        """
        Get the trainer associated with the callback.

        Returns
        -------
        Trainer
            The trainer associated with the callback.
        """
        return self._trainer

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

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of an epoch.

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method but that may change in the future.
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
            Metric results for this training epoch, and for the validation epoch if validation is performed.
            Validation result keys are prefixed with `val_`.
            For the training epoch, the values of the `Model`'s metrics are returned.
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
            Currently no data is passed to this argument for this method but that may change in the future.
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
            Currently no data is passed to this argument for this method but that may change in the future.
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
            Currently no data is passed to this argument for this method but that may change in the future.
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
            Currently no data is passed to this argument for this method but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently the output of the last call to `on_epoch_end()` is passed to this argument for this
            method but that may change in the future.
        """
        pass

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but that may change in the future.
        """
        pass

    def on_test_end(self, logs=None):
        """
        Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently the output of the last call to `on_test_batch_end()` is passed to this argument for
            this method but that may change in the future.
        """
        pass

    def on_predict_begin(self, logs=None):
        """
        Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but that may change in the future.
        """
        pass

    def on_predict_end(self, logs=None):
        """
        Called at the end of prediction.

        Subclasses should override for any actions to run.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method but that may change in the future.
        """
        pass


class TrainerCallbackList(TrainerCallback, BaseCallbackList):
    @classmethod
    def get_callback_type(cls) -> Type[BaseCallback]:
        return TrainerCallback
