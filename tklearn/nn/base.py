import abc
from typing import Any

from tklearn.nn.model import ModelBuilder, AutoModelBuilder
from tklearn.nn.callbacks import TrainerCallbackList

__all__ = [
    "BaseTrainer",
]


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        model,
        callbacks=None,
        verbose=True,
    ):
        """Initialize the trainer.

        Parameters
        ----------
        model : ModelLike, optional
            The model, by default None
        verbose : bool, optional
            Whether to print verbose messages, by default True
        callbacks : CallbackList, optional
            The callbacks, by default None
        **kwargs : dict, optional
            Other keyword arguments passed to TrainerArgs, by default None
        """
        builder = model
        if not isinstance(builder, ModelBuilder):
            builder = AutoModelBuilder(model)
        self._model_builder = builder
        self._callbacks = TrainerCallbackList(callbacks or [])
        self.verbose = verbose

    @property
    def callbacks(self) -> TrainerCallbackList:
        """Get the callbacks."""
        return self._callbacks

    @property
    def model_builder(self) -> ModelBuilder:
        """Get the model builder.

        Returns
        -------
        ModelBuilder
            The model builder.
        """
        return self._model_builder

    def create_model(self, force=False) -> Any:
        """Create the model from the model builder.

        Parameters
        ----------
        force : bool, optional
            Force to create the model, by default False

        Returns
        -------
        ModelLike
            The model.
        """
        if not hasattr(self, "_model") or self._model is None:
            self._model = self._model_builder()
        elif force:
            del self._model
            self.create_model(force=True)
        return self._model

    @property
    def model(self) -> Any:
        """Get or create the model.

        Notes
        -----
        Invokes `create_model` if the model is not created yet.

        Returns
        -------
        ModelLike
            The model.
        """
        return self.create_model()

    def set_model(self, model: Any):
        """Set the model."""
        self._model = model

    @abc.abstractmethod
    def fit(self, x, y=None, **kwargs):
        """Fit the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        """Predict using the model."""
        raise NotImplementedError

    def predict_proba(self, x, **kwargs):
        """Predict probabilities using the model."""
        raise NotImplementedError

    def score(self, x, y=None, sample_weight=None, **kwargs):
        """Score the model."""
        raise NotImplementedError
