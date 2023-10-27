from __future__ import annotations
import abc
import copy
import inspect
import shutil
from typing import Any, Dict, Optional
import weakref

import torch
import transformers
from transformers import PreTrainedModel
import joblib

from tklearn.nn.callbacks import TrainerCallbackList
from tklearn.utils import cache

__all__ = [
    "BaseTrainer",
    "ModelBuilder",
    "AutoModelBuilder",
]


class ModelBuilder(object):
    """Base class for model builders."""

    def __init__(self, **kwargs) -> None:
        """Initialize the model builder.

        Parameters
        ----------
        kwargs : dict
            Parameters of the model.
        """
        self._params = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def config(self) -> ModelConfig:
        return None

    def __getattr__(self, attr: str):
        """Get attribute of the model."""
        if attr in self._params:
            return self._params[attr]
        raise AttributeError(attr)

    def __setattr__(self, attr: str, value):
        """Set attribute of the model."""
        if attr.startswith("_"):
            super(ModelBuilder, self).__setattr__(attr, value)
        else:
            self._params[attr] = value

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters of the model.

        Parameters
        ----------
        deep : bool, optional
            If True, return the parameters as a deep copy, by default True

        Returns
        -------
        params : dict
            Parameters of the model.
        """
        if deep:
            return copy.deepcopy(self._params)
        return copy.copy(self._params)

    @abc.abstractmethod
    def build(self) -> Any:
        """Build the model.

        Returns
        -------
        model : Any
            The model.
        """
        raise NotImplementedError

    def __call__(self) -> Any:
        """Build the model."""
        return self.build()


class AutoModelBuilder(ModelBuilder):
    def __init__(self, base_model: Any) -> None:
        """Initialize the model builder.

        Parameters
        ----------
        base_model : Any
            The base model.
        """
        cache_dir = cache.mkdtemp(prefix="model-")
        self._finalizer = weakref.finalize(self, AutoModelBuilder._finalize, cache_dir)
        self._params = AutoModelBuilder._extract_params(base_model)
        # save the base model to the cache directory
        if isinstance(base_model, transformers.PreTrainedModel):
            # use transformers to load/save the model
            model_config = ModelConfig.from_pretrained(base_model)
            loader_fn = type(base_model).from_pretrained
            model_path = cache_dir
            base_model.save_pretrained(model_path)
        elif isinstance(base_model, torch.nn.Module):
            # use torch to save the model
            model_config = None
            loader_fn = torch.load
            model_path = cache_dir / "model.pt"
            torch.save(base_model, model_path)
        else:
            # try to use joblib to save the model
            # (this will work for scikit-learn based models)
            model_config = None
            loader_fn = joblib.load
            model_path = cache_dir / "model.joblib"
            joblib.dump(base_model, model_path)
        self._model_path = model_path
        self._loader_fn = loader_fn
        self._config = model_config

    @property
    def config(self) -> ModelConfig:
        return self._config

    @staticmethod
    def _finalize(cache_dir):
        shutil.rmtree(cache_dir)

    @staticmethod
    def _extract_params(obj) -> dict:
        if isinstance(obj, transformers.PreTrainedModel):
            return obj.config.to_dict()
        cls = type(obj)
        params = {}
        for key, arg in inspect.signature(cls).parameters.items():
            default = arg.default
            if default is inspect.Parameter.empty:
                params[key] = getattr(obj, key)
            else:
                params[key] = getattr(obj, key, default)
        return params

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the model.

        Returns
        -------
        params : dict
            Parameters of the model.
        """
        return copy.deepcopy(self._params)

    def build(self):
        """Build the model.

        Returns
        -------
        model : Any
            The model.
        """
        return self._loader_fn(self._model_path)


class ModelConfig(object):
    def __init__(
        self,
        *,
        problem_type: Optional[str] = None,
        label2id: Optional[Dict] = None,
        id2label: Optional[Dict] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super(ModelConfig, self).__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = num_labels
        self.problem_type = problem_type

    @property
    def problem_type(self) -> Optional[str]:
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: Optional[str]):
        if problem_type is None:
            if self.num_labels is None:
                return None
            if self.num_labels == 1:
                problem_type = "regression"
            elif self.num_labels > 1:
                problem_type = "single_label_classification"
        if problem_type not in [
            "regression",
            "single_label_classification",
            "multi_label_classification",
            "masked_language_modeling",
        ]:
            raise ValueError(
                f"invalid problem type: {problem_type}, "
                'expected one of ("regression", '
                '"single_label_classification", '
                '"multi_label_classification", '
                '"masked_language_modeling")'
            )
        self._problem_type = problem_type

    @property
    def label2id(self) -> Optional[Dict]:
        return self._label2id

    @label2id.setter
    def label2id(self, label2id):
        for _, idx in label2id.items():
            if isinstance(idx, str):
                idx = int(idx)
        self._label2id = label2id

    @property
    def id2label(self) -> Optional[Dict]:
        return self._id2label

    @id2label.setter
    def id2label(self, id2label):
        if id2label is None:
            if self.label2id is None:
                return None
            return {idx: label for label, idx in self.label2id.items()}
        for idx, label in id2label.items():
            if isinstance(idx, str):
                raise ValueError("invalid id2label, id must be int")
            if label not in self.label2id:
                raise ValueError("invalid id2label, label not in label2id")
            elif self.label2id and self.label2id[label] != idx:
                raise ValueError("invalid id2label, label mismatch")
        self._id2label = id2label

    @property
    def num_labels(self) -> Optional[int]:
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels):
        if num_labels is None:
            if self.label2id is None:
                return None
            return len(self.label2id)
        if self.label2id and len(self.label2id) != num_labels:
            raise ValueError(
                f"invalid num_labels: {num_labels}, " "len(label2id) != num_labels"
            )
        self._num_labels = num_labels

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel):
        """Builds loss function from the provided pre-trained model.

        Parameters
        ----------
        model : PreTrainedModel
            Pre-trained model.

        Returns
        -------
        loss : Callable
            Callable for the given task.
        """
        try:
            label2id = model.config.label2id
        except (AttributeError, KeyError):
            label2id = None
        try:
            id2label = model.config.id2label
        except (AttributeError, KeyError):
            id2label = None
        try:
            num_labels = model.config.num_labels
        except (AttributeError, KeyError):
            num_labels = None
        try:
            problem_type = model.config.problem_type
        except (AttributeError, KeyError):
            problem_type = None
        return cls(
            num_labels=num_labels,
            problem_type=problem_type,
            label2id=label2id,
            id2label=id2label,
        )


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

    def get_or_create_model(self, force=False) -> Any:
        """Get or create the model with the model builder.

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
            self.get_or_create_model(force=True)
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
        return self.get_or_create_model()

    @model.setter
    def model(self, model: Any) -> None:
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
