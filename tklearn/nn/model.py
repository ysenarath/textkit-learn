from __future__ import annotations
import abc
import copy
import inspect
import shutil
from typing import Any, Dict
import weakref

import torch
import transformers
import joblib

from tklearn.utils import cache

__all__ = [
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
            loader_fn = type(base_model).from_pretrained
            model_path = cache_dir
            base_model.save_pretrained(model_path)
        elif isinstance(base_model, torch.nn.Module):
            # use torch to save the model
            loader_fn = torch.load
            model_path = cache_dir / "model.pt"
            torch.save(base_model, model_path)
        else:
            # try to use joblib to save the model
            # (this will work for scikit-learn based models)
            loader_fn = joblib.load
            model_path = cache_dir / "model.joblib"
            joblib.dump(base_model, model_path)
        self._model_path = model_path
        self._loader_fn = loader_fn

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
