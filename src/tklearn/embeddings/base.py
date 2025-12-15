from __future__ import annotations

import abc
import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar, Optional, Protocol, Union, runtime_checkable

import numpy as np
from nightjar import AutoModule, BaseConfig, BaseModule
from numpy.typing import ArrayLike

from tklearn import config, logging

__all__ = [
    "EmbeddingConfig",
    "AutoEmbedding",
    "Embedding",
]

logger = logging.get_logger(__name__)


class EmbeddingConfig(BaseConfig, dispatch="loader"):
    """Configuration for loading and managing embeddings.

    Attributes
    ----------
    loader : str
        The identifier for the embedding loader module to use.
        This is a ClassVar and also the dispatch key for AutoModule.
    name : str
        A unique name identifying the specific embedding set (e.g., 'glove-wiki-gigaword-100').
    verbose : bool or int, optional
        Controls the verbosity of loading and processing messages.
        Defaults to 1.
    """

    loader: ClassVar[str]
    name: str
    verbose: Union[bool, int] = 1


class AutoEmbedding(AutoModule):
    """Factory class for creating Embedding instances based on configuration.

    Uses the 'loader' field in the EmbeddingConfig to dispatch to the
    appropriate Embedding subclass.
    """

    def __new__(cls, config: EmbeddingConfig | Mapping | str) -> Embedding:
        """Instantiate an Embedding from a configuration.

        Parameters
        ----------
        config : EmbeddingConfig or Mapping or str
            The configuration object, dictionary, or string identifier.
            If a string is provided, it's assumed to be the name, and
            'gensim' loader is used by default. If a Mapping is provided,
            it's converted to an EmbeddingConfig.

        Returns
        -------
        Embedding
            An instance of the appropriate Embedding subclass.
        """
        if isinstance(config, str):
            config = {"loader": "gensim", "name": config}
        if not isinstance(config, EmbeddingConfig):
            try:
                config = EmbeddingConfig.from_dict(config)
            except KeyError:
                config = {"loader": "gensim", **config}
                config = EmbeddingConfig.from_dict(config)
        return super().__new__(cls, config)


@runtime_checkable
class Encodable(Protocol):
    """Protocol defining the interface for text encoding models.

    Classes implementing this protocol should provide a method to convert
    text strings into numerical vector representations (embeddings).
    """

    def encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode a given text string or token span into a vector.

        Parameters
        ----------
        texts : str or list[str]
            The text to be encoded. Can be a single string or a list of
            strings (e.g., tokens or phrases).
        **kwargs
            Additional keyword arguments for the encoding process.
            This may include options like batch size or other model-specific
            parameters.

        Returns
        -------
        np.ndarray
            The numerical vector representation (embedding) of the input texts.
        """
        ...

    def get_dimension(self) -> int | None:
        """Get the size of the embedding vector.

        Returns
        -------
        int or None
            The dimensionality of the embedding vectors produced by this model.
        """
        ...


class EmbeddingBase(abc.ABC):
    """Abstract base class for embedding loaders.

    Defines the core interface that concrete embedding loader implementations
    must provide.
    """

    def get_vectors(self) -> dict[str, ArrayLike]:
        """Load or compute the word-to-vector mapping.

        This method should handle the retrieval or generation of the
        embedding vectors from the source (e.g., file, model).

        Returns
        -------
        dict[str, ArrayLike]
            A dictionary mapping words (or entities) to their corresponding
            numerical vector representations.
        """
        raise NotImplementedError

    def get_encoder(self) -> Encodable:
        """Get the text encoder associated with these embeddings.

        This method should return an object conforming to the Encodable
        protocol, capable of encoding arbitrary text based on the loaded
        embedding model.

        Returns
        -------
        Encodable
            An object that can encode text into vectors.
        """
        raise NotImplementedError


class Embedding(BaseModule, Mapping[str, np.ndarray], EmbeddingBase):
    """Represents a loaded set of word embeddings.

    Provides access to word vectors via a mapping interface and potentially
    an underlying text encoding model. Handles caching of processed vectors.

    Attributes
    ----------
    config : EmbeddingConfig
        The configuration used to load these embeddings.
    word_to_index : dict[str, int] or None
        A mapping from words (vocabulary) to their integer index in the
        `vectors` array. Initialized during loading.
    vectors : np.ndarray or None
        A 2D NumPy array where each row corresponds to a word vector,
        ordered according to `word_to_index`. Initialized during loading.
        May be a memory-mapped array for efficiency.
    model : TextEncoder or None
        An optional underlying text encoding model (conforming to TextEncoder)
        that might be used for generating vectors for out-of-vocabulary words
        or for more complex encoding tasks. Initialized during loading.
    """

    config: EmbeddingConfig
    word_to_index: Optional[dict[str, int]] = None
    vectors: np.ndarray = None
    model: Optional[Encodable] = None

    def __post_init__(self) -> None:
        """Initializes the embedding by loading or creating cached data."""
        cache_path = (
            Path(config.assets_dir)
            / self.config.loader
            / "data"
            / f"vectors-{self.config.name}.data"
        )
        # create embedding if not exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model = self.get_encoder()
            if model and not isinstance(model, Encodable):
                msg = f"{model.__class__.__name__} is not an instance of Encodable"
                warnings.warn(msg, UserWarning)
                raise NotImplementedError
            self.model = model
        except NotImplementedError:
            self.model = None
        try:
            self._load(cache_path)
        except FileNotFoundError:
            mapping = self.get_vectors()
            self._from_dict(mapping)
            self._dump(cache_path)

    @staticmethod
    def _get_shape(filename: str | Path) -> tuple[int, int]:
        with open(filename, "rb") as f:
            major, minor = np.lib.format.read_magic(f)
            if major == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif major == 2:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                msg = f"array file format {major}.{minor} not supported"
                raise ValueError(msg)
        return shape

    def _load(self, path: Path | str) -> Embedding:
        path = Path(path)
        with open(path.with_suffix(".word_to_index.json")) as f:
            word_to_index = json.load(f)
        filename = path.with_suffix(".vectors.npy")
        shape = self._get_shape(filename)
        # memory-mapped array
        vectors = np.memmap(filename, dtype=np.float32, mode="r", shape=shape)
        if vectors.ndim != 2 and shape[0] == 0:
            dim_size = getattr(self.model, "get_dimension", lambda: None)()
            if dim_size is None:
                dim_size = 0
            vectors = np.empty((0, dim_size), dtype=np.float32)
        self.vectors = vectors
        self.word_to_index = word_to_index

    def _from_dict(self, wv: dict[str, np.ndarray]) -> Embedding:
        word_to_index = {entity: i for i, entity in enumerate(wv.keys())}
        vectors = np.array(list(wv.values()))
        self.word_to_index = word_to_index
        self.vectors = vectors

    def _dump(self, path: Path | str) -> None:
        path = Path(path)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        with open(path.with_suffix(".word_to_index.json"), "w") as f:
            json.dump(self.word_to_index, f)

    def encode(self, key: str | list[str], **kwargs) -> np.ndarray:
        """Get the vector for a word, potentially using the model for OOV words.

        If an underlying `model` (TextEncoder) is available, it might be used
        to generate vectors for words not directly in the `vectors` table or
        to handle multi-word expressions by averaging sub-word vectors.
        Otherwise, it falls back to `__getitem__`. Results are cached.

        Parameters
        ----------
        key : str or list[str]
            The word or phrase to get the vector for.
        **kwargs
            Additional keyword arguments for the encoder.

        Returns
        -------
        np.ndarray
            The vector representation of the word.
        """
        try:
            # this will be a 1D array
            if not isinstance(key, str):
                # this will be a 2D array
                raise KeyError(key)
            return self.vectors[self.word_to_index[key]]
        except KeyError:
            pass
        if self.model:
            vectors = self.model.encode(key, **kwargs)
            if isinstance(key, str):
                # this will be a 1D array
                return vectors[0]
            # this will be a 2D array
            return vectors
        raise KeyError(key)

    def encode_query(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode the texts as queries."""
        if not hasattr(self.model, "encode_query"):
            raise NotImplementedError(
                f"{self.model.__class__.__name__} does not implement encode_query"
            )
        return self.model.encode_query(texts, **kwargs)

    def encode_document(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode the texts as documents."""
        if not hasattr(self.model, "encode_document"):
            raise NotImplementedError(
                f"{self.model.__class__.__name__} does not implement encode_document"
            )
        return self.model.encode_document(texts, **kwargs)

    def __getitem__(self, key: str) -> np.ndarray:
        """Retrieve the vector for a specific word.

        Parameters
        ----------
        key : str
            The text whose vector is requested.

        Returns
        -------
        np.ndarray
            The vector corresponding to the provided text.

        Raises
        ------
        KeyError
            If the word is not in the vocabulary (`word_to_index`) and the model is None.
        """
        return self.encode(key)

    def __iter__(self) -> Iterable[str]:
        """Iterate over the words in the vocabulary."""
        return iter(self.word_to_index)

    def __len__(self) -> int:
        """Return the number of words in the vocabulary."""
        return len(self.word_to_index)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the underlying vector matrix (vocabulary size, dimension)."""
        return self.vectors.shape
