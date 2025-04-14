========================
Creating Embedding Models
========================

The ``tklearn.embeddings`` module allows you to create custom embedding implementations by extending the base ``Embedding`` class. This guide explains how to implement your own embedding model integrations.

Core Components
--------------

To implement a new embedding type, you need to create two classes:

1. A configuration class that extends ``EmbeddingConfig``
2. An embedding implementation class that extends ``Embedding``

Understanding Base Classes
-------------------------

Before implementing a new embedding, it's important to understand the base classes:

.. code-block:: python

    class EmbeddingConfig(BaseConfig, dispatch="loader"):
        loader: ClassVar[str]
        name: str
        verbose: bool | int = 1


    class EmbeddingBase(abc.ABC):
        def get_vectors(self) -> dict[str, ArrayLike]:
            raise NotImplementedError

        def get_encoder(self) -> Encodable:
            raise NotImplementedError


Key Methods to Implement
-----------------------

When creating a new embedding model, you need to implement at least these methods:

1. ``get_vectors()``: Downloads or loads the embedding data and returns a dictionary mapping words to vectors
2. ``get_encoder()`` (optional): Returns the underlying model object

Implementation Example: FastText
-------------------------------

Here's a complete example showing how to implement FastText embeddings:

.. code-block:: python

    import os
    from contextlib import contextmanager
    from pathlib import Path
    from typing import ClassVar, Dict

    import fasttext
    import fasttext.util
    import numpy as np
    from typing_extensions import Self

    from tklearn import config
    from tklearn.embeddings.base import Embedding, EmbeddingConfig


    @contextmanager
    def change_dir(path: str | Path):
        ...


    class FastTextEmbeddingConfig(EmbeddingConfig):
        loader: ClassVar[str] = "fasttext"
        name: str = "cc.en.300.bin"


    class FastTextEmbedding(Embedding):
        config: FastTextEmbeddingConfig

        def __post_init__(self):
            self.files_dir = (
                Path(config.assets_dir) / self.config.loader / "loader"
            )
            super().__post_init__()

        def _fetch_embedding(self) -> Self:
            lang_id = self.config.name.split(".")[1]
            with change_dir(self.files_dir):
                fasttext.util.download_model(lang_id, if_exists="ignore")
            return self

        def _read_embedding(self) -> Dict[str, np.ndarray]:
            fn = self.config.name
            model = fasttext.load_model(f"{self.files_dir / fn}")
            vectors = {}
            for term in model.get_words():
                vectors[term] = model.get_word_vector(term)
            return vectors

        def get_vectors(self) -> Dict[str, np.ndarray]:
            return self._fetch_embedding()._read_embedding()

        def get_encoder(self) -> FastTextWrapper:
            model = fasttext.load_model(f"{self.files_dir / self.config.name}")
            return FastTextWrapper(model)


    class FastTextWrapper:
        def __init__(self, model: fasttext.FastText._FastText):
            self.model = model

        def encode(self, texts: str | list[str]) -> np.ndarray:
            """Encode the texts."""
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for text in texts:
                vectors.append(self.model.get_word_vector(text))
            return np.array(vectors)

        def get_dimension(self) -> int | None:
            """Get the embedding size."""
            return self.model.get_dimension()


Step-by-Step: Creating a New Embedding Model
-------------------------------------------

1. Define Your Configuration Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration class that specifies the parameters for your embedding:

.. code-block:: python

    class YourEmbeddingConfig(EmbeddingConfig):
        loader: ClassVar[str] = "your_embedding_name"
        name: str = "your_embedding_name"


2. Create Your Embedding Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the core functionality in your embedding class:

.. code-block:: python

    class YourEmbedding(Embedding):
        config: YourEmbeddingConfig
        
        def __post_init__(self):
            # Set up any needed directories or resources
            self.files_dir = Path(config.resources_dir) / self.config.loader / "loader"
            super().__post_init__()
        
        def get_vectors(self) -> Dict[str, np.ndarray]:
            """
            Main method to implement: Load your embedding model and return
            a dictionary mapping words to their vector representations.
            """
            # Example implementation pattern:
            # 1. Download or locate the embedding files if needed
            # 2. Load the embedding data
            # 3. Convert to the required format: {word: vector_array}
            return {}
        
        def get_encoder(self) -> Encodable:
            """
            Optional: Return the underlying model object that implements
            the Encodable protocol.
            """
            return None

Understanding the Caching Mechanism
----------------------------------

The base ``Embedding`` class includes a built-in caching system:

1. When you instantiate your embedding, ``__post_init__`` tries to load cached vectors
2. If no cache exists, it calls your ``get_vectors()`` method to get the vectors
3. It then creates a memory-mapped file for efficient access to the vectors

You generally don't need to modify this caching behavior, but it's useful to understand how it works.

Registering Your Embedding
-------------------------

Your embedding will be automatically registered through the ``EmbeddingConfig`` class's dispatch mechanism, making it available via ``AutoEmbedding``:

.. code-block:: python

    from tklearn.embeddings import AutoEmbedding
    
    # Users can now load your embedding like this:
    embedding = AutoEmbedding.from_config({
        "name": "your_embedding_name",
        "version": "your_version"
    })
