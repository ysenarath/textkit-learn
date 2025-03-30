========================
Creating New Embedding Models
========================

The ``tklearn.embeddings`` framework allows you to create custom embedding implementations by extending the base ``Embedding`` class. This guide explains how to implement your own embedding model integrations.

Core Components
--------------

To implement a new embedding type, you need to create two classes:

1. A configuration class that extends ``EmbeddingConfig``
2. An embedding implementation class that extends ``Embedding``

Understanding Base Classes
-------------------------

Before implementing a new embedding, it's important to understand the base classes:

.. code-block:: python

    class EmbeddingConfig(BaseConfig, dispatch="name"):
        name: ClassVar[str]  # Name identifier for the embedding type
        version: str = "0.0.1"  # Version identifier for the specific model
        verbose: Union[bool, int] = 1  # Verbosity level

    class Embedding(BaseModule, BaseEmbedding):
        config: EmbeddingConfig
        word_to_index: Optional[Dict[str, int]] = None
        vectors: Optional[np.ndarray] = None
        model: Optional[EmbeddingModel] = None
        
        # Methods that handle caching of embeddings
        def __post_init__(self) -> None: ...
        def _load(self, path: Path | str) -> Embedding: ...
        def _from_dict(self, wv: Dict[str, np.ndarray]) -> Embedding: ...
        def _dump(self, path: Path | str) -> None: ...
        
        # Dictionary-like access to embeddings
        def __getitem__(self, key: str) -> np.ndarray: ...
        def __iter__(self) -> Iterable[str]: ...
        def __len__(self) -> int: ...
        
        # Methods you must implement in your subclass
        def load(self) -> Dict[str, ArrayLike]: ...
        def get_model(self) -> EmbeddingModel: ...

Key Methods to Implement
-----------------------

When creating a new embedding model, you need to implement at least these methods:

1. ``load()``: Downloads or loads the embedding data and returns a dictionary mapping words to vectors
2. ``get_model()`` (optional): Returns the underlying model object

Implementation Example: FastText
-------------------------------

Here's a complete example showing how to implement FastText embeddings:

.. code-block:: python

    from __future__ import annotations
    import logging
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
    
    logger = logging.getLogger(__name__)
    
    @contextmanager
    def change_dir(path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        old_path = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_path)
    
    class FastTextEmbeddingConfig(EmbeddingConfig):
        name: ClassVar[str] = "fasttext"
        version: str = "cc.en.300.bin"
    
    class FastTextEmbedding(Embedding):
        config: FastTextEmbeddingConfig
        
        def __post_init__(self):
            self.files_dir = (
                Path(config.resources_dir) / self.config.name / "loader"
            )
            super().__post_init__()
        
        def _fetch_embedding(self) -> Self:
            lang_id = self.config.version.split(".")[1]
            with change_dir(self.files_dir):
                fasttext.util.download_model(lang_id, if_exists="ignore")
            return self
        
        def _read_embedding(self) -> Dict[str, np.ndarray]:
            fn = self.config.version
            model = fasttext.load_model(f"{self.files_dir / fn}")
            vectors = {}
            for term in model.get_words():
                vectors[term] = model.get_word_vector(term)
            return vectors
        
        def load(self) -> Dict[str, np.ndarray]:
            return self._fetch_embedding()._read_embedding()
        
        def get_model(self) -> fasttext.FastText:
            fn = self.config.version
            return fasttext.load_model(f"{self.files_dir / fn}")

Step-by-Step: Creating a New Embedding Model
-------------------------------------------

1. Define Your Configuration Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration class that specifies the parameters for your embedding:

.. code-block:: python

    class YourEmbeddingConfig(EmbeddingConfig):
        name: ClassVar[str] = "your_embedding_name"
        version: str = "default_version"
        # Add any additional parameters needed for your embedding

2. Create Your Embedding Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the core functionality in your embedding class:

.. code-block:: python

    class YourEmbedding(Embedding):
        config: YourEmbeddingConfig
        
        def __post_init__(self):
            # Set up any needed directories or resources
            self.files_dir = Path(config.resources_dir) / self.config.name / "data"
            super().__post_init__()
        
        def load(self) -> Dict[str, np.ndarray]:
            """
            Main method to implement: Load your embedding model and return
            a dictionary mapping words to their vector representations.
            """
            # Example implementation pattern:
            # 1. Download or locate the embedding files if needed
            # 2. Load the embedding data
            # 3. Convert to the required format: {word: vector_array}
            return {}
        
        def get_model(self) -> EmbeddingModel:
            """
            Optional: Return the underlying model object that implements
            the EmbeddingModel protocol (has a get_word_vector method).
            """
            return None

Understanding the Caching Mechanism
----------------------------------

The base ``Embedding`` class includes a built-in caching system:

1. When you instantiate your embedding, ``__post_init__`` tries to load cached vectors
2. If no cache exists, it calls your ``load()`` method to get the vectors
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

Example: Adding a Hugging Face Embedding
---------------------------------------

Here's how you might implement support for Hugging Face embeddings:

.. code-block:: python

    import os
    from pathlib import Path
    from typing import ClassVar, Dict
    import numpy as np
    from transformers import AutoModel, AutoTokenizer
    import torch
    from tklearn import config
    from tklearn.embeddings.base import Embedding, EmbeddingConfig
    
    class HuggingFaceEmbeddingConfig(EmbeddingConfig):
        name: ClassVar[str] = "huggingface"
        version: str = "bert-base-uncased"
    
    class HuggingFaceEmbedding(Embedding):
        config: HuggingFaceEmbeddingConfig
        
        def __post_init__(self):
            self.model_name = self.config.version
            super().__post_init__()
        
        def load(self) -> Dict[str, np.ndarray]:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            # Get embeddings for the vocabulary
            vectors = {}
            for word in tokenizer.get_vocab():
                inputs = tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use the [CLS] token embedding as the word vector
                vector = outputs.last_hidden_state[:, 0, :].numpy()[0]
                vectors[word] = vector
                
            return vectors
        
        def get_model(self):
            # Return a model wrapper that implements get_word_vector
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            class ModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                
                def get_word_vector(self, word):
                    inputs = self.tokenizer(word, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    return outputs.last_hidden_state[:, 0, :].numpy()[0]
            
            return ModelWrapper(model, tokenizer)