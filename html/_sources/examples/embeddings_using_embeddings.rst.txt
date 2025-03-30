===============
Using Embeddings
===============

The ``AutoEmbedding`` class from ``tklearn.embeddings`` provides a unified interface for loading and using word embedding models like GloVe and FastText.

Getting Started
--------------

First, import the necessary class:

.. code-block:: python

    from tklearn.embeddings import AutoEmbedding

Available Embedding Models
-------------------------

Gensim Models
~~~~~~~~~~~~
The library supports all pre-trained word embedding models from the 
`gensim-data repository <https://github.com/piskvorky/gensim-data?tab=readme-ov-file#models>`_,
including popular ones like GloVe, Word2Vec, and FastText.

FastText Models
~~~~~~~~~~~~~
You can use any of the pre-trained word vectors for 157 languages from
`FastText's Crawl Vectors <https://fasttext.cc/docs/en/crawl-vectors.html#models>`_,
which include 300-dimensional word vectors trained on Common Crawl and Wikipedia.

Loading Embedding Models
-----------------------

There are several ways to load embedding models:

Simple String Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick access to Gensim models:

.. code-block:: python

    # Load GloVe Twitter 25d embeddings
    wv = AutoEmbedding.from_config("glove-twitter-25")
    assert wv["hello"].shape == (25,)

Dictionary Configuration
~~~~~~~~~~~~~~~~~~~~~~

With name parameter:

.. code-block:: python

    wv = AutoEmbedding.from_config({"name": "glove-twitter-25"})
    assert wv["hello"].shape == (25,)

With name and version parameters for explicit model selection:

.. code-block:: python

    # GloVe from Gensim 
    wv = AutoEmbedding.from_config({
        "name": "gensim",
        "version": "glove-twitter-25",
    })
    assert wv["hello"].shape == (25,)
    
    # FastText model
    wv = AutoEmbedding.from_config({
        "name": "fasttext",
        "version": "cc.en.300.bin",
    })
    assert wv["hello"].shape == (300,)

Using the Embeddings
-------------------

Once loaded, retrieve word vectors using dictionary-like syntax:

.. code-block:: python

    # Get embedding for a specific word
    vector = wv["hello"]
    
    # The dimensionality depends on the model
    # - 25 dimensions for "glove-twitter-25"
    # - 300 dimensions for FastText's "cc.en.300.bin"

The resulting vectors can be used for various NLP tasks like semantic similarity, text classification, or as input features for neural networks.