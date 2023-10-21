from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tklearn.embedding import keyedvectors


__all__ = [
    "ConvTextClassifier",
]


class ConvTextClassifier(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.

    Reference: https://github.com/cezannec/CNN_Text_Classification/blob/master/CNN_Text_Classification.ipynb
    """

    def __init__(
        self,
        weights: keyedvectors,
        padding_idx: Optional[int] = None,
        output_size: int = 1,
        num_filters: int = 100,
        kernel_sizes: List[int] = [3, 4, 5],
        freeze_embeddings: bool = True,
        drop_prob: float = 0.5,
    ):
        """
        Initialize the model by setting up the layers.
        """
        super(ConvTextClassifier, self).__init__()
        # set class vars
        vocab_size, embedding_dim = weights.shape
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        # 1. embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(
            torch.from_numpy(weights.vectors)
        )  # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False
        # 2. convolutional layers
        self.convs_1d = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k - 2, 0))
                for k in kernel_sizes
            ]
        )
        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)

    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)
        # 1D pool over conv_seq_length
        x_max: torch.Tensor = F.max_pool1d(x, x.size(2))
        # squeeze to get size: (batch_size, num_filters)
        return x_max.squeeze(2)

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds: torch.Tensor = self.embedding(
            x
        )  # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        # final logit
        logit = self.fc(x)
        # sigmoid-activated --> a class score
        return logit
