# Import necessary libraries
from __future__ import annotations

import warnings
from typing import Any, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from tklearn.plotting.helpers import get_renderer, get_style, set_legend

try:
    from umap import UMAP
except ImportError:
    UMAP = None

# Define what functions are publicly available when importing * from this module
__all__ = [
    "plot_embedding",
]


def embed2d(X: np.ndarray, dim_reducer: str = "umap") -> np.ndarray:  # noqa: N803 The capital X is conventional here
    """Perform 2D embedding using UMAP or t-SNE.

    Defaults to 'umap'. Falls back to 'tsne' if 'umap'
    is requested but not installed.

    Parameters
    ----------
    X : np.ndarray
        The high-dimensional data array (n_samples, n_features).
    dim_reducer : str, optional
        The dimensionality reduction technique to use ('umap' or 'tsne').
        Defaults to 'umap'.

    Returns
    -------
    np.ndarray
        The 2D embedded data array (n_samples, 2).

    Raises
    ------
    ValueError
        If an unsupported embedder is specified.
    """
    dim_reducer = dim_reducer.lower()  # Convert embedder name to lowercase
    if dim_reducer == "umap" and UMAP is not None:
        # Use UMAP if available and requested
        # supress warnings from UMAP
        with warnings.catch_warnings():
            # Perform UMAP embedding
            umap = UMAP(n_components=2, random_state=42)
            # Added random_state for reproducibility
            X_embedded = umap.fit_transform(X)
    elif dim_reducer in {"tsne", "t-sne", "umap"}:
        # Use t-SNE if requested, or if UMAP was requested but not installed
        if dim_reducer == "umap":
            msg = "umap-learn is not installed, falling back to scikit-learn's t-SNE."
            warnings.warn(msg, stacklevel=1)
        # Use t-SNE
        X_embedded = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=42,
            n_jobs=1,
        ).fit_transform(X)  # Added common parameters
    else:
        # Raise error for unsupported embedder
        msg = (
            f"Embedder '{dim_reducer}' not supported. Choose 'umap' or 'tsne'."
        )
        raise ValueError(msg)
    return X_embedded


def plot_embedding(
    data: pd.DataFrame,
    x: str = "embedding",
    y: str = "label",
    style: Any = "seaborn",
    cmap: Any = "rainbow",
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (8, 6),  # Adjusted default figsize
    dim_reducer: str = "umap",
    dpi: float = 100,
    legend_max_ncols: int = 5,
) -> plt.Figure:
    """Generate a 2D scatter plot of embedded data, colored by labels.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data. Must have columns specified by
        `x` (containing lists/arrays of embeddings) and `y` (containing labels).
    x : str, optional
        Name of the column containing the high-dimensional embeddings.
        Defaults to "embedding".
    y : str, optional
        Name of the column containing the labels for coloring.
        Defaults to "label".
    style : Any, optional
        Matplotlib style to use for the plot (e.g., 'seaborn', 'ggplot').
        Defaults to "seaborn".
    cmap : Any, optional
        Colormap to use for coloring points by label. Defaults to "rainbow".
    alpha : float, optional
        Transparency level for the scatter points (0=transparent, 1=opaque).
        Defaults to 0.5.
    figsize : Tuple[int, int], optional
        Tuple specifying the figure size (width, height) in inches.
        Defaults to (8, 6).
    dim_reducer : str, optional
        Dimensionality reduction technique ('umap' or 'tsne').
        Defaults to "umap".
    dpi : float, optional
        Dots per inch (resolution) for the figure. Defaults to 100.
    legend_max_ncols : int, optional
        Maximum number of columns to use for the legend if it needs
        to be rearranged due to height. Defaults to 5.

    Returns
    -------
    plt.Figure
        The Matplotlib Figure object containing the plot.

    Raises
    ------
    ValueError
        If columns specified by `x` or `y` are not found in the DataFrame,
        or if embedding vectors in column `x` cannot be stacked (e.g., due
        to inconsistent dimensions).
    TypeError
        If the column specified by `x` does not contain iterable objects
        (like lists or numpy arrays).
    """
    x_col, y_col = x, y  # Assign column names

    # --- Input Validation ---
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")
    if y_col not in data.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")
    if not data[x_col].apply(lambda item: hasattr(item, "__iter__")).all():
        raise TypeError(
            f"Column '{x_col}' must contain iterable embedding vectors (like lists or numpy arrays)."
        )

    # Convert embedding column to a NumPy array
    # Use np.vstack for robust handling of list/array types
    try:
        X_high_dim = np.vstack(data[x_col].values)  # Use vstack for safety
    except ValueError as e:
        raise ValueError(
            f"Could not stack embedding vectors from column '{x_col}'. Ensure all embeddings have the same dimension."
        ) from e

    # Perform 2D embedding
    X_embedded = embed2d(X_high_dim, dim_reducer=dim_reducer)

    # Prepare labels as categorical data
    labels = pd.Series(data[y_col]).astype("category")

    # Get the actual style name and colormap object
    style_name = get_style(style)
    if style_name is None:
        warnings.warn(
            f"Style '{style}' not found. Using default Matplotlib style.",
            stacklevel=1,
        )
        # Use default if not found
        style_name = plt.style.use("default")

    try:
        cmap_obj = plt.colormaps.get_cmap(cmap)
    except ValueError:
        warnings.warn(
            f"Colormap '{cmap}' not found. Using 'viridis'.", stacklevel=1
        )
        # Use default if not found
        cmap_obj = plt.colormaps.get_cmap("viridis")

    num_classes = len(labels.cat.categories)  # Number of unique labels

    # Create the plot using the specified style
    with plt.style.context(style=style_name):
        # Create figure and axes
        # Removed tight_layout=True initially, will apply later
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(dpi)

        handles = []  # List to store handles for the legend
        # Iterate through each unique label to plot its points
        for label_id, label in enumerate(labels.cat.categories):
            idx = np.where(labels == label)[
                0
            ]  # Find indices for the current label
            coords = X_embedded[idx]  # Get the 2D coordinates for these points
            color = cmap_obj(
                label_id / (num_classes - 1) if num_classes > 1 else 0.5
            )  # Get color from colormap

            # Create the scatter plot for the current label
            scatter = ax.scatter(
                x=coords[:, 0],  # x-coordinates
                y=coords[:, 1],  # y-coordinates
                color=color,
                label=str(label),  # Ensure label is a string for legend
                alpha=alpha,
                edgecolors="none",  # No borders around points
            )
            handles.append(scatter)  # Add the scatter plot artist to handles

        # Remove default legend if it exists (we create a custom one)
        if ax.get_legend():
            ax.get_legend().remove()

        # Add the custom legend outside the plot
        legend = set_legend(  # Capture the returned legend object
            handles,
            fig,
            ax,
            title=str(y_col),  # Ensure title is a string
            max_ncols=legend_max_ncols,
        )

        # Add grid lines
        ax.grid(True)
        # Set plot titles and labels (optional, but good practice)
        ax.set_title(
            f"{dim_reducer.upper()} Embedding of {x_col} colored by {y_col}"
        )
        ax.set_xlabel(f"{dim_reducer.upper()} Dimension 1")
        ax.set_ylabel(f"{dim_reducer.upper()} Dimension 2")

        # Apply tight layout at the end to adjust spacing
        try:
            # Adjust rect to try and prevent legend overlap
            # The right boundary is adjusted based on estimated legend width factor
            legend_width_factor = legend.get_window_extent(
                renderer=get_renderer(fig)
            ).width / (fig.dpi * figsize[0])
            right_boundary = max(
                0.8, 1.0 - legend_width_factor * 1.1
            )  # Ensure some minimum plot area
            fig.tight_layout(rect=[0, 0, right_boundary, 1])
        except (ValueError, AttributeError):
            # Catch potential errors during layout
            # Sometimes tight_layout fails, especially with complex legends or backends
            warnings.warn(
                "fig.tight_layout() failed. Plot spacing might not be optimal.",
                stacklevel=1,
            )

    return fig  # Return the figure object
