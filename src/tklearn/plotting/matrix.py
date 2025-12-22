from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["plot_dot_matrix"]


def plot_dot_matrix(data):
    """
    Creates a dot matrix plot from a 2D numpy array and returns the Figure.

    Args:
        data (np.array): A 2D matrix of 0s and 1s.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array.")

    nrows, ncols = data.shape

    # 1. Prepare Data
    plot_data = []
    for y in range(nrows):
        for x in range(ncols):
            plot_data.append({"x": x, "y": y, "value": data[y, x]})
    df = pd.DataFrame(plot_data)

    # 2. Initialize Figure
    fig, ax = plt.subplots(figsize=(nrows, ncols))
    ax.set_facecolor("#AFB2B6")
    colors = {0: "white", 1: "#D68F90"}

    # 3. Create Plot
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="value",
        palette=colors,
        s=50 * max(nrows, ncols),
        marker="o",
        edgecolor=None,
        legend=False,
        ax=ax,
    )

    # 4. Styling
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Add numbers to edges
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(range(ncols), fontsize=ncols * 10 / 7, color="#333333")
    ax.set_yticklabels(range(nrows), fontsize=nrows * 10 / 7, color="#333333")
    ax.xaxis.tick_top()

    # Clean up
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    return fig
