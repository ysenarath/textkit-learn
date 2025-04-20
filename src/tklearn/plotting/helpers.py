# Dictionary to map style names to available Matplotlib styles
# Ensures that 'seaborn' maps to a specific seaborn style
import warnings

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.backend_bases import RendererBase
from matplotlib.legend import Legend

__all__ = [
    "get_style",
]

available_styles = {style: style for style in plt.style.available}
available_styles["seaborn"] = next(
    filter(
        lambda x: x.startswith("seaborn") and x.endswith("whitegrid"),
        plt.style.available,
    ),
    "seaborn-v0_8-whitegrid",  # Fallback if no matching style found
)


def get_style(style: str) -> str | None:
    """Retrieve the actual Matplotlib style name based on the input string.

    Parameters
    ----------
    style : str
        The desired style name (e.g., "seaborn", "ggplot").

    Returns
    -------
    str or None
        The corresponding Matplotlib style name if found, otherwise None.
    """
    return available_styles.get(style)


def get_renderer(fig: plt.Figure) -> RendererBase:
    """Get the renderer instance for a Matplotlib figure.

    Needed for accurately calculating bounding boxes of elements like legends.

    Parameters
    ----------
    fig : plt.Figure
        The Matplotlib figure instance.

    Returns
    -------
    RendererBase
        The renderer instance.

    Raises
    ------
    AttributeError
        If a renderer cannot be found for the figure's backend.
    """
    # Try different ways to get the renderer depending on Matplotlib version/backend
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()
    # Deprecated way, kept for compatibility
    elif hasattr(fig, "_get_renderer"):  # pragma: no cover
        return fig._get_renderer()
    # If no renderer found, raise an error
    backend = matplotlib.get_backend()
    msg = f"Could not find a renderer for the '{backend}' backend."
    raise AttributeError(msg)


def set_legend(
    handles: list[Artist],
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    frameon: bool = True,
    fancybox: bool = True,
    ncols: int = 1,
    max_ncols: int | None = None,
) -> Legend:
    """Create and position a legend outside the plot area.

    Adjusts the number of columns dynamically to fit the available space
    if `max_ncols` is specified.

    Parameters
    ----------
    handles : list[Artist]
        A list of Matplotlib artists (like scatter points) to include in the legend.
    fig : plt.Figure
        The Matplotlib figure.
    ax : plt.Axes
        The Matplotlib axes where the plot resides.
    title : str
        The title for the legend.
    frameon : bool, optional
        Whether to draw a frame around the legend. Defaults to True.
    fancybox : bool, optional
        Whether to use rounded corners for the legend frame. Defaults to True.
    ncols : int, optional
        Initial number of columns for the legend. Defaults to 1.
    max_ncols : int or None, optional
        Maximum number of columns allowed for the legend. If None,
        no dynamic adjustment is made. Defaults to None.

    Returns
    -------
    matplotlib.legend.Legend
        The created Matplotlib legend object.
    """
    loc = "upper left"  # Initial location anchor for the legend
    # Position relative to the axes (outside, top right)
    bbox_to_anchor = (1.01, 1.01)

    renderer = get_renderer(fig)  # Get the renderer to calculate sizes

    # Get plot dimensions
    plot_extent = ax.get_tightbbox(renderer)
    plot_height = plot_extent.height / fig.dpi
    plot_width = plot_extent.width / fig.dpi

    # Extract labels from handles
    labels = [h.get_label() for h in handles]

    # Create the initial legend
    legend = ax.legend(
        handles,
        labels,
        title=title,
        frameon=frameon,
        fancybox=fancybox,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncols=ncols,
    )

    # Calculate initial legend size
    legend_extent = legend.get_tightbbox(renderer)
    legend_height = legend_extent.height / fig.dpi
    legend_width = legend_extent.width / fig.dpi

    # Adjust figure size to accommodate the legend
    # Note: This might need fine-tuning depending on the desired layout
    fig.set_size_inches(
        plot_width + legend_width * 1.1,
        max(plot_height, legend_height),
        forward=True,
    )  # Added buffer * 1.1
    # Ensure tight layout is applied *after* resizing
    try:
        fig.tight_layout()
    except (
        ValueError
    ):  # Sometimes tight_layout fails after resizing, ignore for now
        pass
    # Re-position legend slightly after potential layout adjustments
    legend.set_bbox_to_anchor(bbox_to_anchor)

    # If max_ncols is set, try increasing columns if legend is too tall
    if max_ncols is None:
        return legend

    current_ncols = ncols
    # Loop to increase columns if legend is taller than plot (approximate check)
    # And we haven't reached the maximum allowed columns
    while (current_ncols < max_ncols) and (
        legend_height > plot_height * 1.1
    ):  # Added buffer * 1.1
        current_ncols += 1
        # Remove the old legend
        legend.remove()
        # Store previous figure size
        prev_width, prev_height = fig.get_size_inches()

        # Create new legend with more columns
        legend = ax.legend(
            handles,
            labels,
            title=title,
            frameon=frameon,
            fancybox=fancybox,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncols=current_ncols,
        )

        # Recalculate legend size
        legend_extent = legend.get_tightbbox(renderer)
        legend_height = legend_extent.height / fig.dpi
        legend_width = legend_extent.width / fig.dpi

        # Adjust figure size again
        # Use the original plot width and the new legend width
        new_fig_width = plot_width + legend_width * 1.1
        new_fig_height = max(plot_height, legend_height)
        fig.set_size_inches(new_fig_width, new_fig_height, forward=True)
        try:
            fig.tight_layout()
        except ValueError:
            pass
        legend.set_bbox_to_anchor(bbox_to_anchor)

        # Break if increasing columns didn't significantly reduce height (or made it worse)
        # This prevents potential infinite loops in edge cases
        if (
            current_ncols > ncols + 1
            and legend_height
            >= legend.get_tightbbox(renderer).height / fig.dpi * 0.95
        ):  # Check if height reduction is minimal
            warnings.warn(
                "Legend height not reducing significantly with more columns. Stopping column increase.",
                stacklevel=1,
            )
            break

    return legend
