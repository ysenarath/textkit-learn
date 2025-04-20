# --- Create Sample Data ---
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from tklearn.plotting.embeddings import plot_embedding

n_samples = 150
n_features_high = 50  # Dimension of original embeddings
centers = 4  # Number of clusters/labels

X_high_dim_sample, y_sample = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=n_features_high,
    cluster_std=1.5,  # Spread of clusters
    random_state=42,
)

# Convert high-dimensional data to list of lists/arrays for the DataFrame column
embedding_list = [list(vec) for vec in X_high_dim_sample]

# Create DataFrame
sample_df = pd.DataFrame({
    "embedding": embedding_list,
    "label": y_sample,
    "label_str": [f"Class_{i}" for i in y_sample],
})

print("Sample DataFrame head:")
print(sample_df.head())
print(f"\nEmbedding vector length: {len(sample_df['embedding'].iloc[0])}")
print(f"Number of unique labels: {sample_df['label_str'].nunique()}")

# --- Plotting ---
print("\nGenerating UMAP plot...")
# Plot using UMAP (default) and string labels
fig_umap = plot_embedding(
    data=sample_df,
    x="embedding",
    y="label_str",  # Use string labels
    dim_reducer="umap",
    cmap="viridis",  # Different colormap
    figsize=(9, 7),
    legend_max_ncols=2,  # Allow legend to use up to 2 columns
)
# Try to save the figure before showing
try:
    fig_umap.savefig(
        "examples/outputs/embedding_umap_seaborn.png",
        bbox_inches="tight",
        dpi=150,
    )
    print("UMAP plot saved to embedding_umap_seaborn.png")
except Exception as save_err:
    print(f"Could not save UMAP plot: {save_err}")
plt.show()  # Display the plot
print("UMAP plot generation attempted.")

# --- Additional Plotting Examples ---

# Try t-SNE if UMAP failed (e.g., not installed)
print("\nAttempting t-SNE plot as fallback...")
try:
    fig_tsne = plot_embedding(
        data=sample_df,
        x="embedding",
        y="label_str",
        dim_reducer="tsne",
        cmap="plasma",
        figsize=(9, 7),
        legend_max_ncols=2,
    )
    # Try to save the figure before showing
    try:
        fig_tsne.savefig(
            "examples/outputs/embedding_tsne_seaborn.png",
            bbox_inches="tight",
            dpi=150,
        )
        print("t-SNE plot saved to embedding_tsne_seaborn.png")
    except Exception as save_err:
        print(f"Could not save t-SNE plot: {save_err}")
    plt.show()
    print("t-SNE plot generation attempted.")
except Exception as e_tsne:
    print(f"Could not generate t-SNE plot either: {e_tsne}")

# --- Additional Plotting Examples ---

# Example with numerical labels and different style
print("\nGenerating t-SNE plot with numerical labels and 'ggplot' style...")
try:
    fig_tsne_num = plot_embedding(
        data=sample_df,
        x="embedding",
        y="label",  # Use numerical labels this time
        dim_reducer="tsne",
        style="ggplot",  # Different style
        cmap="tab10",  # Colormap good for distinct categories
        figsize=(9, 7),
        legend_max_ncols=3,
    )
    # Try to save the figure before showing
    try:
        fig_tsne_num.savefig(
            "examples/outputs/embedding_tsne_ggplot.png",
            bbox_inches="tight",
            dpi=150,
        )
        print("t-SNE plot (ggplot) saved to embedding_tsne_ggplot.png")
    except Exception as save_err:
        print(f"Could not save t-SNE plot (ggplot): {save_err}")
    plt.show()
    print("t-SNE plot (ggplot) generation attempted.")
except Exception as e:
    print(f"Could not generate t-SNE plot (ggplot): {e}")
