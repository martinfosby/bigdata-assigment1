import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, Birch, DBSCAN, OPTICS

# Define datasets
datasets = {
    "Blobs": make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42),
    "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
    "Circles": make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
}

# Define clustering algorithms
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "MeanShift": MeanShift(),
    "BIRCH": Birch(n_clusters=4),
    "DBSCAN": DBSCAN(),
    "OPTICS": OPTICS()
}

# Helper function to plot results
def plot_clusters(X, labels, title, ax):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    ax.set_title(title)
    ax.grid(True)

# Iterate over datasets
for dataset_name, (X, y) in datasets.items():
    # Standardize the dataset
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA (2 components for visualization)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    
    fig, axes = plt.subplots(len(clustering_algorithms), 2, figsize=(12, 8))
    fig.suptitle(f"Dataset: {dataset_name}", fontsize=16)
    
    for i, (algo_name, algo) in enumerate(clustering_algorithms.items()):
        # Without PCA
        algo.fit(X_scaled)
        plot_clusters(X_scaled, algo.labels_, f"{algo_name} without PCA", axes[i, 0])
        
        # With PCA
        algo.fit(X_pca)
        plot_clusters(X_pca, algo.labels_, f"{algo_name} with PCA", axes[i, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
