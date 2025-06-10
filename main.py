import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def cluster_with_hdbscan_gmm(input_path, output_path, n_features, fig_path):
    print(f"Processing: {input_path}")

    # Load and preprocess
    df = pd.read_csv(input_path)
    X = df[[str(i + 1) for i in range(n_features)]].values
    X_powered = PowerTransformer(method='yeo-johnson').fit_transform(X)
    X_std = StandardScaler().fit_transform(X_powered)

    # HDBSCAN for structure (not used directly)
    hdb = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=5)
    _ = hdb.fit_predict(X_std)

    # Gaussian Mixture with 4n - 1 clusters
    n_clusters = 4 * n_features - 1
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='tied',
        random_state=42,
        max_iter=400,
        n_init=10
    )
    labels = gmm.fit_predict(X_std)

    # Save output
    df["label"] = labels
    df_out = df[["id", "label"]].sort_values("id")
    df_out.to_csv(output_path, index=False)
    print(f"Done: {output_path}, {n_clusters} clusters.")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', s=10)
    plt.title(f"PCA Clustering Visualization - {input_path}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

# Run for both datasets
cluster_with_hdbscan_gmm(
    input_path="public_data.csv",
    output_path="public_submission.csv",
    n_features=4,
    fig_path="public_clusters.png"
)

cluster_with_hdbscan_gmm(
    input_path="private_data.csv",
    output_path="private_submission.csv",
    n_features=6,
    fig_path="private_clusters.png"
)
