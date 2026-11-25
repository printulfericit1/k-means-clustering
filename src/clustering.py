import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Ensure output directories exist
os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/Country-data.csv")
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe().round(2))

# 2. Clean data and select numerical features
df = df.dropna().copy()
country_col = "country"
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols]

# 3. EDA: Feature correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png", dpi=150)
plt.close()

# 4. Feature standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Evaluate multiple values of K (Elbow Method & Silhouette Score)
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_range, inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(k_range, sil_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score vs. K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("images/elbow_silhouette.png", dpi=150)
plt.close()

# 6. Train final K-Means model with K = 2
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, df["Cluster"])
print(f"\nFinal Model (K={optimal_k}) Silhouette Score: {final_silhouette:.4f}")
print("\nCluster sizes:\n", df["Cluster"].value_counts())
print("\nCluster feature means:\n", df.groupby("Cluster")[numeric_cols].mean().round(2))

# 7. 2D Visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

explained_var = pca.explained_variance_ratio_.sum() * 100

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="viridis",
    s=75,
    alpha=0.85
)
plt.title(f"Country Clusters (K={optimal_k}) - PCA Projection ({explained_var:.1f}% Variance Explained)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("images/clusters_pca.png", dpi=150)
plt.close()

# 8. Display sample countries from each cluster
for c in sorted(df["Cluster"].unique()):
    sample_countries = df[df["Cluster"] == c][country_col].head(10).tolist()
    print(f"\nCluster {c} sample countries:", sample_countries)

# 9. Save clustered dataset
df.drop(columns=["PC1", "PC2"]).to_csv("data/clustered_countries.csv", index=False)
print("\nDone! Plots saved to 'images/' and results saved to 'data/clustered_countries.csv'.")