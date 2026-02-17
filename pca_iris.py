# ==========================================
# 1. Import Libraries
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 2. Load Dataset
# ==========================================
df = pd.read_csv("Iris.csv")

# Separate features and target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

print("Original Shape:", X.shape)

# ==========================================
# 3. Standardization
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nMean After Standardization (approx 0):")
print(np.mean(X_scaled, axis=0))

# ==========================================
# 4. Covariance Matrix
# ==========================================
cov_matrix = np.cov(X_scaled.T)

print("\nCovariance Matrix:")
print(cov_matrix)

# ==========================================
# 5. Eigenvalues and Eigenvectors
# ==========================================
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigen Values:")
print(eigen_values)

print("\nEigen Vectors:")
print(eigen_vectors)

# ==========================================
# 6. Sort Eigenvalues (Descending)
# ==========================================
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

print("\nSorted Eigen Values:")
print(sorted_eigenvalues)

# ==========================================
# 7. Explained Variance
# ==========================================
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

print("\nExplained Variance Ratio:")
print(explained_variance)

# ==========================================
# 8. Select First 2 Principal Components
# ==========================================
eigenvector_subset = sorted_eigenvectors[:, 0:2]

# Transform Data
X_pca = np.dot(X_scaled, eigenvector_subset)

print("\nReduced Data Shape:", X_pca.shape)

# ==========================================
# 9. Convert to DataFrame
# ==========================================
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Species"] = y

# ==========================================
# 10. Visualization
# ==========================================
plt.figure(figsize=(8,6))

colors = {"Iris-setosa":"red",
          "Iris-versicolor":"green",
          "Iris-virginica":"blue"}

for species in colors:
    subset = pca_df[pca_df["Species"] == species]
    plt.scatter(subset["PC1"],
                subset["PC2"],
                c=colors[species],
                label=species,
                s=60)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Manual PCA Visualization (Iris Dataset)")
plt.legend()
plt.grid()
plt.show()

# ==========================================
# 11. Explained Variance Bar Graph
# ==========================================
plt.figure(figsize=(6,4))
plt.bar(["PC1", "PC2", "PC3", "PC4"], explained_variance)
plt.title("Explained Variance by Each Principal Component")
plt.ylabel("Variance Ratio")
plt.show()
