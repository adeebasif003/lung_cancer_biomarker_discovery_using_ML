#step3_eda


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#  Load the processed data 
expr = pd.read_csv("tcga_luad_expression.csv", index_col=0)
labels = pd.read_csv("tcga_luad_labels.csv", index_col=0).squeeze("columns")


print("\n Data Loaded Successfully!")
print("Expression shape:", expr.shape)
print("Labels shape:", labels.shape)
print("\nLabel distribution:\n", labels.value_counts())

# Check missing data 
missing = expr.isna().sum().sum()
print("\nMissing values in expression data:", missing)

# Check variance across genes
variances = expr.var()
print("\nVariance Summary:")
print("  Min:", variances.min())
print("  Max:", variances.max())

# Remove low-variance genes
threshold = 0.01
expr_filtered = expr.loc[:, variances > threshold]
print(f"\nFiltered expression data shape: {expr_filtered.shape}")

# Standardize data
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_filtered)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expr_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Label"] = labels.values

# Plot PCA
plt.figure(figsize=(8,6))
for label, color in zip(["Tumor", "Normal"], ["red", "blue"]):
    subset = pca_df[pca_df["Label"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.6, color=color)
plt.title("PCA - TCGA-LUAD Expression Data")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
plt.legend()
plt.show()



# Class Distribution Plot

plt.figure(figsize=(6, 4))
sns.countplot(x=labels, palette="Set2")

plt.title("Class Distribution (Tumor vs Normal)")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()



#  Variance Distribution Plot

variances = expr.var()

plt.figure(figsize=(8, 5))
sns.histplot(variances, bins=60, kde=True)

plt.title("Gene Variance Distribution")
plt.xlabel("Variance Across Samples")
plt.ylabel("Number of Genes")
plt.tight_layout()
plt.show()

print("\nEDA Basic Plots Generated Successfully")

