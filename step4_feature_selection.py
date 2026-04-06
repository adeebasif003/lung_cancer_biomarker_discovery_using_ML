#step4_feature_selection



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

# Load processed data 
expr = pd.read_csv("tcga_luad_expression.csv", index_col=0)
labels = pd.read_csv("tcga_luad_labels.csv", index_col=0).squeeze("columns")

# Encode labels (Tumor=1, Normal=0)
le = LabelEncoder()
y = le.fit_transform(labels)

print(" Data loaded and labels encoded.")
print(f"Expression data shape: {expr.shape}")
print(f"Classes: {list(le.classes_)}")

# 1. Statistical Feature Selection: ANOVA F-test
print("\n Performing ANOVA feature selection...")
f_values, p_values = f_classif(expr, y)

anova_results = pd.DataFrame({
    "Gene": expr.columns,
    "F_value": f_values,
    "p_value": p_values
}).sort_values("p_value")

# Select top 300 most significant genes
top_anova = anova_results.head(300)
top_anova.to_csv("top300_anova_genes.csv", index=False)
print(" Saved top 300 ANOVA-selected genes → top300_anova_genes.csv")

# 2. ML-Based Feature Selection: Random Forest Importance 
print("\n Performing Random Forest feature selection...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(expr, y)

importances = pd.DataFrame({
    "Gene": expr.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

# Select top 300 most important genes
top_rf = importances.head(300)
top_rf.to_csv("top300_rf_genes.csv", index=False)
print(" Saved top 300 Random Forest-selected genes → top300_rf_genes.csv")

# 3. Merge and Save Common Biomarkers
common_genes = pd.merge(top_anova, top_rf, on="Gene", how="inner")
common_genes.to_csv("common_biomarker_genes.csv", index=False)
print(f"\n Common biomarker genes saved: {len(common_genes)} found")
print("File → common_biomarker_genes.csv")

#  Summary
print("\n Summary:")
# print(f"ANOVA-selected genes: {len(top_anova)}")
# print(f"RandomForest-selected genes: {len(top_rf)}")
# print(f"Common genes: {len(common_genes)}")
# print("Feature selection completed successfully ")
#  Print ALL Selected Gene Names

print("\n All ANOVA Selected Genes:")
print(top_anova["Gene"].tolist())

print("\n All Random Forest Selected Genes:")
print(top_rf["Gene"].tolist())

print("\n All Common Biomarker Genes:")
print(common_genes["Gene"].tolist())

print("\n Feature selection completed successfully ")