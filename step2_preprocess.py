#step2_preprocess


import pandas as pd
import numpy as np

# Load the expression data 
expr_file = "TCGA.LUAD.sampleMap_HiSeqV2.gz"
print("Loading gene expression data...")
expr = pd.read_csv(expr_file, sep="\t", index_col=0)

# Transpose so that samples are rows and genes are columns
expr = expr.T
print("Expression data shape:", expr.shape)

# Load the clinical datachat
clinical_file = "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
print("Loading clinical data...")
clinical = pd.read_csv(clinical_file, sep="\t", index_col=0)
print("Clinical data shape:", clinical.shape)

# Extract tumor/normal sample types
# TCGA sample IDs: last 2 digits indicate sample type
# 01 = primary tumor, 11 = normal tissue
def get_sample_type(sample_id):
    try:
        code = int(sample_id.split("-")[3][:2])
        if code == 1:
            return "Tumor"
        elif code == 11:
            return "Normal"
        else:
            return "Other"
    except:
        return "Other"

expr["SampleType"] = expr.index.map(get_sample_type)

# Filter only tumor and normal samples
expr = expr[expr["SampleType"].isin(["Tumor", "Normal"])]

# Separate labels
labels = expr["SampleType"]
expr = expr.drop(columns=["SampleType"])

print("Filtered expression shape:", expr.shape)
print(labels.value_counts())

# Log2 normalization (already log2(norm_count+1), but ensure numeric type)
expr = expr.apply(pd.to_numeric, errors='coerce')
expr = expr.fillna(0)

# Save processed data
expr.to_csv("tcga_luad_expression.csv")
labels.to_csv("tcga_luad_labels.csv")

print(" Preprocessing complete.")
print("Saved files: tcga_luad_expression.csv, tcga_luad_labels.csv")
