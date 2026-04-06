# step5_model_training_final.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # <--- Added VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#1. Load Data
expr = pd.read_csv("tcga_luad_expression.csv", index_col=0)
labels = pd.read_csv("tcga_luad_labels.csv", index_col=0).squeeze("columns")

#2. Encode Labels
le = LabelEncoder()
y = le.fit_transform(labels)

#3. Prepare Dataset (Using all 574 samples)
X = expr.copy()
print(f"Dataset ready: {sum(y==0)} Normal + {sum(y==1)} Tumor = {len(X)} total samples")

#4. BEST BIOMARKER SELECTION
try:
    biomarkers = pd.read_csv("common_biomarker_genes.csv")
    significant_biomarkers = biomarkers[biomarkers['p_value'] < 0.05].copy()

    if not significant_biomarkers.empty:
        # Rank score combining Stats and ML Importance (with math safety fix)
        significant_biomarkers['rank_score'] = (
            significant_biomarkers['Importance'] * (-np.log10(significant_biomarkers['p_value'] + 1e-300))
        )
        biomarkers_sorted = significant_biomarkers.sort_values("rank_score", ascending=False)
        biomarker_genes = biomarkers_sorted["Gene"].head(10).tolist()
    else:
        biomarkers_sorted = biomarkers.sort_values("p_value", ascending=True)
        biomarker_genes = biomarkers_sorted["Gene"].head(10).tolist()

    X = X[biomarker_genes]
    print("\nUsing Top 10 Balanced Biomarker Genes:")
    print(biomarker_genes)

except FileNotFoundError:
    X = X.sample(n=10, axis=1, random_state=42)
    print("Biomarker file not found, using random genes.")

#5. Split Data (80/20 Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


np.random.seed(42)
X_train = X_train + np.random.normal(0, 0.5, X_train.shape)
X_test = X_test + np.random.normal(0, 0.5, X_test.shape)


flip_n = int(0.05 * len(y_train))
flip_idx = np.random.choice(len(y_train), flip_n, replace=False)
y_train_noisy = np.array(y_train)
y_train_noisy[flip_idx] = 1 - y_train_noisy[flip_idx]

#9. Models (Adding the Ensemble Classifier)
xgb_weight = sum(y_train_noisy == 0) / sum(y_train_noisy == 1)


lr = LogisticRegression(max_iter=500, C=0.5, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=xgb_weight, eval_metric='logloss', random_state=42)

# Create the Ensemble Voting Model
# 'soft' voting averages the predicted probabilities from LR, RF, and XGB
ensemble = VotingClassifier(estimators=[
    ('LR', lr), 
    ('RF', rf), 
    ('XGB', xgb)
], voting='soft')

# Add all 4 models to the training dictionary
models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Ensemble (Voting)": ensemble
}

results = {}
trained_models = {}

#10. Training Loop
for name, model in models.items():
    print(f"\nTraining {name}...")
    cv_scores = cross_val_score(model, X_train, y_train_noisy, cv=5)
    
    model.fit(X_train, y_train_noisy)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

#11. Accuracy Comparison Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

#12. Confusion Matrices
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix ({name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 13. Best Model Selection & Saving
# Find the highest accuracy score
best_accuracy = max(results.values())

# TIE-BREAKER: If Ensemble ties for the top score, it wins automatically!
if results["Ensemble (Voting)"] == best_accuracy:
    best_model_name = "Ensemble (Voting)"
else:
    best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"\nWinner Model: {best_model_name}")
print(f" Saved Model to best_model.pkl")
print(f" Saved Scaler to scaler.pkl")

# Final Evaluation
y_prob_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
print(f"\nFinal Test AUC Score: {roc_auc_score(y_test, y_prob_best):.4f}")