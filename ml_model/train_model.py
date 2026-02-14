import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)

# --- CONFIGURATION ---
DATA_PATH = 'dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'
ARTIFACTS_DIR = 'ml_model/'
PLOTS_DIR = 'ml_model/plots/'

# Create directories if they don't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("🚀 STARTING PROFESSIONAL ML PIPELINE...\n")

# =========================================
# 1. DATA LOADING & INSPECTION
# =========================================
print("Step 1: Loading Data...")
if not os.path.exists(DATA_PATH):
    # Try looking one level up just in case
    DATA_PATH = '../' + DATA_PATH

try:
    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Data Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"   ❌ ERROR: Could not find {DATA_PATH}. Check your folder structure.")
    exit()

# =========================================
# 2. DATA CLEANING & PREPROCESSING
# =========================================
print("\nStep 2: Cleaning Data...")

# Fix TotalCharges (Convert space strings to NaN, then float)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle Missing Values (Impute with Median for robustness)
print(f"   ℹ️ Missing values found in TotalCharges: {df['TotalCharges'].isnull().sum()}")
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop irrelevant ID column
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Convert Target to Binary (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("   ✅ Data Cleaned.")

# =========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================
print("\nStep 3: Generating EDA Charts (Saved to ml_model/plots/)...")

# A. Churn Distribution Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Class Distribution (Churn vs Non-Churn)')
plt.savefig(f'{PLOTS_DIR}churn_distribution.png')
plt.close()
print("   📊 Saved churn_distribution.png")

# B. Correlation Heatmap (Numerical Features)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.savefig(f'{PLOTS_DIR}correlation_heatmap.png')
plt.close()
print("   📊 Saved correlation_heatmap.png")

# =========================================
# 4. FEATURE ENGINEERING
# =========================================
print("\nStep 4: Feature Engineering...")

# Separate Features and Target
X = df.drop(columns=['Churn'])
y = df['Churn']

# One-Hot Encoding for Categorical Variables
# (drop_first=True helps reduce multicollinearity)
X = pd.get_dummies(X, drop_first=True) 

# Save column names to ensure web app matches training input exactly
model_columns = list(X.columns)

# Scaling (Normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"   ✅ Training Set: {X_train.shape}")
print(f"   ✅ Testing Set: {X_test.shape}")

# =========================================
# 5. MODEL TRAINING & COMPARISON
# =========================================
print("\nStep 5: Training Models...")

# Define Models
# Note: class_weight='balanced' helps with the Churn imbalance problem
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
}

results = {}
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"   🔄 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2)
    }
    
    # We choose winner based on F1 Score (better for imbalance) or Accuracy
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

print(f"\n🏆 WINNER: {best_name} with {round(best_score*100, 2)}% Accuracy")

# =========================================
# 6. MODEL EVALUATION & VISUALIZATION
# =========================================
print("\nStep 6: Saving Evaluation Metrics...")

# Generate Confusion Matrix for the Winner
winner_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, winner_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_name}')
plt.savefig(f'{PLOTS_DIR}confusion_matrix.png')
plt.close()
print("   📊 Saved confusion_matrix.png")

# Generate ROC Curve
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_name}')
plt.legend(loc="lower right")
plt.savefig(f'{PLOTS_DIR}roc_curve.png')
plt.close()
print("   📊 Saved roc_curve.png")

# =========================================
# 7. FEATURE IMPORTANCE (Explainability)
# =========================================
print("\nStep 7: Extracting AI Insights...")

# We use Random Forest for feature importance regardless of winner for better explainability
rf_model = models["Random Forest"] 
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 5 Features
top_features = [X.columns[i] for i in indices[:5]]
top_scores = importances[indices][:5]

importance_data = {
    "labels": top_features,
    "data": [round(x * 100, 1) for x in top_scores]
}

print(f"   💡 Top Factor: {top_features[0]}")

# =========================================
# 8. SAVING ARTIFACTS
# =========================================
print("\nStep 8: Saving Files for Web App...")

# 1. Model
with open(f'{ARTIFACTS_DIR}model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# 2. Scaler
with open(f'{ARTIFACTS_DIR}scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Column Names (Crucial for One-Hot Encoding alignment)
with open(f'{ARTIFACTS_DIR}model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# 4. Metrics JSON
final_metrics = {
    "models": results,
    "winner": best_name,
    "importance": importance_data
}
with open(f'{ARTIFACTS_DIR}metrics.json', 'w') as f:
    json.dump(final_metrics, f)

print("\n✅ PIPELINE COMPLETE. READY FOR WEB APP.")