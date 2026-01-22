import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
dataset_path = '../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'
print(f"Loading dataset from: {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print("\n❌ ERROR: File not found! Check the filename in 'dataset' folder.")
    exit()

# 2. DATA CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# 3. ENCODING
le = LabelEncoder()
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Churn']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Encode Target (Yes/No -> 1/0)
df['Churn'] = le.fit_transform(df['Churn'])

# 4. SPLIT DATA
X = df.drop(columns=['Churn'])
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. DEFINE THE CONTENDERS (The Algorithms)
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine (SVM)": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 6. TRAIN AND COMPARE
print("\n⚔️  STARTING MODEL BATTLE... ⚔️\n")

best_model = None
best_accuracy = 0.0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Test it
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    
    print(f"   --> Accuracy: {acc:.2f}%")
    
    # Check if this is the new winner
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print("\n" + "="*30)
print(f"🏆 WINNER: {best_model_name}")
print(f"🎯 ACCURACY: {best_accuracy:.2f}%")
print("="*30)

# 7. SAVE THE WINNER
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n💾 Saved {best_model_name} as 'model.pkl' for the website to use.")