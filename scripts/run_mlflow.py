import mlflow
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DAGSHUB_USERNAME = 'ruchitha-meenakshi'
REPO_NAME = 'CodeX_VirtualInternship'

# 1. Initialize Connection
print("Connecting to DagsHub...")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("CodeX_Pricing_Strategy")

# 2. Load Data (Reads from the local repository (from the laptop))
print("Loading local data...")
try:
    df = pd.read_csv('survey_results_final.csv')
except FileNotFoundError:
    print("Error: 'survey_results_final.csv' not found. Please ensure the file is in this folder.")
    exit()

# 3. Prepare Data
X = df.drop(columns=['respondent_id', 'price_range'])
y = df['price_range']

# Encode Categorical Columns
le = LabelEncoder()
target_le = LabelEncoder()

# Label Encode specific columns
for col in ['age_group', 'income_levels', 'health_concerns', 'consume_frequency(weekly)', 'preferable_consumption_size']:
    if col in X.columns:
        X[col] = le.fit_transform(X[col])

# Label Encode Target
y = target_le.fit_transform(y)

# One-Hot Encode the rest
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Define Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42)
}

# 5. Train & Log to MLflow
print("\nStarting Training & Logging...")
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log Parameters and Metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        
        # Log the model (Safe: This is just math, not data)
        mlflow.sklearn.log_model(model, "model")
        
        print(f" {name}: {acc:.4f} (Logged to DagsHub)")

print(f"\nSuccess! View your results here: https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
