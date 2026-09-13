"""
train_model.py
---------------
Loads data/student_stress.csv, trains a Random Forest classifier to
predict stress_level, prints accuracy, and saves the trained model +
encoders to the model/ folder for later use in the Streamlit app.

Run:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("data/student_stress.csv")

# 2. Split features (X) and target (y)
X = df.drop(columns=["stress_level"])
y = df["stress_level"]

# Encode the target labels (Low/Medium/High -> 0/1/2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Save model + encoder + feature names for the app
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

print("\nModel saved to model/rf_model.pkl")
