import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

SAVED_DIR = 'saved_models'
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)


# ------------------------------------------------------------
# HYBRID STACKING MODEL
# ------------------------------------------------------------
def get_hybrid_stacking_model():
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]

    meta_model = LogisticRegression(max_iter=1000)

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    return model


# ------------------------------------------------------------
# TRAIN + SAVE
# ------------------------------------------------------------
def _train_and_save(name, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = get_hybrid_stacking_model()

    print(f"\nTraining model for {name}...")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, os.path.join(SAVED_DIR, f"{name.lower()}_model.joblib"))
    joblib.dump(scaler, os.path.join(SAVED_DIR, f"{name.lower()}_scaler.joblib"))

    return acc


# ------------------------------------------------------------
# DIABETES TRAINING PIPELINE
# ------------------------------------------------------------
def train_diabetes(path):
    df = pd.read_csv(path)

    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

    imputer = SimpleImputer(strategy='mean')
    df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])

    if 'Outcome' not in df.columns:
        raise ValueError("Outcome column missing.")

    majority = df[df.Outcome == 0]
    minority = df[df.Outcome == 1]

    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df_bal = pd.concat([majority, minority_up])

    X = df_bal.drop('Outcome', axis=1)
    y = df_bal['Outcome']

    return _train_and_save("Diabetes", X, y)


# ------------------------------------------------------------
# GENERAL TRAINING FOR HEART + CANCER
# ------------------------------------------------------------
def train_standard(path, target_col, name, drop_cols=None):

    df = pd.read_csv(path)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    imputer = SimpleImputer(strategy="mean")
    df[df.columns] = imputer.fit_transform(df[df.columns])

    if df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return _train_and_save(name, X, y)


# ------------------------------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------------------------------
def load_model_and_scaler(name):

    model_path = os.path.join(SAVED_DIR, f"{name.lower()}_model.joblib")
    scaler_path = os.path.join(SAVED_DIR, f"{name.lower()}_scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    return joblib.load(model_path), joblib.load(scaler_path)


if __name__ == "__main__":
    print("Run app.py instead.")
