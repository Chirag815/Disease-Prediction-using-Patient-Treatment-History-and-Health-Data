from models import train_diabetes, train_standard
import os

DATA_DIR = "data"

print("Training Diabetes...")
train_diabetes(os.path.join(DATA_DIR, "diabetes.csv"))

print("\nTraining Heart...")
train_standard(os.path.join(DATA_DIR, "heart_disease.csv"), "TenYearCHD", "Heart")

print("\nTraining Cancer...")
train_standard(os.path.join(DATA_DIR, "breast_cancer.csv"), "target", "Cancer")

print("\nAll models trained successfully!")
