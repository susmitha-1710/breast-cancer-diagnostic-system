# ==========================================
# Breast Cancer Diagnostic System
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# Data Cleaning
# ==========================================

# Remove unnecessary columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

print("\nColumns after cleaning:")
print(df.columns)

# ==========================================
# Encode Diagnosis Column
# M = Malignant
# B = Benign
# ==========================================

le = LabelEncoder()

df["diagnosis"] = le.fit_transform(df["diagnosis"])

print("\nEncoded Diagnosis Values:")
print(df["diagnosis"].value_counts())

# ==========================================
# Features and Target
# ==========================================

X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Model Training
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:")
print(feature_df.head(10))

# ==========================================
# Feature Importance Plot
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df.head(10)
)

plt.title("Top 10 Important Features")

plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "models/breast_cancer_model.pkl"
)

print("\nModel saved successfully!")