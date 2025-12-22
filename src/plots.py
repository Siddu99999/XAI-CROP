"""
XAI-CROP: Performance Visualization

Generates performance and comparison plots
similar to those presented on page 15 of the research paper.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# Load dataset and encoders
# --------------------------------------------------
df = pd.read_csv("../data/cropds_cleaned.csv")

district_encoder = joblib.load("../models/district_encoder.pkl")
season_encoder = joblib.load("../models/season_encoder.pkl")
crop_encoder = joblib.load("../models/crop_encoder.pkl")

df["District_Name"] = district_encoder.transform(df["District_Name"])
df["Season"] = season_encoder.transform(df["Season"])
df["Crop"] = crop_encoder.transform(df["Crop"])

X = df.drop("Crop", axis=1)
y = df["Crop"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Train models for comparison (paper-style)
# --------------------------------------------------
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

models = {
    "Decision Tree": dt,
    "Random Forest": rf
}

mse_vals, mae_vals, r2_vals = [], [], []

for model in models.values():
    y_pred = model.predict(X_test)
    mse_vals.append(mean_squared_error(y_test, y_pred))
    mae_vals.append(mean_absolute_error(y_test, y_pred))
    r2_vals.append(r2_score(y_test, y_pred))

# --------------------------------------------------
# Plot 1: Error Metrics Comparison (Paper-style)
# --------------------------------------------------
plt.figure(figsize=(8, 5))
x = np.arange(len(models))

plt.bar(x - 0.25, mse_vals, width=0.25, label="MSE")
plt.bar(x, mae_vals, width=0.25, label="MAE")
plt.bar(x + 0.25, r2_vals, width=0.25, label="R²")

plt.xticks(x, models.keys())
plt.ylabel("Metric Value")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("../results/model_performance_comparison.png")
plt.show()

# --------------------------------------------------
# Plot 2: Actual vs Predicted (Encoded labels)
# --------------------------------------------------
y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel("Actual Crop (Encoded)")
plt.ylabel("Predicted Crop (Encoded)")
plt.title("Actual vs Predicted Crop Labels")
plt.tight_layout()
plt.savefig("../results/actual_vs_predicted.png")
plt.show()

# --------------------------------------------------
# Plot 3: Feature Importance (Paper-style)
# --------------------------------------------------
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(7, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance Analysis")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../results/feature_importance.png")
plt.show()

print("Paper-style plots generated successfully.")
