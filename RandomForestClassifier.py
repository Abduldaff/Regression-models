import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report)

df = pd.read_csv("breast_dataset.csv")

print("Shape of dataset:", df.shape)

X_all = df.drop(columns=["Sample code number", "Class"])
y_all = df["Class"]
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all,y_all,test_size=0.2,random_state=42,stratify=y_all)
print("Training data:", X_train_all.shape)
print("Testing data:", X_test_all.shape)

selected_features = [
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Single Epithelial Cell Size"
]
X_three = df[selected_features]
y_three = df["Class"]
print("Selected features:")
print(X_three.columns)
print("\nShape:", X_three.shape)

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)
print("Scaling completed.")

rf_all = RandomForestClassifier(n_estimators=100,random_state=42)
rf_all.fit(X_train_all,y_train_all)

y_pred_rf_all = rf_all.predict(X_test_all)

print("Random Forest Accuracy:",accuracy_score(y_test_all, y_pred_rf_all))

print("\nClassification Report:")
print(classification_report(y_test_all,y_pred_rf_all,target_names=["Benign", "Malignant"]))

