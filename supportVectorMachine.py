import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,classification_report)

df = pd.read_csv("breast_dataset.csv")

print("Shape of dataset:", df.shape)

X_all = df.drop(columns=["Sample code number", "Class"])
y_all = df["Class"]
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all,y_all,test_size=0.2,random_state=42,stratify=y_all)
print("Training data:", X_train_all.shape)
print("Testing data:", X_test_all.shape)

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)
print("Scaling completed.")

svm_all = SVC(kernel="sigmoid",random_state=42)

svm_all.fit(X_train_all_scaled,y_train_all)
y_pred_svm_all = svm_all.predict(X_test_all_scaled)

print("SVM Accuracy:",accuracy_score(y_test_all, y_pred_svm_all))

print("\nClassification Report:")
print(classification_report(y_test_all,y_pred_svm_all,target_names=["Benign", "Malignant"]))