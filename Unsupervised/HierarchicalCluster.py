import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")

print("Shape of dataset:", df.shape)
print("First 5 rows:")
print(df.head())

X = df[["Annual Income (k$)","Spending Score (1-100)"]]
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling completed.")
print(X_scaled[:5])

hierarchical_ward = AgglomerativeClustering(n_clusters=5,linkage="ward")
labels_ward = hierarchical_ward.fit_predict(X_scaled)
print("Ward cluster labels:")
print(labels_ward)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0],X_scaled[:, 1],c=labels_ward)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering - Ward Linkage")
plt.show()

hierarchical_complete = AgglomerativeClustering(n_clusters=5,linkage="complete",metric="euclidean")
labels_complete = hierarchical_complete.fit_predict(X_scaled)
print("Complete linkage labels:")
print(labels_complete)

hierarchical_average = AgglomerativeClustering(n_clusters=5,linkage="average",metric="euclidean")
labels_average = hierarchical_average.fit_predict(X_scaled)
print("Average linkage labels:")
print(labels_average)

hierarchical_single = AgglomerativeClustering(n_clusters=5,linkage="single",metric="euclidean")
labels_single = hierarchical_single.fit_predict(X_scaled)
print("Single linkage labels:")
print(labels_single)

linkages = [
    "ward",
    "complete",
    "average",
    "single"
]
for linkage_type in linkages:
    if linkage_type == "ward":
        model = AgglomerativeClustering(n_clusters=5,linkage=linkage_type)
    else:
        model = AgglomerativeClustering(n_clusters=5,linkage=linkage_type,metric="euclidean")
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled,labels)
    print(linkage_type,"→ Silhouette Score:",round(score, 4))

linkage_matrix = linkage(X_scaled,method="ward")
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram - Ward")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()