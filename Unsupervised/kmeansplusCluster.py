import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
X = df[[ "Annual Income (k$)","Spending Score (1-100)"]]
print(X.head())

plt.figure(figsize=(8, 6))
plt.scatter( X["Annual Income (k$)"],X["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Data Before Clustering")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling completed.")
print(X_scaled[:5])

kmeans_plus = KMeans(n_clusters=5,init="k-means++",random_state=42,n_init=10)
kmeans_plus.fit(X_scaled)
labels_kmeans_plus = kmeans_plus.labels_
print("K-Means++ cluster labels:")
print(labels_kmeans_plus)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0],X_scaled[:, 1],c=labels_kmeans_plus)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means++ Clustering")
plt.show()

wcss_plus = []
for k in range(1, 11):
    kmeans_plus = KMeans(n_clusters=k,init="k-means++",random_state=42,n_init=10)
    kmeans_plus.fit(X_scaled)
    wcss_plus.append(kmeans_plus.inertia_)
print(wcss_plus)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11),wcss_plus,marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("K-Means++ WCSS Elbow Method")
plt.xticks(range(1, 11))
plt.show()
