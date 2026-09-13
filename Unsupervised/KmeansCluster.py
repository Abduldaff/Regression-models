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

Kmeans = KMeans(n_clusters=5,init="random",random_state=42,n_init=10)
Kmeans.fit(X_scaled)
labels_kmeans = Kmeans.labels_
print("Cluster labels:")
print(labels_kmeans)

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=labels_kmeans)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("k-means clustering")
plt.show()

wcss = []
for k in range(1,11):
    Kmeans = KMeans(n_clusters=k,init="random",random_state=42,n_init=10)
    Kmeans.fit(X_scaled)
    wcss.append(Kmeans.inertia_)
print(wcss)

plt.figure(figsize=(8,6))
plt.plot(range(1,11),wcss,marker='o')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss')
plt.title('k-means wcss elbow method')
plt.xticks(range(1,11))
# plt.grid()
plt.show()
