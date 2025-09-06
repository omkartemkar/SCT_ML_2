import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------
# 1. Load the dataset
# -------------------------
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# -------------------------
# 2. Select features (Annual Income, Spending Score)
# -------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------
# 3. Elbow method to find optimal k
# -------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# -------------------------
# 4. Apply K-Means with k=5 (from elbow method)
# -------------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# -------------------------
# 5. Visualize clusters
# -------------------------
plt.figure(figsize=(10, 6))

plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', marker='*', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# -------------------------
# 6. Save results to CSV (Optional)
# -------------------------
df['Cluster'] = y_kmeans
df.to_csv("Clustered_Customers.csv", index=False)
print("Clustered data saved as Clustered_Customers.csv")
