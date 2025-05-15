import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

# Set number of clusters
k = 3

# Initialize cluster centers randomly
def initialize_clusters(X, k, seed=23):
    np.random.seed(seed)
    clusters = {}
    for idx in range(k):
        center = 2 * (2 * np.random.random(X.shape[1]) - 1)
        clusters[idx] = {
            'center': center,
            'points': []
        }
    return clusters

# Euclidean distance
def distance(p1, p2):
    #alternative ways : np.sqrt(np.sum((p1-p2)**2))
    return np.linalg.norm(p1 - p2)

# Assign points to the nearest cluster
def assign_clusters(X, clusters):
    for x in X:
        distances = [distance(x, clusters[i]['center']) for i in range(k)]
        closest = np.argmin(distances)
        clusters[closest]['points'].append(x)
    return clusters

# Update cluster centers based on mean of assigned points
def update_clusters(clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if len(points) > 0:
            clusters[i]['center'] = points.mean(axis=0)
        clusters[i]['points'] = []
    return clusters

# Predict cluster for each point
def predict_clusters(X, clusters):
    predictions = []
    for x in X:
        distances = [distance(x, clusters[i]['center']) for i in range(k)]
        predictions.append(np.argmin(distances))
    return predictions

# Check for convergence
def has_converged(old_centers, new_centers, tol=1e-4):
    return all(np.linalg.norm(old - new) < tol for old, new in zip(old_centers, new_centers))

# K-Means training loop
clusters = initialize_clusters(X, k)

for iteration in range(100):  # max iterations
    clusters = assign_clusters(X, clusters)
    old_centers = [clusters[i]['center'].copy() for i in range(k)]
    clusters = update_clusters(clusters)
    new_centers = [clusters[i]['center'] for i in range(k)]

    if has_converged(old_centers, new_centers):
        print(f"Converged after {iteration+1} iterations.")
        break
else:
    print("Reached max iterations without full convergence.")

# Final predictions
predictions = predict_clusters(X, clusters)

# Plot final clusters
colors = ['red', 'blue', 'green']
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[np.array(predictions) == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', color='black', s=200, edgecolor='white')
plt.grid(True)
plt.title("K-Means Clustering Results (After Convergence)")
plt.legend()
plt.show()
