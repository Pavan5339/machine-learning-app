import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import numpy as np

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# --- Load Data from file ---
data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'data.txt')
with open(data_file_path, 'r') as f:
    lines = f.readlines()

# --- Parse Titles and Descriptions ---
# We extract both the title and description for each project
project_titles = []
project_descriptions = []
for line in lines:
    if line.startswith('Title: '):
        title = line.replace('Title: ', '').strip().strip('\"')
        project_titles.append(title)
    elif line.startswith('Description: '):
        # Remove the "Description: " prefix and quotes, and strip whitespace
        description = line.replace('Description: ', '').strip().strip('\"')
        project_descriptions.append(description)


# Vectorize the text data
# The vectorizer converts text into a matrix of TF-IDF features.
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(project_descriptions)

# --- Find the optimal number of clusters using the Elbow Method ---
inertia = []
K = range(2, 11)  # We'll test for 2 to 10 clusters
for k in K:
    kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, random_state=42)
    kmeans_model.fit(X)
    inertia.append(kmeans_model.inertia_)

# --- Find the elbow point ---
# We'll find the point with the maximum distance from the line connecting the first and last points.
# This is a simple way to programmatically find the "elbow".
points = np.array([list(K), inertia]).T
first_point = points[0]
last_point = points[-1]

# Calculate the line vector
line_vec = last_point - first_point
line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

# Find the distance of each point from the line
vec_from_first = points - first_point
scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(K), 1)), axis=1)
vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
vec_to_line = vec_from_first - vec_from_first_parallel
dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

# The elbow is the point with the maximum distance
optimal_num_clusters = K[np.argmax(dist_to_line)]
print(f"Optimal number of clusters found: {optimal_num_clusters}")


# --- Train the Clustering Model with the optimal number of clusters ---
num_clusters = optimal_num_clusters
model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, random_state=42)
model.fit(X)

# --- Save the Model and Vectorizer ---
# We save both the trained model and the vectorizer so we can use them later
# in the backend to make predictions on new data.
model_path = os.path.join('model', 'kmeans_model.joblib')
vectorizer_path = os.path.join('model', 'tfidf_vectorizer.joblib')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)# --- Save Cluster Assignments ---
# Save the cluster assignments for each project description.
# This can be useful for analysis or for re-associating descriptions with their clusters later. 
cluster_assignments_path = os.path.join('model', 'cluster_assignments.joblib')
joblib.dump(model.labels_, cluster_assignments_path)
print(f"Cluster assignments saved to: {cluster_assignments_path}")


print("Model training complete.")
print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

# --- Optional: Display clustering results ---
print("\n--- Clustering Results ---")
# Create a dictionary to hold the clusters
clusters = {i: [] for i in range(num_clusters)}
for i, label in enumerate(model.labels_):
    clusters[label].append(project_titles[i])  # Use titles for display

# Print the content of each cluster
for cluster_id, projects in clusters.items():
    print(f"\nCluster {cluster_id + 1}:")
    for project_title in projects:
        print(f"- {project_title}")
