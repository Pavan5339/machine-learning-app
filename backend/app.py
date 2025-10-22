from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.get_json()
    if not data or 'documents' not in data:
        return jsonify({'error': 'Invalid input: "documents" not found in request body'}), 400

    documents = data['documents']
    if len(documents) < 2:
        return jsonify({'error': 'At least 2 documents are required to perform clustering'}), 400

    # --- Vectorize the text data ---
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(documents)

    # --- Find the optimal number of clusters using the Elbow Method ---
    inertia = []
    max_clusters = min(11, len(documents)) # Cannot have more clusters than documents
    K = range(2, max_clusters)
    if not K: # if max_clusters is 2, K will be empty
        optimal_num_clusters = 2
    else:
        for k in K:
            kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, random_state=42)
            kmeans_model.fit(X)
            inertia.append(kmeans_model.inertia_)

        if len(K) > 1:
            # --- Find the elbow point ---
            points = np.array([list(K), inertia]).T
            first_point = points[0]
            last_point = points[-1]
            line_vec = last_point - first_point
            line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
            vec_from_first = points - first_point
            scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(K), 1)), axis=1)
            vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
            vec_to_line = vec_from_first - vec_from_first_parallel
            dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
            optimal_num_clusters = K[np.argmax(dist_to_line)]
        else:
            optimal_num_clusters = 2


    # --- Train the final model and get cluster assignments ---
    model = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=100, n_init=1, random_state=42)
    model.fit(X)
    labels = model.labels_

    # --- Prepare the results ---
    results = []
    for i, doc in enumerate(documents):
        results.append({'document': doc, 'cluster': int(labels[i])})

    return jsonify({'results': results, 'num_clusters': optimal_num_clusters})

if __name__ == '__main__':
    app.run(debug=True, port=5000)