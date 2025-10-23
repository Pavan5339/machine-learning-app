from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import os
from PyPDF2 import PdfReader
import io

app = Flask(__name__)
CORS(app)

def parse_text_to_documents(text):
    documents = []
    lines = text.split('\n')
    current_title = None
    current_description = ""

    for line in lines:
        line = line.strip()
        if line.lower().startswith('title:'):
            if current_title is not None:
                documents.append({'title': current_title, 'description': current_description.strip()})
            current_title = line[6:].strip().strip('"')
            current_description = ""
        elif line.lower().startswith('description:'):
            current_description = line[12:].strip().strip('"')
        elif current_title is not None:
            current_description += " " + line
    
    if current_title is not None:
        documents.append({'title': current_title, 'description': current_description.strip()})

    return documents

def perform_clustering(documents):
    if len(documents) < 2:
        return {'error': 'At least 2 documents are required to perform clustering'}

    titles = [doc.get('title', '') for doc in documents]
    descriptions = [doc.get('description', '') for doc in documents]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(descriptions)

    inertia = []
    max_clusters = min(11, len(descriptions))
    K = range(2, max_clusters)
    if not K:
        optimal_num_clusters = 2
    else:
        for k in K:
            kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, random_state=42)
            kmeans_model.fit(X)
            inertia.append(kmeans_model.inertia_)

        if len(K) > 1:
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

    model = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=100, n_init=1, random_state=42)
    model.fit(X)
    labels = model.labels_

    results = []
    for i, title in enumerate(titles):
        results.append({'document': title, 'cluster': int(labels[i])})

    return {'results': results, 'num_clusters': optimal_num_clusters}

@app.route('/cluster', methods=['POST'])
def cluster_text():
    data = request.get_json()
    if not data or 'documents' not in data:
        return jsonify({'error': 'Invalid input: "documents" not found in request body'}), 400
    
    results = perform_clustering(data['documents'])
    if 'error' in results:
        return jsonify(results), 400
    return jsonify(results)

@app.route('/cluster_file', methods=['POST'])
def cluster_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    text = ""
    if file.filename.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(io.BytesIO(file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            return jsonify({'error': f'Error reading PDF file: {e}'}), 500
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    else:
        return jsonify({'error': 'Unsupported file type. Please upload a .txt or .pdf file.'}), 400

    documents = parse_text_to_documents(text)
    results = perform_clustering(documents)
    if 'error' in results:
        return jsonify(results), 400
    return jsonify(results)