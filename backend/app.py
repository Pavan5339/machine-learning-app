import os
import io
import PyPDF2 
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)
CORS(app) # Allows all origins

# --- NEW HEALTH CHECK ROUTE ---
# This gives us a URL to check if the server is running the latest code.
@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "API is running!"}), 200
# --- END OF NEW CODE ---


# Helper function to parse the text with "Title:" and "Description:"
def parse_documents_from_text(text):
    documents = []
    current_title = None
    current_description = ""
    full_text_list = []

    for line in text.split('\n'):
        line = line.strip()
        if line.lower().startswith('title:'):
            if current_title and current_description:
                documents.append({"title": current_title, "description": current_description.strip()})
                full_text_list.append(f"{current_title} {current_description.strip()}")
            current_title = line[6:].strip().replace('"', '')
            current_description = ""
        elif line.lower().startswith('description:'):
            current_description = line[12:].strip().replace('"', '')
        elif current_title and line:
            current_description += " " + line
    
    if current_title and current_description:
        documents.append({"title": current_title, "description": current_description.strip()})
        full_text_list.append(f"{current_title} {current_description.strip()}")

    return documents, full_text_list

# Helper function to perform the clustering
def perform_clustering(text_list):
    if len(text_list) < 2:
        return None, 0

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(text_list)

    max_k = min(10, len(text_list) - 1)
    if max_k < 2:
        return None, 1
    
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_test.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(X)
    
    return final_labels, best_k

# --- API Endpoints ---
@app.route('/cluster', methods=['POST'])
def cluster_text():
    try:
        data = request.get_json()
        doc_objects = data.get('documents', [])
        text_list = [f"{doc['title']} {doc['description']}" for doc in doc_objects]
        
        if len(text_list) < 2:
            return jsonify({"error": "Not enough documents to cluster."}), 400

        labels, num_clusters = perform_clustering(text_list)
        
        if labels is None:
            return jsonify({"error": "Could not perform clustering."}), 400

        results = [{"document": doc['title'], "cluster": int(labels[i])} for i, doc in enumerate(doc_objects)]
        return jsonify({"results": results, "num_clusters": num_clusters})

    except Exception as e:
        print(f"Error in /cluster: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cluster_file', methods=['POST'])
def cluster_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        text = ""
        if file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif file.filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    text += text_content
        else:
            return jsonify({"error": "Invalid file type."}), 400

        documents, text_list = parse_documents_from_text(text)
        
        if len(text_list) < 2:
            return jsonify({"error": "Not enough documents in file to cluster."}), 400
        
        labels, num_clusters = perform_clustering(text_list)
        
        if labels is None:
            return jsonify({"error": "Could not perform clustering."}), 400

        results = [{"document": doc['title'], "cluster": int(labels[i])} for i, doc in enumerate(documents)]
        return jsonify({"results": results, "num_clusters": num_clusters})

    except Exception as e:
        print(f"Error in /cluster_file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

