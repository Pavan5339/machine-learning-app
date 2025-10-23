import os
import io
import PyPDF2 
from flask import Flask, request, jsonify
from flask_cors import CORS  # Make sure this import is here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# --- THIS IS THE FIX ---
# We are explicitly telling the server to allow requests
# from your live frontend URL.
frontend_url = "https.machine-frontends.onrender.com"
CORS(app, resources={r"/*": {"origins": [frontend_url, "http://127.0.0.1:5500", "http://localhost:5500"]}})
# --- END OF FIX ---


# Helper function to parse the text with "Title:" and "Description:"
def parse_documents_from_text(text):
    documents = []
    current_title = None
    current_description = ""
    full_text_list = []

    for line in text.split('\n'):
        # Clean up lines
        line = line.strip()

        # Check for title
        if line.lower().startswith('title:'):
            # If we have a previous description, save it
            if current_title and current_description:
                documents.append({"title": current_title, "description": current_description.strip()})
                full_text_list.append(f"{current_title} {current_description.strip()}")

            # Start a new document
            current_title = line[6:].strip().replace('"', '')
            current_description = ""
        elif line.lower().startswith('description:'):
            current_description = line[12:].strip().replace('"', '')
        elif current_title and line: # Continue appending to the current description
            current_description += " " + line
    
    # Add the last document
    if current_title and current_description:
        documents.append({"title": current_title, "description": current_description.strip()})
        full_text_list.append(f"{current_title} {current_description.strip()}")

    return documents, full_text_list

# Helper function to perform the clustering
def perform_clustering(text_list):
    if len(text_list) < 2:
        return None, 0 # Not enough data to cluster

    # 1. Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(text_list)

    # 2. Find the best number of clusters (k)
    # We'll test k from 2 to min(10, num_samples - 1)
    max_k = min(10, len(text_list) - 1)
    if max_k < 2:
        return None, 1 # Still not enough unique data
    
    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_test.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # 3. Run final clustering with the best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(X)
    
    return final_labels, best_k

# --- API Endpoint for Text Input ---
@app.route('/cluster', methods=['POST'])
def cluster_text():
    try:
        data = request.get_json()
        # The JS sends 'documents' which is a list of {'title': ..., 'description': ...}
        doc_objects = data.get('documents', [])
        
        # Combine title and description for clustering
        text_list = [f"{doc['title']} {doc['description']}" for doc in doc_objects]
        
        if len(text_list) < 2:
            return jsonify({"error": "Not enough documents to cluster. Please provide at least 2."}), 400

        labels, num_clusters = perform_clustering(text_list)
        
        if labels is None:
            return jsonify({"error": "Could not perform clustering. Not enough unique data."}), 400

        # Format the response exactly as the JS expects
        results = []
        for i, doc in enumerate(doc_objects):
            results.append({
                "document": doc['title'], # The JS expects the 'document' key to be the title
                "cluster": int(labels[i])
            })
            
        return jsonify({"results": results, "num_clusters": num_clusters})

    except Exception as e:
        print(f"Error in /cluster: {e}")
        return jsonify({"error": str(e)}), 500

# --- API Endpoint for File Upload ---
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
            # --- THIS IS THE FIX ---
            text = file.read().decode('utf-8') # Removed the extra dot
            # --- END OF FIX ---
        elif file.filename.endswith('.pdf'):
            # Read PDF content using PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    text += text_content
        else:
            return jsonify({"error": "Invalid file type. Please upload .txt or .pdf"}), 400

        # Now that we have the text, parse it
        documents, text_list = parse_documents_from_text(text)
        
        if len(text_list) < 2:
            return jsonify({"error": "Not enough documents found in file to cluster."}), 400
        
        labels, num_clusters = perform_clustering(text_list)
        
        if labels is None:
            return jsonify({"error": "Could not perform clustering. Not enough unique data."}), 400

        results = []
        for i, doc in enumerate(documents):
            results.append({
                "document": doc['title'],
                "cluster": int(labels[i])
            })
            
        return jsonify({"results": results, "num_clusters": num_clusters})

    except Exception as e:
        print(f"Error in /cluster_file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable Render provides
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



