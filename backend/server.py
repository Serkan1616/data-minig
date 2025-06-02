from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from iris_analysis import analyze_iris
from titanic_analysis import analyze_titanic
from analyze_creditcard import analyze_creditcard
from wine_analysis import analyze_wine  # Add this import

app = Flask(__name__, static_url_path='/static', static_folder='uploads')
CORS(app)


# Absolute paths for folders
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'uploads')
DATASET_FOLDER = os.path.join(CURRENT_DIR, 'datasets')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    dataset_file = data.get('dataset')
    
    if not dataset_file:
        return jsonify({"status": "error", "message": "No dataset specified"}), 400

    dataset_path = os.path.join(app.config['DATASET_FOLDER'], dataset_file)
    
    if not os.path.exists(dataset_path):
        return jsonify({"status": "error", "message": "Dataset not found"}), 404

    # Make sure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Choose analysis based on dataset
    if dataset_file == 'iris.csv':
        plot_paths = analyze_iris(dataset_path)
    elif dataset_file == 'train.csv':
        plot_paths = analyze_titanic(dataset_path)
    elif dataset_file == 'creditcard.csv':
        plot_paths = analyze_creditcard(dataset_path)
    elif dataset_file == 'winequalityN.csv':  # Add this condition
        plot_paths = analyze_wine(dataset_path)
    else:
        return jsonify({"status": "error", "message": "Unsupported dataset"}), 400

    return jsonify({
        "status": "ok",
        "plots": plot_paths
    })

if __name__ == '__main__':
    app.run(debug=True)
