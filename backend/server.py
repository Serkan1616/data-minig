# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import pandas as pd
from iris_analysis import analyze_iris

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    filename = f"temp_{uuid.uuid4().hex}.csv"
    filepath = os.path.join('./uploads', filename)
    os.makedirs('./uploads', exist_ok=True)
    file.save(filepath)

    df = pd.read_csv(filepath)
    
    # Örnek olarak Iris dataset kontrolü
    if set(['Id', 'Species']).issubset(df.columns):
        plot_paths = analyze_iris(filepath)
    else:
        return jsonify({"status": "error", "message": "Unsupported dataset"}), 400

    return jsonify({
        "status": "ok",
        "plots": plot_paths
    })

if __name__ == '__main__':
    app.run(debug=True)
