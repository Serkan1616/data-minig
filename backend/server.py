from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import pandas as pd
from iris_analysis import analyze_iris
from titanic_analysis import analyze_titanic  # Yeni analiz fonksiyonunu ekle

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

    # Hangi analiz fonksiyonu kullanılacak?
    if set(['Id', 'Species']).issubset(df.columns):
        plot_paths = analyze_iris(filepath)
    elif set(['Survived', 'Sex', 'Age', 'Fare']).issubset(df.columns):  # Titanic için basit kontrol
        plot_paths = analyze_titanic(filepath)
    else:
        return jsonify({"status": "error", "message": "Unsupported dataset"}), 400

    return jsonify({
        "status": "ok",
        "plots": plot_paths
    })

if __name__ == '__main__':
    app.run(debug=True)
