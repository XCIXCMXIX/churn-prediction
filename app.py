import json, math
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
M = json.loads((Path(__file__).parent / "model.json").read_text())

@app.route("/")
def home():
    return render_template("index.html", m=M)

@app.post("/predict")
def predict():
    d = request.get_json()
    z = M["b"]
    z += sum(M["num"][c] * float(d[c]) for c in M["num"])
    z += sum(M["cat"][c][d[c]] for c in M["cat"])
    p = 1 / (1 + math.exp(-z))
    return jsonify(prob=round(p * 100, 1),
                   label="Likely to CHURN" if p >= 0.5 else "Likely to STAY")