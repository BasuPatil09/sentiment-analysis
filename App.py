from flask import Flask, request, jsonify, render_template
from src.predictor import SentimentPredictor

app = Flask(__name__)
MODEL_PATH = "outputs/sentiment_pipeline_logistic_regression.joblib"
predictor  = SentimentPredictor.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text"}), 400
    result = predictor.predict_one(text, verbose=False)
    return jsonify(result)

@app.route("/compare")
def compare():
   models = {
    "Logistic Regression": {"accuracy": 0.9024, "f1_macro": 0.9024, "roc_auc": 0.9635},
    "Linear SVM":          {"accuracy": 0.9040, "f1_macro": 0.9040, "roc_auc": 0.9652},
    "Naive Bayes":         {"accuracy": 0.8840, "f1_macro": 0.8840, "roc_auc": 0.9533},
    "Random Forest":       {"accuracy": 0.8572, "f1_macro": 0.8572, "roc_auc": 0.9356},
    "SGD Classifier":      {"accuracy": 0.9060, "f1_macro": 0.9060, "roc_auc": 0.9660},
    }
   return jsonify(models)

if __name__ == "__main__":
    app.run(debug=True)