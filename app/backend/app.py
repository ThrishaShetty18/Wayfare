"""
Wayfare API — a small Flask backend that loads the trained neural network
once at startup and serves live recommendations to the frontend over HTTP.
This is the "real" full-stack version: the frontend never sees the model
weights — it only calls /api/recommend and gets back a ranked JSON list.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 in a browser.
"""
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from inference import get_model, QUESTIONS

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False  # fine for same-origin local use; `pip install flask-cors` for cross-origin

QUESTIONS_BY_KEY = {q["key"]: q for q in QUESTIONS}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
if _HAS_CORS:
    CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    model = get_model()
    return jsonify({"status": "ok", "num_destinations": model.n_dest})


@app.route("/api/questions")
def questions():
    """Frontend fetches the survey schema from the backend, not hardcoded twice."""
    return jsonify(QUESTIONS)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(force=True) or {}
    answers = payload.get("answers", {})
    top_k = int(payload.get("top_k", 10))
    top_k = max(1, min(top_k, 50))

    if not isinstance(answers, dict):
        return jsonify({"error": "answers must be an object of {form_key: [selected_indices]}"}), 400

    model = get_model()
    try:
        results = model.recommend(answers, top_k=top_k)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"recommendations": results, "num_destinations_scored": model.n_dest})


@app.route("/api/destinations")
def destinations():
    """Optional: full destination list, for search/autocomplete features."""
    model = get_model()
    return jsonify({"count": model.n_dest, "names": model.dest_names})


@app.route("/api/destination/<int:dest_id>")
def destination_detail(dest_id):
    """Everything the destination page needs: name, country, and the
    dataset's descriptive fields for that place. Doesn't include a score —
    scores only exist relative to a specific survey answer set."""
    model = get_model()
    dest = model.get_destination(dest_id)
    if dest is None:
        return jsonify({"error": "No destination with that id"}), 404
    return jsonify(dest)


@app.route("/api/estimate", methods=["POST"])
def estimate():
    """Estimated trip cost in USD, built from the traveller's own survey
    answers (budget style, trip length, party size). The frontend converts
    this to the traveller's local currency using their detected location."""
    payload = request.get_json(force=True) or {}
    answers = payload.get("answers", {})
    dest_id = payload.get("destination_id")

    model = get_model()
    dest_meta = model.dest_meta.get(dest_id, {}) if dest_id is not None else {}
    try:
        result = model.estimate_budget(answers, dest_meta, QUESTIONS_BY_KEY)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


if __name__ == "__main__":
    get_model()  # warm the model at startup, not on first request
    app.run(host="0.0.0.0", port=5000, debug=False)
