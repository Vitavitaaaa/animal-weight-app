from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from pathlib import Path

import traceback

from backend.utils.inference import run_pipeline

app = Flask(__name__)
CORS(app)

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True)

# =========================
# HOME
# =========================

@app.route("/")
def home():
    return {
        "message": "Cow Weight Prediction API is running"
    }

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400

    filename = secure_filename(file.filename)

    save_path = UPLOAD_FOLDER / filename

    try:

        file.save(str(save_path))

        result = run_pipeline(str(save_path))

        if not isinstance(result, dict):
            return jsonify({
                "success": False,
                "error": "Invalid pipeline output"
            }), 500

        result["success"] = True

        return jsonify(result)

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if save_path.exists():
            save_path.unlink()

# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )