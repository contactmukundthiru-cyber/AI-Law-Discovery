"""
Flask dashboard for Maximum Intelligence experiments.
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-5012")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "project": "Maximum Intelligence",
        "version": "1.0.0",
    })


@app.route("/api/experiments")
def api_experiments():
    experiments = []
    if RESULTS_DIR.exists():
        for exp_file in RESULTS_DIR.glob("*.json"):
            experiments.append({"id": exp_file.stem, "path": str(exp_file)})
    return jsonify({"experiments": experiments})


@app.route("/api/run", methods=["POST"])
def api_run_experiment():
    config = request.json or {}
    return jsonify({"status": "started", "config": config})


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 5012))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
