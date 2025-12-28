"""
Flask dashboard for Physics Simulator experiments.
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
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "physics-sim-dev-key")
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
        "project": "Physics Simulators",
        "version": "1.0.0",
    })


@app.route("/api/experiments")
def api_experiments():
    experiments = []
    if RESULTS_DIR.exists():
        for exp_dir in RESULTS_DIR.glob("experiment_*"):
            if exp_dir.is_dir():
                experiments.append({
                    "id": exp_dir.name,
                    "path": str(exp_dir),
                })
    return jsonify({"experiments": experiments})


@app.route("/api/probing/results")
def api_probing_results():
    results_file = RESULTS_DIR / "probing_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No results found"})


@app.route("/api/run", methods=["POST"])
def api_run_experiment():
    config = request.json or {}
    return jsonify({"status": "started", "config": config})


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 5002))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
