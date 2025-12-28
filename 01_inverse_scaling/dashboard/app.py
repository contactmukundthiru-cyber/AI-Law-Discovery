"""
Flask dashboard for Inverse Scaling experiment management.

Provides web interface for:
- Configuring and launching experiments
- Monitoring real-time progress
- Visualizing results
- Managing datasets
- Exporting publication-ready figures
"""

import os
import sys
import json
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, generate_full_dataset
from src.models import ModelRegistry, get_registry
from src.experiments import ExperimentRunner, ExperimentConfig, TrialStatus
from src.analysis import ResultsAnalyzer
from src.visualization import InteractivePlotter, PublicationFigureGenerator

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "inverse-scaling-dev-key")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "results" / "data"
RESULTS_DIR = BASE_DIR / "results"
CONFIG_PATH = BASE_DIR / "config" / "experiment_config.yaml"

# Global instances
registry = None
runner = None
data_loader = None
analyzer = ResultsAnalyzer()
plotter = InteractivePlotter()
figure_gen = PublicationFigureGenerator()

# Active experiment tracking
active_experiment = None
experiment_lock = threading.Lock()


def init_app():
    """Initialize application components."""
    global registry, runner, data_loader

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Initialize model registry
    registry = get_registry()
    if CONFIG_PATH.exists():
        registry.register_from_config(CONFIG_PATH)

    # Initialize data loader
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.json")):
        data_loader = DataLoader(DATA_DIR)
    else:
        data_loader = None

    # Initialize experiment runner
    runner = ExperimentRunner(DATA_DIR, RESULTS_DIR, registry)
    runner.add_progress_callback(broadcast_progress)


def broadcast_progress(trial, progress, message):
    """Broadcast experiment progress via WebSocket."""
    socketio.emit("experiment_progress", {
        "trial_id": trial.trial_id,
        "progress": progress,
        "message": message,
        "status": trial.status.value,
        "current_model": trial.current_model,
        "current_task": trial.current_task,
    })


# ==================== Routes ====================

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Get system status."""
    return jsonify({
        "status": "ok",
        "data_loaded": data_loader is not None,
        "models_registered": len(registry.list_models()) if registry else 0,
        "active_experiment": active_experiment.trial_id if active_experiment else None,
    })


# Dataset Management

@app.route("/api/datasets")
def api_datasets():
    """List available datasets."""
    if not data_loader:
        return jsonify({"datasets": [], "message": "No data loaded"})

    tasks = data_loader.get_available_tasks()
    info = data_loader.get_dataset_info()

    return jsonify({
        "datasets": tasks,
        "info": info,
    })


@app.route("/api/datasets/generate", methods=["POST"])
def api_generate_dataset():
    """Generate new task datasets."""
    global data_loader

    params = request.json or {}
    samples_per_task = params.get("samples_per_task", 500)
    seed = params.get("seed", 42)

    try:
        generate_full_dataset(DATA_DIR, samples_per_task, seed)
        data_loader = DataLoader(DATA_DIR)

        return jsonify({
            "success": True,
            "message": f"Generated {samples_per_task} samples per task",
            "tasks": data_loader.get_available_tasks(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/datasets/<task_name>")
def api_dataset_details(task_name):
    """Get details for a specific dataset."""
    if not data_loader:
        return jsonify({"error": "No data loaded"}), 404

    try:
        dataset = data_loader.load_task(task_name)
        return jsonify({
            "name": task_name,
            "size": len(dataset),
            "subtasks": dataset.get_subtasks(),
            "checksum": dataset.checksum,
            "sample": [inst.to_dict() for inst in dataset.sample(5)],
        })
    except FileNotFoundError:
        return jsonify({"error": f"Task not found: {task_name}"}), 404


# Model Management

@app.route("/api/models")
def api_models():
    """List registered models."""
    if not registry:
        return jsonify({"models": []})

    return jsonify({
        "models": registry.list_models(),
        "enabled": registry.get_all_enabled(),
    })


@app.route("/api/models/<path:model_key>/toggle", methods=["POST"])
def api_toggle_model(model_key):
    """Enable or disable a model."""
    if not registry:
        return jsonify({"error": "Registry not initialized"}), 500

    action = request.json.get("action", "toggle")

    if action == "enable":
        registry.enable(model_key)
    elif action == "disable":
        registry.disable(model_key)
    else:
        # Toggle
        models = registry.list_models()
        for m in models:
            if m["key"] == model_key:
                if m["enabled"]:
                    registry.disable(model_key)
                else:
                    registry.enable(model_key)
                break

    return jsonify({"success": True})


# Experiment Management

@app.route("/api/experiments")
def api_experiments():
    """List all experiments."""
    if not runner:
        return jsonify({"experiments": []})

    trials = runner.list_trials()
    return jsonify({
        "experiments": [
            {
                "trial_id": t.trial_id,
                "name": t.name,
                "status": t.status.value,
                "progress": t.progress,
                "created_at": t.created_at.isoformat(),
                "results_count": len(t.results),
            }
            for t in trials
        ]
    })


@app.route("/api/experiments/<trial_id>")
def api_experiment_detail(trial_id):
    """Get experiment details."""
    if not runner:
        return jsonify({"error": "Runner not initialized"}), 500

    trial = runner.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    return jsonify(trial.to_dict())


@app.route("/api/experiments/run", methods=["POST"])
def api_run_experiment():
    """Start a new experiment."""
    global active_experiment

    if not runner or not data_loader:
        return jsonify({"error": "System not fully initialized"}), 500

    with experiment_lock:
        if active_experiment and active_experiment.status == TrialStatus.RUNNING:
            return jsonify({"error": "Experiment already running"}), 400

    params = request.json or {}

    # Build configuration
    config = ExperimentConfig(
        name=params.get("name", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        models=params.get("models", registry.get_all_enabled()),
        tasks=params.get("tasks", data_loader.get_available_tasks()),
        samples_per_task=params.get("samples_per_task", 100),
        num_runs=params.get("num_runs", 1),
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_tokens", 256),
        seed=params.get("seed", 42),
    )

    # Run in background thread
    def run_experiment():
        global active_experiment
        with experiment_lock:
            active_experiment = runner.run_experiment(config)

    thread = threading.Thread(target=run_experiment)
    thread.start()

    return jsonify({
        "success": True,
        "message": "Experiment started",
    })


@app.route("/api/experiments/<trial_id>/cancel", methods=["POST"])
def api_cancel_experiment(trial_id):
    """Cancel a running experiment."""
    global active_experiment

    if active_experiment and active_experiment.trial_id == trial_id:
        runner.cancel()
        return jsonify({"success": True, "message": "Cancellation requested"})

    return jsonify({"error": "Experiment not found or not running"}), 404


# Analysis and Visualization

@app.route("/api/experiments/<trial_id>/analysis")
def api_experiment_analysis(trial_id):
    """Get analysis for an experiment."""
    if not runner:
        return jsonify({"error": "Runner not initialized"}), 500

    trial = runner.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    analysis = analyzer.analyze_trial(trial)

    # Convert to JSON-serializable format
    def convert(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    return jsonify(convert(analysis))


@app.route("/api/experiments/<trial_id>/plots/<plot_type>")
def api_experiment_plot(trial_id, plot_type):
    """Get interactive plot for an experiment."""
    trial = runner.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    analysis = analyzer.analyze_trial(trial)

    if plot_type == "scaling_curves":
        # Get first task with inverse scaling as example
        for task_name, task_data in analysis.get("tasks", {}).items():
            if hasattr(task_data, "model_scales") and task_data.model_scales:
                plot_json = plotter.scaling_curve_interactive(
                    task_data.model_scales,
                    task_data.accuracies,
                    task_data.model_names,
                    title=f"Scaling: {task_name}",
                )
                return jsonify({"plot": json.loads(plot_json)})

    elif plot_type == "summary":
        plot_json = plotter.inverse_scaling_summary(analysis)
        return jsonify({"plot": json.loads(plot_json)})

    elif plot_type == "patterns":
        pattern_dist = analysis.get("overall", {}).get("pattern_distribution", {})
        plot_json = plotter.pattern_distribution_interactive(pattern_dist)
        return jsonify({"plot": json.loads(plot_json)})

    return jsonify({"error": f"Unknown plot type: {plot_type}"}), 400


@app.route("/api/experiments/<trial_id>/export", methods=["POST"])
def api_export_results(trial_id):
    """Export experiment results."""
    trial = runner.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    params = request.json or {}
    format_type = params.get("format", "json")

    analysis = analyzer.analyze_trial(trial)
    export_path = RESULTS_DIR / "exports" / f"{trial_id}_analysis.{format_type}"

    analyzer.export_results(analysis, export_path, format_type)

    return send_file(
        export_path,
        as_attachment=True,
        download_name=f"inverse_scaling_{trial_id}.{format_type}",
    )


@app.route("/api/experiments/<trial_id>/figure", methods=["POST"])
def api_generate_figure(trial_id):
    """Generate publication-ready figure."""
    trial = runner.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    analysis = analyzer.analyze_trial(trial)
    figure_path = RESULTS_DIR / "figures" / f"figure_{trial_id}"

    fig = figure_gen.create_main_figure(analysis, figure_path)
    saved = figure_gen.save_figure(fig, figure_path)

    return jsonify({
        "success": True,
        "files": [str(p) for p in saved],
    })


# WebSocket events

@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    emit("connected", {"status": "ok"})


@socketio.on("subscribe_experiment")
def handle_subscribe(data):
    """Subscribe to experiment updates."""
    trial_id = data.get("trial_id")
    # Add to room for targeted updates
    emit("subscribed", {"trial_id": trial_id})


# ==================== Main ====================

if __name__ == "__main__":
    init_app()
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
