"""
Flask dashboard for Emergence Mirage analysis.
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'emergence-mirage-secret')
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisState:
    """Global state for analysis tracking."""
    def __init__(self):
        self.results = {}
        self.is_running = False
        self.progress = 0
        self.current_model = None
        self.current_capability = None


state = AnalysisState()


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current analysis status."""
    return jsonify({
        'is_running': state.is_running,
        'progress': state.progress,
        'current_model': state.current_model,
        'current_capability': state.current_capability,
        'results_count': len(state.results)
    })


@app.route('/api/results')
def get_results():
    """Get analysis results."""
    return jsonify(state.results)


@app.route('/api/config')
def get_config():
    """Get experiment configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'experiment_config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return jsonify(config)
    return jsonify({})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit('status', {
        'is_running': state.is_running,
        'progress': state.progress
    })


@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Start emergence analysis."""
    if state.is_running:
        emit('error', {'message': 'Analysis already running'})
        return

    state.is_running = True
    state.progress = 0
    emit('analysis_started', {'status': 'started'})

    try:
        run_analysis(data)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        emit('error', {'message': str(e)})
    finally:
        state.is_running = False


def run_analysis(config):
    """Run the emergence analysis."""
    from src.data.capability_tasks import CapabilityTaskGenerator
    from src.experiments.emergence_detector import EmergenceDetector
    from src.experiments.metric_analyzer import MetricAnalyzer
    from src.analysis.curve_fitting import EmergenceCurveFitter

    capabilities = config.get('capabilities', ['arithmetic'])
    n_samples = config.get('n_samples', 50)

    generator = CapabilityTaskGenerator()
    detector = EmergenceDetector(generator, n_samples)
    analyzer = MetricAnalyzer()
    fitter = EmergenceCurveFitter()

    total_tasks = len(capabilities)

    for i, capability in enumerate(capabilities):
        state.current_capability = capability
        state.progress = int((i / total_tasks) * 100)

        socketio.emit('progress', {
            'progress': state.progress,
            'capability': capability,
            'status': f'Analyzing {capability}'
        })

        # Generate synthetic results for demonstration
        # In production, this would use actual models
        results = generate_synthetic_results(capability)

        # Analyze metrics
        comparisons = analyzer.analyze_metric_patterns(results)

        # Fit curves
        for metric, comp in comparisons.items():
            scales = [s[0] for s in comp.scores_by_scale]
            scores = [s[1] for s in comp.scores_by_scale]

            if len(scales) >= 3:
                import numpy as np
                fit_results = fitter.fit_all_models(
                    np.array(scales),
                    np.array(scores)
                )
                classification = fitter.classify_emergence(fit_results)

                state.results[f"{capability}_{metric}"] = {
                    'capability': capability,
                    'metric': metric,
                    'classification': classification.pattern_type,
                    'confidence': classification.confidence,
                    'best_model': classification.best_model,
                    'is_continuous': comp.is_continuous
                }

        socketio.emit('capability_complete', {
            'capability': capability,
            'results': state.results
        })

    state.progress = 100
    socketio.emit('analysis_complete', {'results': state.results})


def generate_synthetic_results(capability):
    """Generate synthetic results for demonstration."""
    from src.experiments.emergence_detector import EmergenceResult
    import numpy as np

    results = []
    scales = [1e8, 3e8, 1e9, 3e9, 1e10]

    for scale in scales:
        # Simulate different emergence patterns
        log_scale = np.log10(scale)

        # Binary shows step-like behavior
        binary = 1 / (1 + np.exp(-2 * (log_scale - 9.5)))

        # Continuous metrics show smoother growth
        partial = 0.2 + 0.6 * (log_scale - 8) / 2
        partial = min(max(partial, 0), 1)

        # Log prob is noisier
        log_prob = -5 + 0.5 * log_scale + np.random.normal(0, 0.1)

        results.append(EmergenceResult(
            capability=capability,
            model_name=f"model_{scale:.0e}",
            parameter_count=int(scale),
            binary_accuracy=binary + np.random.normal(0, 0.02),
            partial_credit_score=partial + np.random.normal(0, 0.02),
            log_probability_score=log_prob,
            rank_score=partial * 0.8 + np.random.normal(0, 0.02),
            entropy_score=0.3 + 0.4 * (log_scale - 8) / 2,
            sample_results=[],
            n_samples=50,
            mean_difficulty=0.5
        ))

    return results


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5006))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
