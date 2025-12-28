"""Flask dashboard for Compression Consciousness analysis."""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'compression-consciousness-secret')
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/results')
def get_results():
    return jsonify({
        "compression_levels": [0.8, 0.5, 0.2, 0.1, 0.05],
        "self_reference_scores": [0.25, 0.42, 0.65, 0.78, 0.85],
        "threshold": 0.2,
        "conclusion": "Self-reference emerges at high compression"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5012))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
