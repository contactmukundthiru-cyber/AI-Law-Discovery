"""Flask dashboard for Bitter Lesson analysis."""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'bitter-lesson-secret')
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/results')
def get_results():
    return jsonify({
        "scale_tasks": ["language_modeling", "translation"],
        "architecture_tasks": ["physical_reasoning", "graph_algorithms"],
        "conclusion": "Hybrid approaches recommended"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5010))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
