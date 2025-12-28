/**
 * Inverse Scaling Research Dashboard
 * Client-side JavaScript
 */

// Global state
let socket = null;
let currentExperiment = null;
let models = [];
let datasets = [];
let experiments = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initSocket();
    refreshStatus();
    loadModels();
    loadDatasets();
    loadExperiments();

    // Set up event listeners
    document.getElementById('results-experiment-select').addEventListener('change', onResultsExperimentChange);
    document.getElementById('analysis-experiment-select').addEventListener('change', onAnalysisExperimentChange);
});

// ==================== Socket.IO ====================

function initSocket() {
    socket = io();

    socket.on('connect', function() {
        updateStatusIndicator('Connected', 'success');
    });

    socket.on('disconnect', function() {
        updateStatusIndicator('Disconnected', 'danger');
    });

    socket.on('connected', function(data) {
        console.log('Socket connected:', data);
    });

    socket.on('experiment_progress', function(data) {
        updateExperimentProgress(data);
    });
}

function updateStatusIndicator(text, status) {
    const indicator = document.getElementById('status-indicator');
    indicator.innerHTML = `<span class="badge bg-${status}">${text}</span>`;
}

// ==================== API Calls ====================

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    const response = await fetch(`/api/${endpoint}`, options);
    return response.json();
}

async function refreshStatus() {
    const status = await apiCall('status');
    console.log('System status:', status);

    if (status.active_experiment) {
        currentExperiment = status.active_experiment;
        document.getElementById('progress-card').style.display = 'block';
    }
}

// ==================== Models ====================

async function loadModels() {
    const data = await apiCall('models');
    models = data.models || [];
    renderModelsList();
}

function renderModelsList() {
    const container = document.getElementById('models-list');

    if (models.length === 0) {
        container.innerHTML = '<p class="text-muted">No models registered</p>';
        return;
    }

    container.innerHTML = models.map(model => `
        <div class="model-item ${model.enabled ? '' : 'disabled'}" onclick="toggleModel('${model.key}')">
            <span>
                <i class="bi bi-${model.enabled ? 'check-circle-fill text-success' : 'circle text-muted'}"></i>
                ${model.name.split('/').pop().substring(0, 20)}
            </span>
            <span class="badge model-badge bg-secondary">${model.estimated_params}</span>
        </div>
    `).join('');
}

async function toggleModel(modelKey) {
    await apiCall(`models/${encodeURIComponent(modelKey)}/toggle`, 'POST', { action: 'toggle' });
    loadModels();
}

// ==================== Datasets ====================

async function loadDatasets() {
    const data = await apiCall('datasets');
    datasets = data.datasets || [];
    renderDatasetsList();
}

function renderDatasetsList() {
    const container = document.getElementById('datasets-list');

    if (datasets.length === 0) {
        container.innerHTML = '<p class="text-muted">No datasets. Click "Generate Dataset" to create.</p>';
        return;
    }

    container.innerHTML = datasets.map(task => `
        <div class="dataset-item" onclick="viewDataset('${task}')">
            <span><i class="bi bi-file-earmark-text"></i> ${task}</span>
        </div>
    `).join('');
}

async function generateDataset() {
    const samplesPerTask = prompt('Samples per task:', '500');
    if (!samplesPerTask) return;

    const data = await apiCall('datasets/generate', 'POST', {
        samples_per_task: parseInt(samplesPerTask),
        seed: 42
    });

    if (data.success) {
        alert(`Dataset generated with ${data.tasks.length} tasks`);
        loadDatasets();
    } else {
        alert('Error: ' + data.error);
    }
}

async function viewDataset(taskName) {
    const data = await apiCall(`datasets/${taskName}`);
    console.log('Dataset details:', data);
    alert(`Task: ${taskName}\nSize: ${data.size} samples\nSubtasks: ${data.subtasks.join(', ')}`);
}

// ==================== Experiments ====================

async function loadExperiments() {
    const data = await apiCall('experiments');
    experiments = data.experiments || [];
    renderExperimentsTable();
    populateExperimentSelects();
}

function renderExperimentsTable() {
    const tbody = document.querySelector('#experiments-table tbody');

    if (experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No experiments yet</td></tr>';
        return;
    }

    tbody.innerHTML = experiments.map(exp => `
        <tr>
            <td><code>${exp.trial_id}</code></td>
            <td>${exp.name}</td>
            <td><span class="badge status-${exp.status}">${exp.status}</span></td>
            <td>
                <div class="progress" style="height: 20px; width: 100px;">
                    <div class="progress-bar" style="width: ${exp.progress * 100}%">
                        ${Math.round(exp.progress * 100)}%
                    </div>
                </div>
            </td>
            <td>${new Date(exp.created_at).toLocaleString()}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary btn-action" onclick="viewExperiment('${exp.trial_id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-success btn-action" onclick="analyzeExperiment('${exp.trial_id}')">
                    <i class="bi bi-bar-chart"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function populateExperimentSelects() {
    const completedExps = experiments.filter(e => e.status === 'completed');
    const optionsHtml = completedExps.map(exp =>
        `<option value="${exp.trial_id}">${exp.name} (${exp.trial_id})</option>`
    ).join('');

    document.getElementById('results-experiment-select').innerHTML =
        '<option value="">Choose an experiment...</option>' + optionsHtml;
    document.getElementById('analysis-experiment-select').innerHTML =
        '<option value="">Choose an experiment...</option>' + optionsHtml;
}

function showNewExperimentModal() {
    // Populate models
    const enabledModels = models.filter(m => m.enabled);
    document.getElementById('exp-models-list').innerHTML = enabledModels.map(m =>
        `<div class="form-check">
            <input class="form-check-input" type="checkbox" value="${m.key}" id="model-${m.key}" checked>
            <label class="form-check-label" for="model-${m.key}">${m.name}</label>
        </div>`
    ).join('');

    // Populate tasks
    document.getElementById('exp-tasks-list').innerHTML = datasets.map(task =>
        `<div class="form-check">
            <input class="form-check-input" type="checkbox" value="${task}" id="task-${task}" checked>
            <label class="form-check-label" for="task-${task}">${task}</label>
        </div>`
    ).join('');

    new bootstrap.Modal(document.getElementById('newExperimentModal')).show();
}

async function startExperiment() {
    // Gather form data
    const name = document.getElementById('exp-name').value || `Experiment ${new Date().toISOString().slice(0, 16)}`;
    const samples = parseInt(document.getElementById('exp-samples').value) || 100;
    const runs = parseInt(document.getElementById('exp-runs').value) || 1;
    const temp = parseFloat(document.getElementById('exp-temp').value) || 0;
    const tokens = parseInt(document.getElementById('exp-tokens').value) || 256;

    // Get selected models
    const selectedModels = Array.from(
        document.querySelectorAll('#exp-models-list input:checked')
    ).map(el => el.value);

    // Get selected tasks
    const selectedTasks = Array.from(
        document.querySelectorAll('#exp-tasks-list input:checked')
    ).map(el => el.value);

    if (selectedModels.length === 0 || selectedTasks.length === 0) {
        alert('Please select at least one model and one task');
        return;
    }

    // Start experiment
    const result = await apiCall('experiments/run', 'POST', {
        name: name,
        models: selectedModels,
        tasks: selectedTasks,
        samples_per_task: samples,
        num_runs: runs,
        temperature: temp,
        max_tokens: tokens
    });

    if (result.success) {
        bootstrap.Modal.getInstance(document.getElementById('newExperimentModal')).hide();
        document.getElementById('progress-card').style.display = 'block';
        loadExperiments();
    } else {
        alert('Error: ' + result.error);
    }
}

function updateExperimentProgress(data) {
    const progressBar = document.getElementById('progress-bar');
    const progressPercent = Math.round(data.progress * 100);

    progressBar.style.width = progressPercent + '%';
    progressBar.textContent = progressPercent + '%';

    document.getElementById('progress-model').textContent = `Model: ${data.current_model || '--'}`;
    document.getElementById('progress-task').textContent = `Task: ${data.current_task || '--'}`;

    if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        document.getElementById('progress-card').style.display = 'none';
        loadExperiments();
    }
}

async function cancelExperiment() {
    if (!currentExperiment) return;

    if (confirm('Are you sure you want to cancel the experiment?')) {
        await apiCall(`experiments/${currentExperiment}/cancel`, 'POST');
    }
}

async function viewExperiment(trialId) {
    const data = await apiCall(`experiments/${trialId}`);
    console.log('Experiment details:', data);

    // Switch to results tab and select this experiment
    document.querySelector('[data-bs-target="#results-tab"]').click();
    document.getElementById('results-experiment-select').value = trialId;
    onResultsExperimentChange();
}

async function analyzeExperiment(trialId) {
    // Switch to analysis tab and select this experiment
    document.querySelector('[data-bs-target="#analysis-tab"]').click();
    document.getElementById('analysis-experiment-select').value = trialId;
    onAnalysisExperimentChange();
}

// ==================== Results & Analysis ====================

async function onResultsExperimentChange() {
    const trialId = document.getElementById('results-experiment-select').value;
    if (!trialId) return;

    const data = await apiCall(`experiments/${trialId}/plots/summary`);
    if (data.plot) {
        Plotly.newPlot('results-plot', data.plot.data, data.plot.layout);
    }
}

async function onAnalysisExperimentChange() {
    const trialId = document.getElementById('analysis-experiment-select').value;
    if (!trialId) return;

    // Load analysis
    const analysis = await apiCall(`experiments/${trialId}/analysis`);
    renderAnalysisSummary(analysis);

    // Load plot
    const plotData = await apiCall(`experiments/${trialId}/plots/summary`);
    if (plotData.plot) {
        Plotly.newPlot('analysis-plot', plotData.plot.data, plotData.plot.layout);
    }
}

function renderAnalysisSummary(analysis) {
    const container = document.getElementById('analysis-summary');
    const overall = analysis.overall || {};
    const inverseTasks = analysis.inverse_scaling_tasks || [];

    container.innerHTML = `
        <div class="metric">
            <span>Total Tasks</span>
            <span class="metric-value">${overall.total_tasks || 0}</span>
        </div>
        <div class="metric">
            <span>Models Evaluated</span>
            <span class="metric-value">${overall.models_evaluated || 0}</span>
        </div>
        <div class="metric">
            <span>Inverse Scaling Detected</span>
            <span class="metric-value ${inverseTasks.length > 0 ? 'inverse-scaling-detected' : ''}">${overall.inverse_scaling_detected || 0}</span>
        </div>
        <div class="metric">
            <span>Proportion</span>
            <span class="metric-value">${((overall.inverse_scaling_proportion || 0) * 100).toFixed(1)}%</span>
        </div>
        ${inverseTasks.length > 0 ? `
            <hr>
            <h6 class="inverse-scaling-detected">Tasks with Inverse Scaling:</h6>
            <ul class="small">
                ${inverseTasks.map(t => `<li>${t}</li>`).join('')}
            </ul>
        ` : ''}
    `;
}

async function exportResults() {
    const trialId = document.getElementById('analysis-experiment-select').value;
    if (!trialId) {
        alert('Please select an experiment first');
        return;
    }

    const format = prompt('Export format (json or csv):', 'json');
    if (!format) return;

    window.location.href = `/api/experiments/${trialId}/export?format=${format}`;
}

async function generateFigure() {
    const trialId = document.getElementById('analysis-experiment-select').value;
    if (!trialId) {
        alert('Please select an experiment first');
        return;
    }

    const result = await apiCall(`experiments/${trialId}/figure`, 'POST');
    if (result.success) {
        alert('Figures generated:\n' + result.files.join('\n'));
    } else {
        alert('Error generating figures');
    }
}
