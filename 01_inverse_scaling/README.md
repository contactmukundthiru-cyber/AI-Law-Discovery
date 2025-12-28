# Inverse Scaling Exists and Matters

## Research Hypothesis

Everyone assumes bigger = better. This project investigates whether some capabilities peak at intermediate scale and then **decline**—not plateau, but actually get worse.

## Why This Would Be Huge

- Would shatter the scaling paradigm
- Would mean there's an "optimal size" for certain tasks
- Would fundamentally change how we build and deploy models

## Where We Look

- Tasks requiring precise rule-following
- Simple arithmetic operations
- Consistent persona maintenance
- Faithful retrieval tasks
- Cases where bigger models "overthink" simple problems

## Methodology

### Phase 1: Task Design
We design tasks specifically likely to exhibit inverse scaling:
1. **Strict Rule Following**: Tasks with explicit, simple rules that must be followed exactly
2. **Simple Arithmetic**: Basic math that larger models may overcomplicate
3. **Pattern Completion**: Simple sequences that invite overthinking
4. **Literal Instruction Following**: Tasks requiring exact adherence to stated instructions

### Phase 2: Multi-Scale Evaluation
- Evaluate across multiple model scales (70M to 70B+ parameters)
- Use standardized prompting to control for prompt engineering effects
- Collect performance metrics with confidence intervals

### Phase 3: Analysis
- Identify U-shaped or inverted-U performance curves
- Characterize the "overthinking" phenomenon
- Map optimal scale per task type

## Project Structure

```
01_inverse_scaling/
├── config/                    # Experiment configurations
├── src/
│   ├── data/                  # Task datasets and loaders
│   ├── models/                # Model interface wrappers
│   ├── experiments/           # Experiment runners
│   ├── analysis/              # Statistical analysis
│   └── visualization/         # Plotting utilities
├── dashboard/                 # Web-based experiment control
├── scripts/                   # CLI entry points
├── tests/                     # Unit tests
└── results/                   # Output data and figures
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python scripts/run_experiments.py --config config/experiment_config.yaml

# Launch dashboard
python dashboard/app.py
```

## Dashboard

Access the experiment dashboard at `http://localhost:5001` after launching.

Features:
- Configure and launch experiment trials
- Monitor real-time progress
- Visualize scaling curves
- Export publication-ready figures
- Manage datasets and add new task types

## Citation

```bibtex
@article{inverse_scaling_2024,
  title={Inverse Scaling Exists and Matters: Evidence for Non-Monotonic
         Performance in Language Models},
  author={[Authors]},
  journal={Nature},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.
