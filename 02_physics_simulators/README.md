# Language Models as Implicit Physics Simulators

## Research Hypothesis

Language models may not simply learn "language patterns" but instead learn compressed simulations of the physical and social processes that generate text. They function as world-simulators that output text as a byproduct.

## Significance

- Reframes understanding of LLM capabilities fundamentally
- Explains emergent reasoning about novel physical situations
- Provides theoretical grounding for in-context learning
- Connects language modeling to simulation-based cognition theories

## Methodology

### Phase 1: Physical Variable Probing
Train linear probes on model activations to detect encoding of:
- Spatial relationships (distance, position, direction)
- Temporal relationships (before/after, duration, sequence)
- Causal relationships (cause/effect chains)
- Physical properties (mass, velocity, temperature)

### Phase 2: Physics Prediction Tasks
Evaluate model performance on physical reasoning tasks:
- Trajectory prediction
- Object permanence
- Collision outcomes
- Conservation laws

### Phase 3: Representation Analysis
Analyze internal representations for physics-like structure:
- Dimensionality of spatial encodings
- Consistency of physical variable encodings across contexts
- Transfer of physical reasoning across domains

## Project Structure

```
02_physics_simulators/
├── config/                    # Experiment configurations
├── src/
│   ├── data/                  # Physics task datasets
│   ├── models/                # Model interfaces
│   ├── probes/                # Linear probe implementations
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
pip install -r requirements.txt
python scripts/run_experiments.py --config config/experiment_config.yaml
python dashboard/app.py  # Launch dashboard at http://localhost:5002
```

## License

MIT License
