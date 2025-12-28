# AI Discovery Laws: Investigating Fundamental Principles in Machine Learning

A comprehensive research framework investigating twelve radical hypotheses about the fundamental nature of artificial intelligence and machine learning systems.

## Overview

This repository contains twelve independent research projects, each exploring a potentially paradigm-shifting hypothesis about how neural networks learn, remember, and function.

## Research Projects

### 1. Inverse Scaling Exists and Matters
**Directory:** `01_inverse_scaling/`

Investigates whether some capabilities peak at intermediate scale and then decline. Challenges the "bigger is always better" assumption.

### 2. Language Models Are Secretly Physics Simulators
**Directory:** `02_physics_simulators/`

Tests whether LLMs learn compressed simulations of physical and social processes rather than just language patterns.

### 3. Training Leaves Permanent Scars
**Directory:** `03_training_scars/`

Examines whether early training creates systematic, irreversible constraints on model capabilities.

### 4. Features Have Ecology
**Directory:** `04_feature_ecology/`

Applies ecological frameworks to understand how features compete, evolve, and go extinct during training.

### 5. The Forgetting Is The Feature
**Directory:** `05_forgetting_feature/`

Studies systematic forgetting as a potentially crucial component of intelligence, not a bug.

### 6. Models Develop Like Embryos
**Directory:** `06_embryonic_development/`

Investigates universal developmental stages in learning systems, analogous to embryonic development.

### 7. Cognition Has Weather
**Directory:** `07_cognitive_weather/`

Explores chaotic dynamics in inference, including "storms" and "calm periods" in activation space.

### 8. Scale Catastrophe
**Directory:** `08_scale_catastrophe/`

Searches for critical scales beyond which architectures catastrophically fail.

### 9. Hallucination Is Compressed Memory
**Directory:** `09_hallucination_memory/`

Reframes hallucinations as windows into compressed memory retrieval mechanisms.

### 10. Models Have an Immune System
**Directory:** `10_model_immune_system/`

Investigates emergent immune-like defenses against adversarial inputs.

### 11. Dreaming is Necessary
**Directory:** `11_dreaming_necessary/`

Tests whether offline "dreaming" phases are necessary for certain capabilities.

### 12. There's a Maximum Intelligence
**Directory:** `12_maximum_intelligence/`

Explores fundamental limits on intelligence, independent of compute constraints.

## Project Structure

Each project is completely independent with its own:
- Configuration (`config/`)
- Source code (`src/`)
- Dashboard (`dashboard/`)
- Scripts (`scripts/`)
- Tests (`tests/`)
- Results (`results/`)

```
project_name/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   ├── analysis/
│   └── visualization/
├── dashboard/
│   ├── app.py
│   └── templates/
├── scripts/
│   └── run_experiments.py
├── tests/
└── results/
```

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/contactmukundthiru-cyber/AI-Law-Discovery.git
cd AI-Law-Discovery

# Install dependencies for a specific project
cd 01_inverse_scaling
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run experiments
python scripts/run_experiments.py --config config/experiment_config.yaml

# Launch dashboard
python dashboard/app.py
```

### API Keys (if using external models)

Set environment variables for API access:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export HUGGINGFACE_API_KEY="your-key"
```

## Dashboard Ports

Each project has its own dashboard running on a unique port:

| Project | Port |
|---------|------|
| Inverse Scaling | 5001 |
| Physics Simulators | 5002 |
| Training Scars | 5003 |
| Feature Ecology | 5004 |
| Forgetting Feature | 5005 |
| Embryonic Development | 5006 |
| Cognitive Weather | 5007 |
| Scale Catastrophe | 5008 |
| Hallucination Memory | 5009 |
| Model Immune System | 5010 |
| Dreaming Necessary | 5011 |
| Maximum Intelligence | 5012 |

## Contributing

Contributions are welcome. Please read the contributing guidelines before submitting pull requests.

## License

MIT License - See individual project LICENSE files for details.

## Author

Mukund Thiru (contactmukundthiru@gmail.com)

## Acknowledgments

This research framework was developed to systematically investigate fundamental questions about machine learning systems that could reshape our understanding of artificial intelligence.
