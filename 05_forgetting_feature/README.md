# The Forgetting Is The Feature

## Research Hypothesis

The real intelligence in models might lie not in what they learn, but in what they systematically forget. The compression—the systematic discarding of "wrong" information—might be the actual intelligence.

## Significance

- Inverts the field's focus from learning to forgetting
- Connects to neuroscience research on sleep and consolidation
- Suggests optimizing forgetting, not just learning
- Provides new metrics for model quality

## Methodology

### Phase 1: Forgetting Characterization
Analyze what gets forgotten during training:
- Track information content over training
- Identify patterns in forgotten information
- Measure forgetting rates for different data types

### Phase 2: Predictive Models
Develop models to predict forgetting:
- Which information will be forgotten
- When forgetting occurs
- Relationship between forgetting and generalization

### Phase 3: Optimized Forgetting
Test whether better forgetting improves models:
- Forgetting schedules
- Targeted forgetting mechanisms
- Sleep-inspired consolidation phases

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
python dashboard/app.py  # http://localhost:5005
```

## License

MIT License
