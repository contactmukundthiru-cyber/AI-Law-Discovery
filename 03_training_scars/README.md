# Training Leaves Permanent Scars

## Research Hypothesis

The training process creates systematic damage—"scars"—that cannot be fixed by further training. Early training experiences permanently constrain what the model can become, analogous to critical periods in neurodevelopment.

## Significance

- Would necessitate radical rethinking of curriculum design
- Explains stubborn failure modes that persist despite fine-tuning
- Suggests models have something like "developmental trauma"
- Could predict long-term model behavior from early training

## Methodology

### Phase 1: Controlled Training Experiments
Train identical architectures with different data orderings:
- Randomized curriculum vs. structured curriculum
- Early exposure to different data types
- Varying proportions in early vs. late training

### Phase 2: Irreversibility Testing
Attempt to "fix" early training effects:
- Extended fine-tuning on counter-examples
- Targeted unlearning procedures
- Representation surgery

### Phase 3: Scar Characterization
Analyze the nature of training scars:
- Which capabilities are most affected
- Relationship between scar depth and training timing
- Transferability of scars across tasks

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
python dashboard/app.py  # http://localhost:5003
```

## License

MIT License
