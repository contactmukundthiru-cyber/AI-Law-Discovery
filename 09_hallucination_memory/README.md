# Hallucination Is Compressed Memory

## Research Hypothesis

"Hallucinations" aren't errors—they're what happens when a model retrieves from its compressed representation of training data. Understanding hallucination IS understanding memory.

## Significance

- Reframes hallucination from bug to window into cognition
- Connects to human false memory research
- Provides ways to study memory without labeled retrieval
- Offers principled hallucination reduction strategies

## Methodology

### Phase 1: Hallucination Characterization
Systematically analyze hallucinations:
- Structural patterns in hallucinated content
- Relationship to training data
- Predictability of hallucination types

### Phase 2: Memory Retrieval Probing
Probe memory mechanisms:
- Activation patterns during hallucination
- Comparison to veridical retrieval
- Memory interference effects

### Phase 3: Controlled Hallucination
Test hallucination induction:
- Predictable hallucination triggers
- Memory manipulation experiments
- Hallucination-based training data archaeology

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
python dashboard/app.py  # http://localhost:5009
```

## License

MIT License
