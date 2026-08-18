# Contributing

Contributions must preserve the thesis evidence boundaries and must not add patient data, raw datasets, checkpoints, case-level outputs, credentials, or restricted artifacts.

1. Create a focused branch.
2. Keep Model-1, Model-2, and Model-3 claims separate.
3. Add dataset-free tests for behavioral changes.
4. Run `python -m compileall -q src tests` and `pytest -q`.
5. Document split units, selection data, seeds, and reference quality for experiments.
6. Never tune on locked-test labels or hidden Findings/Impression.

No reuse license has currently been selected. Discuss licensing with the authors before contributing code intended for redistribution.
