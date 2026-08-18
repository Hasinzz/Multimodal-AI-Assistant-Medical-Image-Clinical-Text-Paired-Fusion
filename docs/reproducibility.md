# Reproducibility

## Frozen design

- Brain MRI seed 42; similarity-grouped nested evaluation.
- ChestXray14 seed 42; patient-wise nested evaluation.
- IU split seed 2026; 1,935 development and 352 locked-test studies.
- IU image ensemble seeds 42, 123, and 2026.
- Indication model: word/character TF–IDF, one-vs-rest logistic regression, `C=0.25`.
- Late fusion: `0.75 × image + 0.25 × text`.
- Thresholds, aggregation, fusion weight, and templates selected on development data only.

Dataset-free checks:

```bash
pip install -r requirements-dev.txt
python -m compileall -q src tests
pytest -q
```

CLI inspection:

```bash
python -m src.run_case --help
python -m src.iu_paired.run --help
```

The `src/iu_paired_improved/phase*.py` modules are experiment programs, not polished CLIs. Read their configuration before execution; do not pass `--help` expecting a dry run.

The selected flow is: audit/fix splits; build caches; produce development OOF image/text predictions; compare development-only fusion/calibration strategies; freeze configuration and templates; train on all development data; generate locked predictions without Findings/Impression; reveal locked references once for metrics and ROUGE.

`python -m src.iu_paired_improved.final_pipeline` is the final artifact-dependent GPU stage. Any new execution is a reproduction, not another opportunity for selection on the reported test.
