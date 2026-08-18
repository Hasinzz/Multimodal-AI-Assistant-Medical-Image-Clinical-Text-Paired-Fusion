# Verified thesis results

Evidence types are reported separately rather than reduced to one overall accuracy.

- **Model-1A:** grouped accuracy 0.9317 ± 0.0168; macro F1 0.9316 ± 0.0169; macro AUROC 0.9905 ± 0.0011; 7,200 OOF predictions.
- **Model-1B:** patient-wise macro AUROC 0.7746 ± 0.0044; micro AUROC 0.7932 ± 0.0052; tuned macro/micro F1 0.2229 ± 0.0031 / 0.2847 ± 0.0061; 112,120 OOF predictions.
- **Model-2A:** OCR/entity-field/structured-output completion 1.0000/0.8755/1.0000 across 241 records. These are coverage measures, not semantic accuracy.
- **Model-2B specialty:** accuracy 0.3209; macro F1 0.3468; weighted F1 0.2962 on 969 held-out records.
- **Model-2B weak entities:** token accuracy 0.9937; entity F1 0.0148. This does not support a reliable expert-NER claim.

## Model-3 locked test (n=352)

| Branch | Macro AUROC | Macro AUPRC | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|
| Image only | 0.7910 | 0.3064 | 0.2915 | 0.3173 |
| Indication only | 0.6282 | 0.1257 | 0.1442 | 0.1887 |
| Late fusion | 0.7954 | 0.3107 | 0.3071 | 0.3295 |

Fusion-minus-image paired differences were +0.0044 AUROC (95% CI −0.0042–0.0133), +0.0043 AUPRC (−0.0186–0.0227), and +0.0156 F1 (−0.0128–0.0444). All intervals cross zero.

Generated Findings achieved ROUGE-1/2/L of 0.4569/0.3692/0.4895. ROUGE measures lexical overlap, not clinical correctness or superiority over differently evaluated systems.
