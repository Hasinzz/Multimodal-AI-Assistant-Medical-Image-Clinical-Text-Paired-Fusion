# Dataset access and expected layout

Raw data are not distributed here. Obtain each dataset from its original source, accept its terms, cite it, and keep restricted material outside Git.

| Component | Dataset role | Reported local cohort |
|---|---|---|
| Model-1A | Four-class Brain Tumor MRI collection | 7,200 readable images |
| Model-1B | NIH ChestXray14 | 112,120 images; 30,805 patients |
| Model-2A | Prescription and medical-report resources | 241 processed records |
| Model-2B | MTSamples-derived transcriptions | 4,841 retained specialty records |
| Model-3 | Indiana University chest X-ray/report collection | 2,287 eligible paired studies |

The [final report](final-thesis-report.pdf) contains authoritative citations and access routes. The IU source publication describes approximately 3,996 reports and 8,121 images; the accessed derivative exposed 3,851 report rows and 7,466 projections before filtering. Do not conflate these representations.

Expected repository-relative roots include `data/`, `checkpoints/`, and `outputs/`. Exact manifests depend on the entry point; inspect `src/config.py`, `src/iu_paired/config.py`, and `src/iu_paired_improved/config.py` before running.

Never commit raw images, prescriptions, reports, transcriptions, case-level outputs, identifiers, or model weights. Preserve versions, hashes, exclusions, and seeds in a controlled local record.
