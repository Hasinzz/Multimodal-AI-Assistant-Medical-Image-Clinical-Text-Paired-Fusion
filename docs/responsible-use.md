# Responsible use and limitations

This software is for retrospective research and technical inspection only. It must not provide diagnoses, treatment decisions, triage, or patient-facing advice.

- No prospective, independent-hospital, clinician-rated, subgroup-safety, or patient-outcome evaluation was performed.
- Brain MRI similarity grouping cannot prove patient independence.
- ChestXray14 labels are automatically mined and imbalanced.
- Scanned documents lack expert-corrected transcripts and field references.
- Typed-text entity references are weak labels.
- IU separation is study-wise; unrecognized repeat patients cannot be excluded.
- ROUGE can reward template wording while missing clinically important errors.

Any future clinical-data work would require institutional approval, lawful access, encryption, least privilege, audit logs, retention controls, expert annotation, patient-wise separation, external validation, calibration, human oversight, and applicable regulatory/security review.
