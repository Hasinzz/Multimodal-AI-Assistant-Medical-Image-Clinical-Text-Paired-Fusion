# Architecture and methodology

The thesis evaluates three independently bounded systems. Model-1 and Model-2 outputs are not passed into Model-3.

## Model-1

```mermaid
flowchart LR
  subgraph MRI[Model-1A Brain MRI]
    M0[MRI image] --> M1[Resize and normalize] --> M2[Training-only augmentation] --> M3[DenseNet-121] --> M4[Four-class probabilities] --> M5[Similarity-grouped five-fold CV]
  end
  subgraph CXR[Model-1B Chest X-ray]
    X0[Chest X-ray] --> X1[Resize and normalize] --> X2[Training-only augmentation] --> X3[DenseNet-121] --> X4[Fourteen-label probabilities] --> X5[Inner-selected thresholds] --> X6[Patient-wise five-fold CV]
  end
```

Model-1A confines detected exact/perceptual-similarity components to folds. Model-1B uses patients as both outer- and inner-split units.

## Model-2

```mermaid
flowchart LR
  S0[Scanned document] --> S1[Image preparation] --> S2[OCR] --> S3[Cleaning] --> S4[Rule extraction] --> S5[Structured JSON]
  T0[Doctor-typed text] --> T1[Cleaning and vectorization] --> T2[Specialty classifier] --> T3[Specialty output]
  T1 --> T4[Separate weak-label entity module] --> T5[Weak-label entity output]
```

Processing completion is not expert semantic accuracy, and weak-label agreement is not clinician-verified entity recognition.

## Model-3 paired IU architecture

```mermaid
flowchart TB
  U[One eligible IU study] --> I[Study radiograph projection(s)]
  U --> T[Same-study Indication]
  I --> IE[Three-seed ImageNet DenseNet-121 ensemble] --> PI[p_image]
  T --> TF[Word + character TF-IDF] --> LR[Ten one-vs-rest logistic regressions] --> PT[p_text]
  PI --> F[p_fusion = 0.75 p_image + 0.25 p_text]
  PT --> F
  F --> TH[Development-selected thresholds] --> L[Ten predicted findings] --> G[Training-derived frozen templates] --> GF[Generated Findings]
  H[Hidden same-study Findings] --> R[ROUGE evaluation only]
  GF --> R
  Q[Findings and Impression never enter predictor branches] -. safeguard .-> H
```

The image seeds are 42, 123, and 2026. Reference Findings and Impression remain hidden until after prediction and generation.

## Validation and leakage controls

```mermaid
flowchart LR
  B0[7,200 Brain MRI images] --> B1[Similarity groups] --> B2[Five grouped outer folds] --> B3[Grouped inner validation]
  C0[112,120 X-rays; 30,805 patients] --> C1[Patient groups] --> C2[Five patient-wise outer folds] --> C3[Patient-wise inner validation]
  U0[2,287 eligible IU studies] --> U1[1,935 development] --> U3[Select model, fusion, thresholds] --> U4[Freeze]
  U0 --> U2[352 locked test]
  U4 --> U2 --> U5[One final evaluation]
```

IU separation is study-wise because a stable patient identifier was unavailable. Repeat patients cannot be excluded.

## Code provenance

- `src/iu_paired_improved/`: thesis-selected locked-test implementation.
- `src/iu_paired/`: earlier paired implementation and shared helpers.
- `src/iu_paired_rescue/`: development-only follow-up experiments; no new final-test claim.
- `src/model3/`: earlier general fusion/RAG prototype, not the IU locked-test architecture.
