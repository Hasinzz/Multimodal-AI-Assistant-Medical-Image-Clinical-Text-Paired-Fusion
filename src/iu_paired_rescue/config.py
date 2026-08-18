from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
IMPROVED=ROOT/'outputs'/'iu_paired_improved'
V1=ROOT/'outputs'/'iu_paired'
OUT=ROOT/'outputs'/'iu_paired_rescue'
CKPT=ROOT/'checkpoints'/'iu_paired_rescue'
IMAGES=ROOT/'data'/'iu_xray'/'images'/'images_normalized'
PROJECTIONS=ROOT/'data'/'iu_xray'/'indiana_projections.csv'
LABELS=["Cardiomegaly","Pulmonary Atelectasis","Calcified Granuloma","Cicatrix","Pleural Effusion","Atherosclerosis","Airspace Disease","Scoliosis","Granulomatous Disease","Nodule"]
YCOL=[f'label_{i}' for i in range(10)]
PROB=[f'prob_{i}' for i in range(10)]
SEED=2026
MISSING_LABEL_SUBSTANTIAL_RATE=.02  # frozen before audit: >=2% of structured negatives explicitly positive in Findings

def setup():
    for p in [IMPROVED/'final_locked_split.csv',IMPROVED/'development_folds.csv',IMPROVED/'oof'/'model1_best_ensemble.csv',IMPROVED/'oof'/'model2_word_char.csv']:
        if not p.exists(): raise FileNotFoundError(p)
    OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
