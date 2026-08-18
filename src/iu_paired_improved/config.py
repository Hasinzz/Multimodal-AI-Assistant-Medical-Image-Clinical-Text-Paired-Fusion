from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_OUT = ROOT / "outputs" / "iu_paired"
V1_CKPT = ROOT / "checkpoints" / "iu_paired"
OUT = ROOT / "outputs" / "iu_paired_improved"
CKPT = ROOT / "checkpoints" / "iu_paired_improved"
IMAGES = ROOT / "data" / "iu_xray" / "images" / "images_normalized"
PROJECTIONS = ROOT / "data" / "iu_xray" / "indiana_projections.csv"
LABELS = ["Cardiomegaly", "Pulmonary Atelectasis", "Calcified Granuloma", "Cicatrix",
          "Pleural Effusion", "Atherosclerosis", "Airspace Disease", "Scoliosis",
          "Granulomatous Disease", "Nodule"]
SEED = 2026

def setup():
    for p in (V1_OUT / "cohort.csv", V1_OUT / "splits.csv", V1_CKPT / "model1" / "densenet121_best.pt", IMAGES, PROJECTIONS):
        if not p.exists(): raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
