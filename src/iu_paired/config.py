from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IU_ROOT = PROJECT_ROOT / "data" / "iu_xray"
REPORTS_CSV = IU_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV = IU_ROOT / "indiana_projections.csv"
IMAGES_DIR = IU_ROOT / "images" / "images_normalized"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iu_paired"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "iu_paired"

TARGET_LABELS = [
    "Cardiomegaly", "Pulmonary Atelectasis", "Calcified Granuloma", "Cicatrix",
    "Pleural Effusion", "Atherosclerosis", "Airspace Disease", "Scoliosis",
    "Granulomatous Disease", "Nodule",
]
SEED = 42
IMAGE_SIZE = 224

def assert_paths() -> None:
    for path in (REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Required IU path missing: {path}")
