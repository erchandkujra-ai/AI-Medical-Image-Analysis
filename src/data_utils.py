
from pathlib import Path

# Project root: parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Create output folders automatically.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
