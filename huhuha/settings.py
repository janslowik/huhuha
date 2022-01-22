from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
RAW_DATA_DIR = PROJECT_DIR.resolve() / 'raw_data'
DATA_DIR = PROJECT_DIR.resolve() / 'data'
STORAGE_DIR = PROJECT_DIR / "storage"
LOGS_DIR = STORAGE_DIR / "logs"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"
