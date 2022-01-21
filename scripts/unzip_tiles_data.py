import zipfile

from huhuha.settings import RAW_DATA_DIR

zipfile_dir = RAW_DATA_DIR / 'opentopomap_tiles.zip'

with zipfile.ZipFile(zipfile_dir, 'r') as zip_object:
    zip_object.extractall(RAW_DATA_DIR)
