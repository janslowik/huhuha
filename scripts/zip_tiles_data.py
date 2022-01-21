import os
import zipfile
from pathlib import Path

from huhuha.settings import RAW_DATA_DIR

tile_dir = RAW_DATA_DIR / 'opentopomap_tiles'

zf = zipfile.ZipFile(RAW_DATA_DIR / 'opentopomap_tiles.zip', "w")
for dirname, subdirs, files in os.walk(tile_dir):
    dirname = Path(dirname)
    realtive_dirname = dirname.relative_to(RAW_DATA_DIR)
    zf.write(dirname, realtive_dirname)
    for filename in files:
        zf.write(dirname / filename, realtive_dirname / filename)
zf.close()
