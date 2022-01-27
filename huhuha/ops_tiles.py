import math
import os
from itertools import cycle
from typing import Tuple

import numpy as np
import requests
from PIL import Image

from huhuha.settings import DATA_DIR

LAT_BOUND = 85.0511
LON_BOUND = 180

server_iterator = cycle(["a", "b", "c"])

tiles_datadir = DATA_DIR / "opentopomap_tiles"
os.makedirs(tiles_datadir, exist_ok=True)


def get_otm_tile(
    lat: float,
    lon: float,
    zoom: int = 16,
    rm_png_file: bool = False,
    force_download: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Get tile of OpenTopoMap based on geographical coordinates

    Args:
        lat (float): latitude
        lon (float): longitude
        zoom (int): zoom of map, minimal value is 1, maximal is 17
        rm_png_file (bool): boolean indicating if remove saved png file
        force_download (bool): force download even if a tile saved on drive

    Return:
        url (str): link to tile of map
    """
    url = get_otm_tile_url(lat, lon, zoom)
    url_split = url.split("/")

    file_name = f"openstreat_map_z_{url_split[3]}_x_{url_split[4]}_y_{url_split[5]}"
    filepath = tiles_datadir / f"zoom_{zoom}" / file_name
    if not filepath.is_file() or force_download:
        download_and_save_image(url, filepath)

    image = Image.open(filepath).convert("RGB")

    if rm_png_file:
        os.remove(filepath)

    return np.asarray(image), file_name


def get_otm_tile_url(lat: float, lon: float, zoom: int = 16) -> str:
    """
    Get link to OpenTopoMap tile based on geographical coordinates

    Args:
        lat (float): latitude
        lon (float): longitude
        zoom (int): zoom of map, minimal value is 1, maximal is 17

    Return:
        url (str): link to tile of map
    """
    if zoom > 17:
        raise ValueError("Zoom value cannot exceed 17!")

    x, y = deg2num(lat, lon, zoom)
    url = (
        f"https://{server_iterator.__next__()}.tile.opentopomap.org/{zoom}/{x}/{y}.png"
    )
    return url


def download_and_save_image(url: str, savedir: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        with open(savedir, "wb") as file:
            file.write(response.content)
    else:
        print("Error", response.status_code)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """
    Change cooridnates to x and y numbers of tile
    # https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """
    if abs(lat_deg) > LAT_BOUND or abs(lon_deg) > LON_BOUND:
        raise ValueError(
            f"Coordinates out of bounds! Latitude has to be in"
            + f"[-{LAT_BOUND},{LAT_BOUND}] and longitude in [-{LON_BOUND},{LON_BOUND}]."
        )

    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile
