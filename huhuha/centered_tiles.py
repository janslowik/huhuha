import itertools
import math
import os
from itertools import cycle
from typing import Tuple
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from huhuha.settings import DATA_DIR

LAT_BOUND = 85.0511
LON_BOUND = 180

server_iterator = cycle(["a", "b", "c"])

tiles_datadir = DATA_DIR / "opentopomap_tiles"
os.makedirs(tiles_datadir, exist_ok=True)


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

    _x = (lon_deg + 180.0) / 360.0 * n
    _y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    xtile = int(_x)
    ytile = int(_y)

    x = int(256 * (_x - xtile))
    y = int(256 * (_y - ytile))

    return xtile, ytile, x, y


def download_and_save_image(url: str, savedir: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        with open(savedir, "wb") as file:
            file.write(response.content)
    else:
        print("Error", response.status_code)


def get_otm_tile_url_by_xyz(
    x: int,
    y: int,
    zoom: int = 16,
    source: str = "opentopomap",
):
    if source == "opentopomap":
        return f"https://{server_iterator.__next__()}.tile.opentopomap.org/{zoom}/{x}/{y}.png"

    if source == "arcgis":
        return f"http://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{zoom}/{y}/{x}"

    raise Exception("no such source")


def get_otm_center_tile(
    lat: float,
    lon: float,
    zoom: int = 16,
    source: str = "opentopomap",
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
    # url = get_otm_tile_url(lat, lon, zoom)
    xt, yt, x, y = deg2num(lat, lon, zoom)

    # coordinates for big tile
    x = 256 + x - 128
    y = 256 + y - 128

    tiles_xz = itertools.product([xt - 1, xt, xt + 1], [yt - 1, yt, yt + 1])

    folder_path = f"{tiles_datadir}/{source}/{zoom}"

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    tiles = []

    for _x, _y in tiles_xz:

        url = get_otm_tile_url_by_xyz(_x, _y, zoom, source)
        file_name = f"{_x}_{_y}.png"

        file_path = os.path.join(folder_path, file_name)

        # check file exists
        if not os.path.isfile(file_path):
            download_and_save_image(url, file_path)

        _image = Image.open(file_path).convert("RGB")
        _image = np.asanyarray(_image)

        tiles.append(_image)

    big = np.concatenate(
        [
            np.concatenate(tiles[:3], axis=0),
            np.concatenate(tiles[3:6], axis=0),
            np.concatenate(tiles[6:], axis=0),
        ],
        axis=1,
    )

    centered = big[y : y + 256, x : x + 256]

    return centered
