import random
from typing import List, Tuple

import requests


def random_float(low, high):
    return random.random() * (high - low) + low


def get_elevation(points: List[Tuple[float, float]]) -> List[float]:
    """
    Get elevation for localization with given coordinates. Data are obtained from opentopodata API.

    Args:
        points (list): list of tuples with coordinates (lat, long)

    Return:
        url (list): list of elevations
    """

    if len(points) > 100:
        print('Too many locations, max number of location per request is 100. Empty list returned.')
        return []
    else:
        locations = '|'.join([f'{lat},{long}' for lat, long in points])
        query = f'https://api.opentopodata.org/v1/srtm30m?locations=' + locations
        result = requests.get(query)  # json object, various ways you can extract value
        return [single_result['elevation'] for single_result in result.json()['results']]
