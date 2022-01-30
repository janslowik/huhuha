# huhuha
nasza zima zła by JS&WK

## Instalacja
```
poetry install
poe install-torch
```

__UWAGA__ Na Windowsie odpowiednią wersję biblioteki `fiona` należy doinstalować ręcznie przed komendą `poetry install` zgodnie z instrukcją:
https://vincent.doba.fr/posts/20210407_install-fiona-on-windows/

Należy ściągnąć odpowiednie wersję biblioteki GDAL i Fiona, zwracając uwagę na ściągnięcie plików `.whl` odpowiednich dla architektury systemy oraz wersji pythona. Możliwe, ze będzie trzeba dodać zmienną środwiskową `GDAL_DATA` wskazującą na 
```
<adres środowiska python (venva)>\Lib\site-packages\osgeo\data\gdal
```

Należy również ręcznie pobrać odpowiedni wheel dla `rasterio` ze strony https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio oraz zainstalować w swoim środowisku pythonwym, np.
```
pip install rasterio-1.2.10-cp38-cp38-win_amd64.whl
```

Dokładna instrukcja: https://iotespresso.com/installing-rasterio-in-windows/

---
## Pobranie danych
```
dvc pull
```
Pobrane dane z _tile_'ami OpenTopoMap należy wypakowąć przy pomocy skryptu `scripts\unzip_tiles_data.py`

---
## Trenowanie

Podział projektu:
 - `huhuha/data` -- tutaj znajdują się klasy `DataModule` oraz `Dataset` dla danych.
 - `huhuha/learning` -- tutaj znajduje się klasa modelu klasyfikującego będąca modułem PyTorch Ligthning oraz
   funkcja przeprowadzająca trenowanie i testowanie
 - `huhuha/learning` -- tutaj dodajemy modele własne modele rozszerzające `torch.nn.Module`
 - `huhuha/experiments` -- skrypty wykonujące ekseprymenty

## experymenty

