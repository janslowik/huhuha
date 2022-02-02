#!/bin/bash

# opentopomap mlp 15 <- najgorszy
python huhuha/experiments/experiment_1.py run --epochs-list=1 --rep-num=1 --image-src=opentopomap --zoom=15 --model-names=MLP --pred=True

# najlpeszy z pojedynczych
# CNN_AUG_MLP 12 opentopomap
python huhuha/experiments/experiment_1.py run --epochs-list=10 --rep-num=5 --image-src=opentopomap --zoom=12 --model-names=CNN_AUG_MLP --pred=True


# # najlepszy z miksow
# # CNN_AUG_MLP 13,15 arcgis/opentopomap
python huhuha/experiments/experiment_1.py run --epochs-list=10 --rep-num=5 --image-src=arcgis --image-src=opentopomap  --zoom=13 --zoom=15 --model-names=CNN_AUG_MLP --pred=True

