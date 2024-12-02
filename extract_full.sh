#!/bin/bash
# extract_full.sh - Full dataset extraction script

echo "BEVFusion Full Dataset Extraction"
echo "================================"

cd /mmdet3d/data/nu

sudo apt-get install pigz  # On Debian/Ubuntu
sudo apt install zip unzip
for file in *.zip; do unzip "$file" && rm "$file"; done; for file in *.tar; do tar --use-compress-program=pigz --blocking-factor=512 -xvf "$file" && rm "$file"; done; mv expansion basemap prediction maps/


# Download and setup info files
rm -f nuscenes_infos_train.pkl nuscenes_infos_val.pkl
wget https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl
wget https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl

echo "Full dataset extraction completed!"
