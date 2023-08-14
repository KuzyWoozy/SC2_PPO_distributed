#!/bin/bash

wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.9.3.zip
unzip SC2.4.9.3.zip
chmod a+x SC2.4.9.3/StarCraftII/Versions/Base75025/SC2_x64
chmod a+x evaluate.py train.py

module load cray-python/3.9.13.1
python -m venv sc_virt
module unload cray-python/3.9.13.1
source sc_virt/bin/activate

python -m pip install --upgrade pip==22.0.4
python -m pip install setuptools
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
python -m pip install -r scripts/requirements.txt
