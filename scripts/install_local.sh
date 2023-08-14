#!/bin/sh

wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.9.3.zip
unzip SC2.4.9.3.zip
chmod a+x SC2.4.9.3/StarCraftII/Versions/Base75025/SC2_x64
chmod a+x evaluate.py train.py

python3.9 -m venv sc_virt
source sc_virt/bin/activate

python3.9 -m pip install --upgrade pip==22.0.4
python3.9 -m pip install setuptools
python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
python3.9 -m pip install -r scripts/requirements.txt
