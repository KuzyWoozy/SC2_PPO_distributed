#!/bin/sh
	 
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.9.3.zip
unzip SC2.4.9.3.zip
chmod a+x SC2.4.9.3/StarCraftII/Versions/Base75025/SC2_x64

python -m pip install --upgrade pip==22.0.4
python -m pip install setuptools
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
python -m pip install -r data/requirements.txt
