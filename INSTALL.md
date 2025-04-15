# Installation

This codebase is tested on Ubuntu 22.04.4 LTS  with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).

```
# Create a conda environment
conda create -y -n TLAC python=3.10

# Activate the environment
conda activate TLAC

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

* Install dassl library.
```
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone TLAC code repository and install requirements

```
# Clone MaPLe code base
git clone https://github.com/ans92/TLAC.git

cd TLAC/

# Install requirements
pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```
