#!/bin/bash

# Create conda environment
conda create -n frcnn_env python=3.9 -y

# Activate environment
conda activate frcnn_env

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install pytorch-lightning
pip install wandb
pip install efficientnet-pytorch
pip install pycocotools
pip install opencv-python
pip install albumentations

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"