from setuptools import setup, find_packages

setup(
    name="faster_rcnn_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'pytorch-lightning>=2.0.0',
        'efficientnet-pytorch>=0.7.1',
        'pycocotools>=2.0.6',
        'wandb>=0.13.0',
        'opencv-python>=4.5.0',
        'albumentations>=1.0.0',
    ],
)