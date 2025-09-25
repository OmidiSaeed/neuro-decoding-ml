# neuro-decoding-ml

This repository demonstrates how to load and structure data from 
Allen Brain Observatory Visual Behavior Project [https://allensdk.readthedocs.io/en/stable/_static/examples/nb/visual_behavior_ophys_dataset_manifest.html].

Then, it explores machine learning methods for decoding neural activity from 2P calcium imaging data.

_The goal is to create clean, reproducible pipelines from raw data -> preprocessing and feature extraction -> ML models_

Source of data:
- Allen Brain Observatory calcium imaging dataset
- Decoding visual stimuli (drifting grating) from neural activity 

# Environment Setup

We recommend using a dedicated conda environment with its own Python version to ensure compatibility with AllenSDK.

# create the conda env, activate it, register the kernel
conda env create -f environment.yml
conda activate allensdk-env
python -m ipykernel install --user --name=allensdk-env --display-name "Python (allensdk)"