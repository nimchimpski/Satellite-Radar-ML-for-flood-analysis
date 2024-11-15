#!/bin/bash

# Name of the new conda environment
ENV_NAME="floodai_env"

# Create a new conda environment with Python 3.8
conda create -n $ENV_NAME python=3.8 -y

# Activate the new environment
source activate $ENV_NAME

# Install essential packages with conda
pip install -y -n $ENV_NAME cudatoolkit=11.8 -c nvidia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda install -y -n $ENV_NAME numpy scipy pandas matplotlib seaborn
conda install -y -n $ENV_NAME pillow opencv pyyaml jupyterlab
conda install -y -n $ENV_NAME pip setuptools



# Install additional Python packages using pip
pip install wandb dvc

# Optional: Load additional packages from YAML
# Uncomment the line below if you want to load from a YAML file:
# conda env update -n $ENV_NAME --file /path/to/your_environment.yaml --prune

echo "Conda environment '$ENV_NAME' created successfully!"


# DO THIS FIRST: 
# chmod +x torch_env.sh
# RUN WITH:
# ./setup_floodai_env.sh

