import sys
# print('---sys path= ',sys.path)
import os
import click
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional, List
import time
import wandb
import random 
import numpy as np
import os.path as osp
from pathlib import Path 
from dotenv import load_dotenv  
import sys

# -------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as Func
from torchvision import transforms
from torch import Tensor, einsum
from pytorch_lightning import seed_everything
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
# from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
#-------------------------------------------
import tifffile as tiff
import matplotlib.pyplot as plt
import signal
from PIL import Image
from tqdm import tqdm
from operator import itemgetter, mul
from functools import partial
from wandb import Artifact
#--------------------------------------------

# from scripts.train_modules.z.boundaryloss import BoundaryLoss
from scripts.train_modules.train_helpers import *
from scripts.train_modules.train_classes import  UnetModel,   Segmentation_training_loop 
from scripts.train_modules.train_functions import handle_interrupt
from scripts.train_modules.train_functions import calculate_metrics
# log_metrics_to_wandb
# from scripts.train_modules.training_loops import Segmentation_training_loop

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Simulate one training step
model = UnetModel(encoder_name='resnet34', in_channels=1, classes=1, pretrained=False).to(pick_device())
dummy_input = torch.randn(2, 1, 256, 256).to(pick_device())
dummy_mask = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32).to(pick_device())  # Binary mask

output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should match dummy_mask shape
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(output, dummy_mask)
print(f"Loss: {loss.item()}")
