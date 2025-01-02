from torchvision.transforms import functional as Func
from torchvision import transforms
from torch.utils.data import Subset 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch import Tensor, einsum
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
import torch
import torch.nn as nn
#------------------------------------------
from functools import partial
from operator import itemgetter, mul
from typing import Tuple, Callable, List, TypeVar, Any
import wandb
import numpy as np
import random
#--------------------------
# from scripts.train_modules.train_classes import FloodDataset
#------------------------------------------



def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]))

def convert_tensor_to_array(tensor):
    return tensor.cpu().numpy()

def create_numpy_array(img):
    return np.array(img)[...]


# METRICS

def acc_background(input, target, threshold=0.5):
    'define pixel-level accuracy just for background'
    mask = target != 1
    # return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    return ((input>threshold)[mask]==target[mask]).float().mean()

def acc_flood(input, target, threshold=0.5):
    'define pixel-level accuracy just for flood'
    mask = target != 0
    # return (input.argmax(dim=1)[mask]==target[mask]).float().mean()    THIS IS WRONG!!! ITS MULTI-CLASS
    return ((input>threshold)[mask]==target[mask]).float().mean()

def nsd(input, target, threshold=1):
    surface_distances = compute_surface_distances(target, input, [1,1])
    nsd = compute_surface_dice_at_tolerance(surface_distances, threshold)
    return nsd

#########


def reassemble_tiles(tiles, coords, image_shape, tile_size):
    """
    Combine predicted tiles into the original image shape.
    """
    h, w = image_shape
    prediction_image = np.zeros((h, w), dtype=np.uint8)

    for (x, y), tile in zip(coords, tiles):
        prediction_image[y:y+tile_size, x:x+tile_size] = tile

    return prediction_image

def is_sweep_run():
    return wandb.run.sweep_id is not None