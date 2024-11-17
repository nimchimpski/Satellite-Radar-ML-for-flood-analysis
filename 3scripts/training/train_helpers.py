from torchvision.transforms import functional as Func
from torchvision import transforms
from torch.utils.data import Subset 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch import Tensor, einsum
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from torch.utils.data import DataLoader
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
from train_classes import FloodDataset
#------------------------------------------


def initialize_wandb(project, job_type, run_name):
    """
    Initializes WandB if not already initialized.
    
    Args:
    - project (str): The name of the WandB project.
    - job_type (str): The type of job (e.g., 'train', 'reproduce').
    - run_name (str): The name of the WandB run.
    
    Returns:
    - wandb.run: The active WandB run.
    """
    # Check if WandB is already initialized
    if wandb.run is None:
        # Initialize WandB
        run = wandb.init(
            project=project,
            job_type=job_type,
            name=run_name
        )
        return run
    else:
        # If already initialized, return the existing run
        return wandb.run
    

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]))

def convert_tensor_to_array(tensor):
    return tensor.cpu().numpy()

def create_numpy_array(img):
    return np.array(img)[...]

#NOT IN TRAIN
# D = Union[Image.Image, np.ndarray, Tensor]
# def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
#         return transforms.Compose([
#                 # lambda img: np.array(img)[...],
#                 partial(create_numpy_array),
#                 lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
#                 partial(class2one_hot, K=K),
#                 itemgetter(0)  # Then pop the element to go back to img shape
#         ])

#NOT IN TRAIN
# def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
#         return transforms.Compose([
#                 gt_transform(resolution, K),

#                 lambda t: t.cpu().numpy(),
#                 partial(one_hot2dist, resolution=resolution),
#                 lambda nd: torch.tensor(nd, dtype=torch.float32)
#         ])

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

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

def create_subset(file_list, event, stage,  subset_fraction , inputs, bs, num_workers, persistent_workers):
    dataset = FloodDataset(file_list, event, stage=stage, inputs=inputs)    
    subset_indices = random.sample(range(len(dataset)), int(subset_fraction * len(dataset)))
    subset = Subset(dataset, subset_indices)
    dl = DataLoader(subset, batch_size=bs, num_workers=num_workers, persistent_workers= persistent_workers,  shuffle = (stage == 'train'))
    return dl