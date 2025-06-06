import sys
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
# import handle interupt
import signal

# .............................................................
import torch
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
from pytorch_lightning.tuner.tuning import Tuner
# .............................................................
import tifffile as tiff
import matplotlib.pyplot as plt
import signal
from PIL import Image
from tqdm import tqdm
from operator import itemgetter, mul
from functools import partial
from wandb import Artifact
from datetime import datetime
# .............................................................

from scripts.process_modules.process_helpers import handle_interrupt
from scripts.train_modules.train_helpers import is_sweep_run
from scripts.train_modules.train_classes import  UnetModel,   Segmentation_training_loop 
from scripts.train_modules.train_functions import  loss_chooser, wandb_initialization, job_type_selector, create_subset

#.............................................................
# PATHS DEFINITIONS ANd CONSTANTS
repo_root = Path(__file__).resolve().parents[2]
print(f"repo root: {repo_root}")

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def handle_interrupt(signum, frame):
    print("\n---Custom signal handler: SIGINT received. Exiting.")
    sys.exit(0)
# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

@click.command()
@click.option('--train', is_flag=True,  help="Train the model")
@click.option('--test', is_flag=True, help="Test the model")

def main(train, test):
    """
    CONDA ENVIRONMENT = 'floodenv2'
    """
    device = pick_device()                       # used everywhere below
    print(f"⇢ Using device: {device}")

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print("Warning: .env not found; using shell environment")


    if test and train:
        raise ValueError("You can only specify one of --train or --test.")
    train = True

    if  test:
        train = False
        print('>>> ARE YOU TESTING THE CORRECT CKPT? <<<')
    print(f"train={train}, test={test}")

    job_type = "train" if train else "test"

    # Basic Setup
    start = time.time()
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    signal.signal(signal.SIGINT, handle_interrupt)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    ###########################################################
    # Paths and Parameters
    dataset_path = repo_root / "data" / "4final" / "train_INPUT"
    if test:
        dataset_path = repo_root / "data" / "4final" / "test_INPUT"

    test_ckpt_path = repo_root / "checkpoints" / "ckpt_INPUT"
    save_path = repo_root / "results"
    project = "TSX"
    subset_fraction = 1
    bs = 8
    max_epoch =100
    early_stop = False
    patience=10
    num_workers = 8
    WBOFFLINE = False
    LOGSTEPS = 50
    PRETRAINED = True
    inputs = ['hh', 'mask']
    in_channels = 1
    DEVRUN = 0
    user_loss = 'bce_dice' #'smp_bce' # 'bce_dice' #'focal' # 'bce_dice' # focal'
    focal_alpha = 0.8
    focal_gamma = 8
    bce_weight = 0.35 # FOR BCE_DICE
    ###########################################################
    # Dataset Setup
    input_folders = [i for i in dataset_path.iterdir()]
    assert len(input_folders) == 1
    dataset_name = input_folders[0].name
    dataset_path = dataset_path / dataset_name
    print(f"Dataset: {dataset_path}")

    if user_loss != 'focal':
        focal_alpha = None
        focal_gamma = None
    if user_loss != 'bce_dice':
        bce_weight = None

    run_name = "_"
    # run_name = 'sweep1'

        # Dataset Lists
    train_list = dataset_path / "train.txt"
    val_list = dataset_path / "val.txt"
    test_list = dataset_path / "test.txt"

        # Initialize W&B using your custom function
    wandb_config = {
        "name": run_name,
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "batch_size": bs,
        "user_loss": user_loss,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "bce_weight": bce_weight,
        "max_epoch": max_epoch,
    }
    wandb_logger = wandb_initialization(job_type, repo_root, project, dataset_name, run_name,train_list, val_list, test_list, wandb_config)

    config = wandb.config

    print(f"---Current config: {wandb.config}")
    if user_loss == "focal":
        print(f"---focal_alpha: {wandb.config.get('focal_alpha', 'Not Found')}")
        print(f"---focal_gamma: {wandb.config.get('focal_gamma', 'Not Found')}")
        loss_desc = f"{user_loss}_{config.focal_alpha}_{config.focal_gamma}" 
    elif user_loss == "bce_dice":
        loss_desc = f"{user_loss}_{config.bce_weight}"
    else:
        loss_desc = user_loss
            

    run_name = f"{dataset_name}_{timestamp}_BS{config.batch_size}_s{config.subset_fraction}_{loss_desc}"  

    wandb.run.name = run_name
    wandb.run.save()
    print(f"---config.name: {config.name}")

    if is_sweep_run():
        print(">>> IN SWEEP MODE <<<")
    
    persistent_workers = num_workers > 0
    if job_type == "train":
        print(">>> Creating data loaders")
        train_dl = create_subset(train_list, dataset_path, 'train', subset_fraction, inputs, bs, num_workers, persistent_workers)
        val_dl = create_subset(val_list, dataset_path, 'val', subset_fraction, inputs, bs, num_workers, persistent_workers)

    if test:
        test_dl = create_subset(test_list, dataset_path, 'test', subset_fraction, inputs, bs, num_workers, persistent_workers)
        ckpt_to_test = next(test_ckpt_path.rglob("*.ckpt"), None)
        if ckpt_to_test is None:
            raise FileNotFoundError(f"No checkpoint found in {test_ckpt_path}")
    # Model Initialization
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED).to(device)

    loss_fn = loss_chooser(user_loss, config.focal_alpha, config.focal_gamma, config.bce_weight)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,  # Stop if no improvement for 3 consecutive epochs
        mode="min",
    )
    # Trainer Setup
    ckpt_dir = repo_root / "5checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=run_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    if early_stop:
        callbacks=[checkpoint_callback, early_stopping]
    else:
        callbacks=[checkpoint_callback]
    # ---------- precision & accelerator automatically picked ------------- #
    #  – MPS  : bf16 autocast, accelerator="mps"
    #  – CUDA : fp16 autocast, accelerator="gpu"
    #  – CPU  : fp32
    #

    accelerator = (
        "mps"
        if device.type == "mps"
        else "gpu"
        if device.type == "cuda"
        else "cpu"
    )

    precision = (
        "bf16-mixed"
        if device.type == "mps"
        else "16-mixed"
        if device.type == "cuda"
        else 32
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=LOGSTEPS,
        max_epochs=config.max_epoch,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        fast_dev_run=DEVRUN,
        num_sanity_val_steps=2,
        callbacks=callbacks,
    )




    # Training or Testing
    if train:
        print(">>> Starting training")
        training_loop = Segmentation_training_loop(model, loss_fn, save_path, user_loss)
        trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl)
    elif test:
        print(f">>> Starting testing with checkpoint: {ckpt_to_test}")
        training_loop = Segmentation_training_loop.load_from_checkpoint(
            ckpt_to_test, model=model, loss_fn=loss_fn, save_path=save_path
        )
        trainer.test(model=training_loop, dataloaders=test_dl)

    # Cleanup
    run_time = (time.time() - start) / 60
    print(f">>> Total runtime: {run_time:.2f} minutes")
    wandb.finish()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()