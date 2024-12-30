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
# .............................................................
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
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
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

from scripts.train_modules.train_helpers import *
from scripts.train_modules.train_classes import  UnetModel,   Segmentation_training_loop 
from scripts.train_modules.train_functions import handle_interrupt, loss_chooser, wandb_initialization, job_type_selector

#.............................................................
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

@click.command()
@click.option('--train', is_flag=True, help="Train the model")
@click.option('--test', is_flag=True, help="Test the model")
def main(train, test):
    """
    CONDA ENVIRONMENT = 'floodenv2'
    """
    if train and test:
        raise click.UsageError("You can only specify one of --train or --test.")
    elif not (train or test):
        while True:
            user_input = input("Choose an option (--train or --test): ").strip().lower()
            if user_input == "--train":
                click.echo("Training the model...")
                train = True
                break
            elif user_input == "--test":
                click.echo("Testing the model...")
                test = True
                break
            else:
                click.echo("Invalid input. Please choose '--train' or '--test'.")

    job_type = "train" if train else "test"

    # Basic Setup
    start = time.time()
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    signal.signal(signal.SIGINT, handle_interrupt)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    ###########################################################
    # Paths and Parameters
    repo_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2")
    dataset_path = repo_path / "1data" / "3final" / "train_INPUT"
    test_ckpt_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\5checkpoints\ckpt_INPUT")
    save_path = repo_path / "4results"
    project = "TSX"

    subset_fraction = 1
    bs = 16
    max_epoch = 400
    early_stop = False
    patience=5
    num_workers = 8
    WBOFFLINE = False
    LOGSTEPS = 50
    PRETRAINED = True
    inputs = ['hh', 'mask']
    in_channels = 1
    DEVRUN = 0
    user_loss = 'focal'
    focal_alpha = 0.9
    focal_gamma = 5.0
    bce_weight = 0.5
    ###########################################################
    # Dataset Setup
    input_folders = [i for i in dataset_path.iterdir()]
    assert len(input_folders) == 1
    dataset_name = input_folders[0].name
    dataset_path = dataset_path / dataset_name

    loss_desc = f"{user_loss}_{focal_alpha}_{focal_gamma}" if user_loss == "focal" else f"{user_loss}_{bce_weight}"
    run_name = f"{dataset_name}_{timestamp}_BS{bs}_s{subset_fraction}_{loss_desc}"

        # Dataset Lists
    train_list = dataset_path / "train.txt"
    val_list = dataset_path / "val.txt"
    test_list = dataset_path / "test.txt"

        # Initialize W&B using your custom function
    wandb_config = {
        "name": run_name,
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "bs": bs,
        "user_loss": user_loss,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "bce_weight": bce_weight,
    }
    wandb_logger = wandb_initialization(job_type, repo_path, project, dataset_name, run_name,train_list, val_list, test_list,wandb_config)

    persistent_workers = num_workers > 0
    train_dl = create_subset(train_list, dataset_path, 'train', subset_fraction, inputs, bs, num_workers, persistent_workers)
    val_dl = create_subset(val_list, dataset_path, 'val', subset_fraction, inputs, bs, num_workers, persistent_workers)

    if test:
        test_dl = create_subset(test_list, dataset_path, 'test', subset_fraction, inputs, bs, num_workers, persistent_workers)
        ckpt_to_test = next(test_ckpt_path.rglob("*.ckpt"), None)
        if ckpt_to_test is None:
            raise FileNotFoundError(f"No checkpoint found in {test_ckpt_path}")

    # Model Initialization
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED).to('cuda')
    loss_fn = loss_chooser(user_loss, focal_alpha, focal_gamma, bce_weight)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,  # Stop if no improvement for 3 consecutive epochs
        mode="min",
    )

    # Trainer Setup
    ckpt_dir = repo_path / "5checkpoints"
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

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=LOGSTEPS,
        max_epochs=max_epoch,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        fast_dev_run=DEVRUN,
        num_sanity_val_steps=2,
        callbacks=callbacks
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
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()