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
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
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

from scripts.train_modules.z.boundaryloss import BoundaryLoss
from scripts.train_modules.train_helpers import *
from scripts.train_modules.train_classes import FloodDataset, UnetModel, SimpleCNN, SurfaceLoss , ResNetBinaryClassifier 
from scripts.train_modules.train_functions import handle_interrupt
from scripts.train_modules.train_functions import calculate_metrics, log_metrics_to_wandb
from scripts.train_modules.training_loops import Segmentation_training_loop

#########################################################################


load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

#TODO add DICE , NSD, IOU, PRECISION, RECALL metrics

@click.command()
@click.option('--evaluate', is_flag=True, show_default=False)
@click.option('--reproduce', is_flag=True, show_default=False)

#########################################################################

def main(evaluate=None, reproduce=None):
    '''
    CONDA ENVIRONMENT = 'floodenv2"

    expects that the data is in tile_root, with 3 tile_lists for train, test and val
    ***NAVIGATE IN TERMINAL TO THE UNMAPPED ADRESS TO RUN THIS SCRIPT.***

    cd //cerndata100/AI_files/users/ai_flood_service/1new_data/3scripts/training

    TODO poss switch to ClearML (opensource)
    '''
    print('---in main')
    start = time.time()

    signal.signal(signal.SIGINT, handle_interrupt)

    torch.set_float32_matmul_precision('medium')  # TODO try high
    pl.seed_everything(42, workers=True, verbose = False)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    repo_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2")
    # dataset_name = 'ds_flaiv2_split'
    # dataset_path  = repo_path / "1data" / "3final" / dataset_name
    # dataset_name = 'ds_flaiv2_split'
    dataset_path  = repo_path / "1data" / "3final" / "train_input"
    project = "floodai_v2"
    # DATA PARAMS
    dataset_version = 'pixel threshold 0.5'
    subset_fraction = 1 # 1 = full dataset
    bs = 16
    max_epoch = 10
    # DATALOADER PARAMS
    num_workers = 0
    # WANDB PARAMS
    WBOFFLINE = True
    LOGSTEPS = 50 # STEPS/EPOCH = DATASET SIZE / BATCH SIZE
    # MODEL PARAMS
    PRETRAINED = True
    inputs = ['hh' , 'mask'] 
    in_channels = len(inputs)
    DEVRUN = 1
    metric_threshold = 0.9
    loss = "BCEWithLogitsLoss"
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    input = [i for i in dataset_path.iterdir()]
    assert len(input) == 1
    dataset_name = input[0].name
    dataset_path = dataset_path / dataset_name
    print(f'---dataset path = {dataset_path}')

    mode = "train"
    if evaluate:
        mode = "test"
    persistent_workers = False
    if num_workers > 0:
        persistent_workers = True

    if not dataset_path.exists():
        print('---base path not exists')

    if not reproduce:
        print('---start train')
        # itereate through dataset_paths here 
        train_list = dataset_path / "train.txt" # converts to absolute path with resolve
        test_list = dataset_path / "test.txt"
        val_list  = dataset_path / "val.txt"

        # WANDB INITIALISATION
        with wandb.init(project=project, 
                        job_type="data-process", 
                        name='load-data', 
                        mode="online",
                        dir=repo_path / "4results", 
                        settings=wandb.Settings(program=__file__)
                        ) as run:
                # ARTIFACT CREATION
                data_artifact = wandb.Artifact(
                    dataset_name, 
                    type="dataset",
                    description=f"{dataset_name} all tiles from unosat original dataset",
                    metadata={"train_list": str(train_list),
                            "test_list": str(test_list),
                            "val_list": str(val_list)})
                # TODO check the uri - only uri accepted? may work in Z: now
                train_list_path = str(train_list).replace("\\", "/")
                train_list_uri = f"file:////{train_list_path}"
                # ADDING REFERENCES
                data_artifact.add_reference(train_list_uri, name="train_list")
                run.log_artifact(data_artifact, aliases=[dataset_version, dataset_name])
    else:
        print('---reproduce')
        artifact_dataset_name = f'unosat_emergencymapping-United Nations Satellite Centre/{project}/{dataset_name}/ {dataset_name}'
        
        print("---initialising Wandb")
        with wandb.init(project=project, job_type="reproduce", name='reproduce', mode="disabled") as run:
            data_artifact = run.use_artifact(artifact_dataset_name)
            metadata_data = data_artifact.metadata
            print("Current Artifact Metadata:", metadata_data)
            train_list = Path(metadata_data['train_list'])
            test_list = Path(metadata_data['test_list'])
            val_list = Path(metadata_data['val_list'])    
    # TODO check the path chains here
    train_dl = create_subset(train_list, dataset_path, 'train' , subset_fraction, inputs, bs, num_workers, persistent_workers)
    test_dl = create_subset(test_list, dataset_path, 'test', subset_fraction, inputs, bs, num_workers, persistent_workers)   
    val_dl = create_subset(val_list, dataset_path, 'val',  subset_fraction, inputs, bs, num_workers, persistent_workers)  

    # MAKE MODEL
    # model = SimpleCNN(in_channels=in_channels, classes=2)
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED)
    # print('---model =', model)
    model = model.to('cuda')  # Ensure the model is on GPU
    device = next(model.parameters()).device
    print(f'---model location: {device}')

    run_name = f'm:{mode}_FR:{subset_fraction}_BS:{bs}_CH{in_channels}_EP:{max_epoch}_{loss}'
    print(f'---RUN NAME= {run_name}')

    wandb_logger = WandbLogger(
        project="floodai_v2",
        name=run_name,
    )

    # Logging additional run metadata (e.g., dataset info)
    wandb_logger.experiment.config.update({ 
        "dataset_name": dataset_name,
        "max_epoch": max_epoch,
        "subset_fraction": subset_fraction,
        "devrun": DEVRUN,
        "offline_mode": WBOFFLINE
    })

    ckpt_dir = repo_path / "4results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print('---ckpt exists = ', ckpt_dir.exists())
          # DEFINE THE TRAINER
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,  # Save checkpoints locally in this directory
        filename=f"{dataset_name} {subset_fraction} {max_epoch:02d}",  # Custom filename format
        # filename="best-checkpoint",  # Custom filename format
        monitor="val_loss",              # Monitor validation loss
        mode="min",                      # Save the model with the lowest validation loss
        save_top_k=1                   # Only keep the best model
    )
    
    print('---trainer')
    trainer= pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=LOGSTEPS,
        max_epochs=max_epoch,
        accelerator='gpu', 
        devices=1, 
        precision='16-mixed',
        fast_dev_run=DEVRUN,
        num_sanity_val_steps=2,
        callbacks = [checkpoint_callback]
    )
    print('---trainer done')

    if not evaluate:
        print('---training / val  (NOT test)')
        training_loop = Segmentation_training_loop(model)

        trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl,)

        best_val_loss = trainer.callback_metrics.get("val_loss", None)
        if best_val_loss is not None:
            wandb_logger.experiment.summary["final_val_loss"] = float(best_val_loss)

        # RUN A TRAINER.TEST HERE FOR A SIMPLE ONE RUN TRAIN/TEST CYCLE        
        # trainer.test(model=training_loop, dataloaders=test_dl, ckpt_path='best')
        
        
    else:
        print('---evaluate')
        
        # TODO encapuulate in 'get checkpoint name' function
        # ckpt = ckpt_dir / f"{dataset_name} f{subset_fraction} ep{max_epoch:02d}.ckpt"

        ckpt = ckpt_dir / f'{dataset_name} {subset_fraction} {max_epoch:02d}.ckpt'

        training_loop = Single_channel_training_loop.load_from_checkpoint(ckpt, model=model, accelerator='gpu')
        training_loop = training_loop.cuda()
        training_loop.eval()
        # trainer.test(model=training_loop, dataloaders=test_dl)

        for batch in tqdm(test_dl, total=len(test_dl)):
            images, masks = batch
            # im = Func.normalize(image.repeat(1,3,1,1)/255, mean=imagenet_stats[0], std=imagenet_stats[1])

            with torch.no_grad():
                logits = training_loop(images.cuda())
                logits = logits.cpu().softmax(1)[:,1:] # Softmax + select "flood" class

            metrics = calculate_metrics(logits, masks, metric_threshold)

            # Log metrics and visualizations to wandb
            log_metrics_to_wandb(metrics, wandb_logger, logits, masks)

    # Ensure the model is on GPU
    model = model.to('cuda')
    end = time.time()
    run_time = f'{(end - start)/60:.2f}'
    if wandb.run:
        wandb_logger.experiment.config.update({ 
        "run_time": run_time,
    })
        wandb.finish()  # Properly finish the W&B run
    torch.cuda.empty_cache()
    # end timing
    
    print(f'>>>total time = {run_time} minutes')  


if __name__ == '__main__':
    main()
