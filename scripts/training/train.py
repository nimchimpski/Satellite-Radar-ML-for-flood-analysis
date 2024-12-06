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
# .............................................................

from scripts.train_modules.train_helpers import *
from scripts.train_modules.train_classes import  UnetModel,   Segmentation_training_loop , BoundaryLoss, SurfaceLoss, DiceLoss, FocalLoss
from scripts.train_modules.train_functions import handle_interrupt
from scripts.train_modules.train_functions import calculate_metrics, log_metrics_to_wandb
#########################################################################

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

@click.command()
@click.option('--test', is_flag=True, show_default=False)
@click.option('--reproduce', is_flag=True, show_default=False)
@click.option('--debug', is_flag=True, show_default=False)
#########################################################################

def main(test=None, reproduce=None, debug=None):
    '''
    CONDA ENVIRONMENT = 'floodenv2"
    expects that the data is in tile_root, with 3 tile_lists for train, test and val
    ***NAVIGATE IN TERMINAL TO THE UNMAPPED ADRESS TO RUN THIS SCRIPT.***
    cd //cerndata100/AI_files/users/ai_flood_service/1new_data/3scripts/training
    '''
    print('>>>in main')
    start = time.time()
    signal.signal(signal.SIGINT, handle_interrupt)
    torch.set_float32_matmul_precision('medium')  # TODO try high
    pl.seed_everything(42, workers=True, verbose = False)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    repo_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2")
    dataset_path  = repo_path / "1data" / "3final" / "train_input"
    save_path = repo_path / "4results"
    project = "TSX"
    # DATA PARAMS
    
    subset_fraction = 1 # 1 = full dataset
    bs = 16
    max_epoch = 10
    # DATALOADER PARAMS
    num_workers = 0
    # WANDB PARAMS
    WBOFFLINE = False
    wandb_mode = "online"
    LOGSTEPS = 50 # STEPS/EPOCH = DATASET SIZE / BATCH SIZE
    # MODEL PARAMS
    PRETRAINED = True
    inputs = ['hh' , 'mask'] 
    in_channels = 1
    DEVRUN = 0
    metric_threshold = 0.9
    loss = "WEIGHTED_BCE" # "FOCALLOSS" "DICE" SURFACE" "BOUNDARY"

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    input_folders = [i for i in dataset_path.iterdir()]
    assert len(input_folders) == 1
    dataset_name = input_folders[0].name
    dataset_path = dataset_path / dataset_name
    print(f'>>>dataset path = {dataset_path}')
    if debug:
        wandb_mode = "disabled"
        logger = None
    else:
        run_name = f'{dataset_name}__BS{bs}__EP{max_epoch}_{loss}'
        wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        mode='online',
    )
    logger = wandb_logger
    dataset_version = run_name

    if loss == "WEIGHTED_BCE":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    elif loss == "FOCAL":
        loss_fn = FocalLoss(alpha=0.25, gamma=2)
    elif loss == "DICE":
        loss_fn = DiceLoss()
    elif loss == "SURFACE":
        loss_fn = SurfaceLoss()
    elif loss == "BOUNDARY":
        loss_fn = BoundaryLoss()
    else:
        raise ValueError(f"Unknown loss: {loss}")

    persistent_workers = False
    if num_workers > 0:
        persistent_workers = True

    if not dataset_path.exists():
        print('>>>base path not exists')

    if not reproduce:
        print('>>>-start train or eval')
        # itereate through dataset_paths here 
        train_list = dataset_path / "train.txt" # converts to absolute path with resolve
        test_list = dataset_path / "test.txt"
        val_list  = dataset_path / "val.txt"

        # WANDB INITIALISATION
        with wandb.init(project=project, 
                        job_type="data-process", 
                        name='load-data', 
                        mode=wandb_mode,
                        dir=repo_path / "4results", 
                        settings=wandb.Settings(program=__file__)
                        ) as run:
                # ARTIFACT CREATION
                data_artifact = wandb.Artifact(
                    dataset_name, 
                    type="dataset",
                    description=f"{dataset_name} 12 events",
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
        print('>>>reproduce')
        artifact_dataset_name = f'unosat_emergencymapping-United Nations Satellite Centre/{project}/{dataset_name}/ {dataset_name}'
        
        print(">>>initialising Wandb")
        with wandb.init(project=project, job_type="reproduce", name='reproduce', mode="disabled") as run:
            data_artifact = run.use_artifact(artifact_dataset_name)
            metadata_data = data_artifact.metadata
            print(">>>Current Artifact Metadata:", metadata_data)
            train_list = Path(metadata_data['train_list'])
            test_list = Path(metadata_data['test_list'])
            val_list = Path(metadata_data['val_list'])    
    # TODO check the path chains here
    if not test:
        print(f'>>>not test')
        train_dl = create_subset(train_list, dataset_path, 'train' , subset_fraction, inputs, bs, num_workers, persistent_workers)

        # DEBUG
        # for image, mask in train_dl:
        #     num_flood_pixels = (mask == 1).sum()
        #     num_non_flood_pixels = (mask == 0).sum()
        #     print(f">>>[Dataset] Flood Pixels: {num_flood_pixels}, Non-Flood Pixels: {num_non_flood_pixels}")
        #     break  # Print for just one sample to avoid flooding the output
        # for batch in train_dl:
        #     print('>>>batch+ ',batch)  # Ensure batches are correctly created
        #     break
        # for model_input, mask in train_dl:
        #     print(f">>>///Model input shape: {model_input.shape}, Mask shape: {mask.shape}")
        #     break  
        val_dl = create_subset(val_list, dataset_path, 'val',  subset_fraction, inputs, bs, num_workers, persistent_workers)  
    elif test:
        test_dl = create_subset(test_list, dataset_path, 'test', subset_fraction, inputs, bs, num_workers, persistent_workers) 

    # MAKE MODEL
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED)
    # print('>>>model =', model)
    # Check the first convolution layer
    print(model.model.encoder.conv1)

    # dummy_input = torch.randn(16, 1, 256, 256).to('cuda')  # Batch size 16, 1 input channel
    # with torch.no_grad():
    #     outputd = model(dummy_input)
    # print(f">>>/// dummy model output shape: {outputd.shape}")# Expected: torch.Size([16, 1, 256, 256])

    model = model.to('cuda')  # Ensure the model is on GPU
    device = next(model.parameters()).device


    print(f'>>>RUN NAME= {run_name}')

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
        "offline_mode": WBOFFLINE,  
    })

    ckpt_dir = repo_path / "4results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print('>>>ckpt exists = ', ckpt_dir.exists())
          # DEFINE THE TRAINER
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,  # Save checkpoints locally in this directory
        filename=run_name,  # Custom filename format
        # filename="best-checkpoint",  # Custom filename format
        monitor="val_loss",              # Monitor validation loss
        mode="min",                      # Save the model with the lowest validation loss
        save_top_k=1                   # Only keep the best model
    )
    # print(f'>>>///model location: {device}')
    print('>>>trainer')
    trainer= pl.Trainer(
        logger=logger,
        log_every_n_steps=LOGSTEPS,
        max_epochs=max_epoch,
        accelerator='gpu', 
        devices=1, 
        precision='16-mixed',
        fast_dev_run=DEVRUN,
        num_sanity_val_steps=2,
        callbacks = [checkpoint_callback]
    )
    # print('>>>trainer done')

    if not test:
        print('>>>>>>>>>>>>>>>>>>>>   training / val  (NOT test)')
        training_loop = Segmentation_training_loop(model, loss_fn, save_path)

        trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl,)

        best_val_loss = trainer.callback_metrics.get("val_loss", None)
        if best_val_loss is not None:
            wandb_logger.experiment.summary["final_val_loss"] = float(best_val_loss)

        # RUN A TRAINER.TEST HERE FOR A SIMPLE ONE RUN TRAIN/TEST CYCLE        
        # trainer.test(model=training_loop, dataloaders=test_dl, ckpt_path='best')
        
    else:
        print('>>>>>>>>>>>>>> test >>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        
        # ckpt = ckpt_dir / f'{run_name}.ckpt'
        ckpt = ckpt_dir / 'TSX_logclipmm_g_mt0.3__BS16__EP10_WEIGHTED_BCE.ckpt'
        print(f'>>>evaluation ckpt = {ckpt.name}')
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        model.to('cuda')

        training_loop = Segmentation_training_loop.load_from_checkpoint(ckpt, model=model, loss_fn=loss_fn, save_path=save_path, accelerator='gpu')
        print(f'>>>checkpoint loaded ok {ckpt.name}')
        # print(f'>>>ckpt test = {torch.load(ckpt)}')
        # training_loop = training_loop.to('cuda')
        # training_loop.model = training_loop.model.to('cuda')  # Ensure submodules are on GPU
        training_loop.eval()
        # training_loop = training_loop.to('cuda')
        # Debug weight placement
        # for name, param in training_loop.model.named_parameters():
        #     print(f">>>Layer {name}. device {device}: {param.device==device}")
        
        
        # Debugging
        print(f">>>Training loop device: {next(training_loop.parameters()).device}")
        trainer.test(model=training_loop, dataloaders=test_dl)

        print('>>>loading batch')

        for batch in tqdm(test_dl, total=len(test_dl)):
            images, masks = batch
                # Debug device alignment
            images, masks = images.to('cuda'), masks.to('cuda')  # Ensure input tensors are on GPU
            # print(f">>>images device: {images.device}")
            # print(f">>>masks device: {masks.device}")
            model = model.to('cuda')    
            with torch.no_grad():
                try:
                    logits = training_loop(images)
                    # print(f">>>Logits device: {logits.device}")
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    raise

                logits = torch.sigmoid(logits) # Softmax + select "flood" class
            logits = logits.to('cuda')
            masks = masks.to('cuda')
            # print(f">>>Logits device: {logits.device}")
            # print(f">>>Images device: {images.device}")
            try:
                metrics = calculate_metrics(logits, masks, metric_threshold)
            except Exception as e:
                print(f"Error in calculate_metrics: {e}")
                continue


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
