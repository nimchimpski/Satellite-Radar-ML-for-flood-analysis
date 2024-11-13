'''
USE UNC PATHS AS WANDB GETS PISSED WITH MAPPED NETWORK DRIVES
'''
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
from PIL import Image
from tqdm import tqdm
from operator import itemgetter, mul
from functools import partial
from wandb import Artifact
#--------------------------------------------
from segmentation_training_loop import Segmentation_training_loop
from boundaryloss import BoundaryLoss
from train_helpers import *
from train_classes import FloodDataset, UnetModel, SimpleCNN, SurfaceLoss

load_dotenv()

#TODO add DICE , NSD, IOU, PRECISION, RECALL metrics

@click.command()
@click.option('--test', is_flag=True, show_default=False)
@click.option('--reproduce', is_flag=True, show_default=False)

#########################################################################

def main(test=None, reproduce=None):
    '''
    expects that the data is in tile_root, with 3 tile_lists for train, test and val
    ***NAVIGATE IN TERMINAL TO THE UNMAPPED ADRESS TO RUN THIS SCRIPT.***
    cd //cerndata100/AI_files/users/ai_flood_service/1new_data/3scripts/training
    '''
    print('---in main')
    start = time.time()

    base_path = Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\1data\2interim")
    dataset_name = 'UNOSAT_FloodAI_Dataset_v2_norm'
    dataset_path  = base_path / dataset_name
    if not dataset_path.exists():
        print('---base path not exists')

    torch.set_float32_matmul_precision('medium')  # TODO try high
    pl.seed_everything(42, workers=True, verbose = False)

    project = "floodai_v2"
    dataset_version = 'v1'

    # iterate through events
    event = dataset_path / "FL_20200730_MMR1C48"

    if not reproduce:
        print('---start train')
        # itereate through events here 
        train_list = event / "train.txt" # converts to absolute path with resolve
        test_list = event / "test.txt"
        val_list  = event / "val.txt"
        # WANDB INITIALISATION
        with wandb.init(project=project, 
                        job_type="data-process", 
                        name='load-data', 
                        mode="online",
                        dir=str(Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\4results")), 
                        settings=wandb.Settings(program=__file__)
                        ) as run:
                print('---initialising Wandb')
                # ARTIFACT CREATION
                data_artifact = wandb.Artifact(
                    dataset_name, 
                    type="dataset",
                    description=f"{dataset_name} dataset for FloodAI training",
                    metadata={"train_list": str(train_list),
                            "test_list": str(test_list),
                            "val_list": str(val_list)})
                # data_artifact.add_reference(name="train_list", uri=str(train_list)) ****CHANGED*****
                # Convert to absolute path and correct URI format    
                train_list_path = str(train_list).replace("\\", "/")
                train_list_uri = f"file:////{train_list_path}"
                print(f"---uri path: {train_list_uri}")
                # Create an artifact and add the reference
                data_artifact = Artifact(name="dataset_name", type="dataset")
                data_artifact.add_reference(train_list_uri, name="train_list")
                run.log_artifact(data_artifact, aliases=[dataset_version, dataset_name])
    else:
        print('---reproduce')
        artifact_dataset_name = 'unosat_emergencymapping-United Nations Satellite Centre/{}/{}:{}'.format(project, dataset_name, dataset_name)
        
        print("---initialising Wandb")
        with wandb.init(project=project, job_type="reproduce", name='reproduce', mode="disabled") as run:
            data_artifact = run.use_artifact(artifact_dataset_name)
            metadata_data = data_artifact.metadata
            print("Current Artifact Metadata:", metadata_data)
            train_list = Path(metadata_data['train_list'])
            test_list = Path(metadata_data['test_list'])
            val_list = Path(metadata_data['val_list'])    


    bs = 32
    max_epoch = 1
    inputs = ['vv', 'vh', 'grd', 'dem' , 'slope', 'mask', 'analysis extent'] 
    in_channels = len(inputs) 

    # Define the fraction of the dataset you want to use
    subset_fraction = 0.1  # Use 10% of the dataset for quick experiments

    train_dl = create_subset(train_list, event, 'train' , subset_fraction, inputs, bs)
    test_dl = create_subset(test_list, event, 'test', subset_fraction, inputs, bs)   
    val_dl = create_subset(val_list, event, 'val',  subset_fraction, inputs, bs)  

    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=2, pretrained=True)
    # Instantiate the model
    # model = SimpleCNN(in_channels=in_channels, classes=2)
    model = model.to('cuda')  # Ensure the model is on GPU
    # check model location
  # Get the device of the model by checking one of its parameters
    device = next(model.parameters()).device
    print(f'---model location: {device}')

    experiment_name = 'unet_unosat-ai-dataset_grd-{}_epoch-{}_{}'.format(in_channels, max_epoch, 'crossentropy')
    print('---EXPERIMENT NAME= {}'.format(experiment_name))
    wandb_logger = WandbLogger(
        project="floodai_v2",
        name=experiment_name)
    
      # DEFINE THE TRAINER
    checkpoint_callback = ModelCheckpoint(
    dirpath=r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\4results\checkpoints",  # Save checkpoints locally in this directory
    # filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",  # Custom filename format
    filename="best-checkpoint",  # Custom filename format
    monitor="val_loss",              # Monitor validation loss
    mode="min",                      # Save the model with the lowest validation loss
    save_top_k=1                     # Only keep the best model
)
    
    print('---trainer')
    trainer= pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epoch,
        accelerator='gpu', 
        devices=1, 
        precision='16-mixed',
        fast_dev_run=True,
        num_sanity_val_steps=0,
        callbacks = [checkpoint_callback]
    )
    print('---trainer done')

    if not test:
        print('---not test ')
        training_loop = Segmentation_training_loop(model)
        print('#################  training loop  ##################')
        trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl,)
        # RUN A TRAINER.TEST HERE FOR A SIMPLE ONE RUN TRAIN/TEST CYCLE        
        # trainer.test(model=training_loop, dataloaders=test_dl, ckpt_path='best')
        
    else:
        print('---test')
        threshold = 0.9

        ckpt = Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\4results\checkpoints\best_checkpoint")

        training_loop = Segmentation_training_loop.load_from_checkpoint(ckpt, model=model, accelerator='gpu')
        training_loop = training_loop.cuda()
        training_loop.eval()
        # trainer.test(model=training_loop, dataloaders=test_dl)

        preds, gts = [], []
        cnt = 0
        tps, fps, fns, tns = [], [], [], []
        nsds = []
        for sample in tqdm(test_dl, total=len(test_dl)):
            image, mask = sample
            # im = Func.normalize(image.repeat(1,3,1,1)/255, mean=imagenet_stats[0], std=imagenet_stats[1])

            with torch.no_grad():
                logits = training_loop(image.cuda())
                logits = logits.cpu()

            logits = logits.softmax(1)[:,1:]

            tp, fp, fn, tn = smp.metrics.get_stats(logits, mask.long(), mode='binary', threshold=threshold)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)

            for pd, gt in zip(logits, mask):
                nsd_value = nsd(pd[0].cpu().numpy()>threshold, gt[0].cpu().numpy().astype(bool))
                nsds.append(nsd_value)

            preds.append(logits.detach().cpu().numpy())
            gts.append(mask.detach().numpy())
            # for pd, gt in zip(logits, mask):
            #     plt.subplot(131)
            #     plt.imshow(pd[0])
            #     plt.subplot(132)
            #     plt.imshow(pd[0] > 0.8)
            #     plt.subplot(133)
            #     plt.imshow(gt[0].cpu())
            #     plt.show()
            
        tps = torch.vstack(tps)
        fps = torch.vstack(fps)
        fns = torch.vstack(fns)
        tns = torch.vstack(tns)
            
        water_accuracy = tps.sum() / (tps.sum() + fns.sum())
        iou = smp.metrics.iou_score(tps ,fps, fns, tns).mean()
        precision = smp.metrics.precision(tps ,fps, fns, tns).mean()
        recall = smp.metrics.recall(tps ,fps, fns, tns).mean()
        nsd_avg = np.mean(nsds)

        num_pixels = len(test_dl.dataset)*256*256
        tp_perc = tps.sum() / num_pixels * 100
        tn_perc = tns.sum() / num_pixels * 100
        fp_perc = fps.sum() / num_pixels * 100
        fn_perc = fns.sum() / num_pixels * 100

        print("water accuracy: {}".format(water_accuracy))
        print("IoU: {}".format(iou))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("TP percentage: {}".format(tp_perc))
        print("TN percentage: {}".format(tn_perc))
        print("FP percentage: {}".format(fp_perc))
        print("FN percentage: {}".format(fn_perc))
        print("NSD: {}".format(nsd_avg))


    # Ensure the model is on GPU
    model = model.to('cuda')

    # Now print the summary, but with the input_size explicitly indicating it's on the GPU
    # summary(model, input_size=(3, 256, 256), device="cuda")

    if wandb.run:
        wandb.finish()  # Properly finish the W&B run
    torch.cuda.empty_cache()
    # end timing
    end = time.time()
    print(f'>>>total time = {end - start}')  


if __name__ == '__main__':
    main()
