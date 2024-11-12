    # import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset # ***ADDED****
import random # ***ADDED****
from torchvision.transforms import functional as Func
from torchvision import transforms
import numpy as np
import os.path as osp
import tifffile as tiff
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
import click
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional, List
import os
from pathlib import Path # ***ADDED****
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum
from operator import itemgetter, mul
from functools import partial
import wandb
from pytorch_lightning import seed_everything
from dotenv import load_dotenv
from wandb import Artifact # ***ADDED****
# from torchsummary import summary
#  pl.utilities.seed.seed_everything(seed=42, workers=False)  *****CHANGED****
'''
local modules 
'''
from segmentation_training_loop import Segmentation_training_loop
from boundaryloss import BoundaryLoss
from helpers import *

# ****EXPERIMENT WITH THIS****
# torch.set_float32_matmul_precision('high')
'''
expects that the data is in tile_root, with 3 tile_lists for train, test and val
'''

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["WANDB_NETRC"] = "0"  # Disable .netrc usage


pl.seed_everything(42, workers=True, verbose = False)

D = Union[Image.Image, np.ndarray, Tensor]

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]))

def convert_tensor_to_array(tensor):
    return tensor.cpu().numpy()

def create_numpy_array(img):
    return np.array(img)[...]


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                # lambda img: np.array(img)[...],
                partial(create_numpy_array),
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),

                lambda t: t.cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

class FloodDataset(Dataset):
    def __init__(self, tile_list, tile_root, stage='train', inputs=None):
        with open(tile_list, 'r') as _in:
            sample_list = _in.readlines()
        self.sample_list = [t[:-1] for t in sample_list]
        
        if stage == 'train':
            self.tile_root = osp.join(tile_root, 'train')
        elif stage == 'test':
            self.tile_root = osp.join(tile_root, 'test')
        elif stage == 'val':
            self.tile_root = osp.join(tile_root, 'val')
    
        if inputs is None:
            # self.inputs = ['vv', 'vh', 'dem']
            self.inputs = ['vv', 'vh', 'mask']
        else:
            self.inputs = inputs

        self.input_map_dlr = {
            'vv':0, 'vh': 1, 'dem':2, 'slope':3, 'mask':4, 'analysis extent':5
        }
        self.input_map_unosat = {
            'vv':0, 'vh': 1, 'grd':2, 'dem':3, 'slope':4, 'mask':5, 'analysis extent':6
        }

        self.mean = [-1476.4767353809093] * 4 # For dlr dataset ???which layer? should match the num layers
        self.std = [1020.6359514352469] * 4 # as above
        # self.mean = [mean_vv, mean_vh, mean_dem, mean_slope]
        # self.std = [std_vv, std_vh, std_dem, std_slope]

        self.imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

        # self.dist_transform = dist_map_transform([1,1], 2)


    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.sample_list)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        tile = tiff.imread(osp.join(self.tile_root, filename))

################### SPECIFY THE INPUTS HERE ##############################

        input_idx = []
        for input in self.inputs:
            input_idx.append(self.input_map_unosat[input])

        model_input = tile[:,:,input_idx]

        
        # onehot = class2one_hot(torch.from_numpy(val_mask).to(torch.int64), K=2)
        # dist = one_hot2dist(onehot.cpu().numpy().transpose(1,0,2), resolution=[1,1])

        # model_input = Func.to_tensor(model_input/255)
        model_input = model_input.cuda()
        # TODO REDUNDANT ???
        # model_input_norm = Func.normalize(model_input, self.mean, self.std).cuda()

        # extract the mask
        # val_mask = tile[:,:,4]
        val_mask = tile[:,:, self.input_map_unosat['mask']]
        val_mask = Func.to_tensor(val_mask)
        # dist = torch.from_numpy(dist)

        # return [model_input_norm.float(), val_mask.float(), dist.float()]
        return [model_input.float(), val_mask.float()]
        # return model_input.float(), val_mask.float()

class UnetModel(nn.Module):
    def __init__(self,encoder_name='resnet34', in_channels=3, classes=1, pretrained=False):
        super().__init__()
        self.model= smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights='imagenet',
            in_channels=in_channels, 
            classes=classes,
            activation=None)
        
        # if pretrained:
        #     checkpoint_dict= torch.load(f'/kaggle/working/Fold={index}_Model=mobilenet_v2.ckpt')['state_dict']
        #     self.model.load_state_dict(load_weights(checkpoint_dict))
    def forward(self,x):
        x= self.model(x)
        return x

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

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss
    
def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

def create_subset(file_list, datacube_tile_root, stage, inputs=None, bs=32, subset_fraction=0.05):
    dataset = FloodDataset(file_list, datacube_tile_root, stage=stage, inputs=inputs)    
    subset_indices = random.sample(range(len(dataset)), int(subset_fraction * len(dataset)))
    subset = Subset(dataset, subset_indices)
    dl = DataLoader(subset, batch_size=bs, num_workers=12, persistent_workers=True,  shuffle = (stage == 'train'))
    return dl

@click.command()
@click.option('--test', is_flag=True, show_default=False)
@click.option('--reproduce', is_flag=True, show_default=False)

def main(test=None, reproduce=None):
    print('---in main')
    project = "floodai_v2"
    dataset_name = 'UNOSAT_FloodAI_Dataset_v2_norm'
    dataset_version = 'v1'



    if not reproduce:
        print('---start train')
        datacube_tile_root = Path(r"UNOSAT_FloodAI_Dataset_v2_norm")
        train_list = Path("texts/train_list.txt").resolve() # converts to absolute path with resolve
        test_list = Path("texts/test_list.txt").resolve()
        val_list = Path("texts/val_list.txt").resolve()

        # with wandb.init(project=project, job_type="data-process", name='load-data', mode="disabled") as run:
        #     data_artifact = wandb.Artifact(
        #         dataset_name, type="dataset",
        #         description="{} dataset for FloodAI training".format(dataset_name),
        #         metadata={"train_list": train_list,
        #                 "test_list": test_list,
        #                 "val_list": val_list})
        #     # data_artifact.add_reference(name="train_list", uri=str(train_list)) ****CHANGED*****
        #     # Convert to absolute path and correct URI format
        #     absolute_train_list_path = train_list.resolve()
        #     uri_path = absolute_train_list_path.as_uri()           # Debug print to check the formatted path
        #     print(f"Formatted path: {uri_path}")
        #     # Create an artifact and add the reference
        #     data_artifact = Artifact(name="dataset_name", type="dataset")
        #     data_artifact.add_reference(name="train_list", uri=uri_path)
        #     run.log_artifact(data_artifact, aliases=[dataset_version, dataset_name])
    else:
        artifact_dataset_name = 'unosat_emergencymapping-United Nations Satellite Centre/{}/{}:{}'.format(project, dataset_name, dataset_name)
        
        print("---initialising Wandb")
        with wandb.init(project=project, job_type="reproduce", name='reproduce', mode="disabled") as run:
            data_artifact = run.use_artifact(artifact_dataset_name)
            metadata_data = data_artifact.metadata
            print("Current Artifact Metadata:", metadata_data)
            # train_list = Path(metadata_data['train_list'])
            # test_list = Path(metadata_data['test_list'])
            # val_list = Path(metadata_data['val_list'])
            train_list = Path("texts/train_list.txt").resolve()
            test_list = Path("texts/test_list.txt").resolve()
            val_list = Path("texts/val_list.txt").resolve()
            #****CHANGED*****
            # datacube_tile_root = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\unosat_ai_v2\tiles' ****CHANGED*****
            
            datacube_tile_root_raw = r"\\cerndata100\AI_Files\Users\AI_Flood_Service\datacube\tiles\unosat_ai_v2\tiles"
            # datacube_tile_root_raw = r"data\tiles"
            datacube_tile_root = Path(datacube_tile_root_raw)

    bs = 32
    max_epoch = 2
    inputs = ['vv', 'vh', 'grd', 'dem' , 'slope', 'mask', 'analysis extent'] 
    in_channels = len(inputs) 

    """
    create a subset for testing purposes
    """
    # Define the fraction of the dataset you want to use
    subset_fraction = 0.05  # Use 10% of the dataset for quick experiments

    train_dl = create_subset(train_list, datacube_tile_root, 'train' , subset_fraction, inputs=inputs)
    test_dl = create_subset(test_list, datacube_tile_root, 'test', subset_fraction, inputs=inputs)   
    val_dl = create_subset(val_list, datacube_tile_root, 'val',  subset_fraction, inputs=inputs)  

    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=2, pretrained=True)
    # Instantiate the model
    # model = SimpleCNN(in_channels=in_channels, classes=2)
    model = model.to('cuda')  # Ensure the model is on GPU

    experiment_name = 'unet_unosat-ai-dataset_grd-{}_epoch-{}_{}'.format(in_channels, max_epoch, 'crossentropy')
    print('[{}]'.format(experiment_name))
    wandb_logger = WandbLogger(
        project="floodai_v2",
        name=experiment_name)

    
    print('---trainer')
    trainer= pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epoch,
        accelerator='gpu', 
        devices=1, 
        fast_dev_run=False,
        num_sanity_val_steps=0,

        default_root_dir = Path(r'Z:\1NEW_DATA\1data\2interim\TESTS')

    )

    if not test:
        print('---not test ')
        training_loop = Segmentation_training_loop(model)
        trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl,)
        # trainer.test(model=training_loop, dataloaders=test_dl, ckpt_path='best')
        
    else:
        print('---val????')
        threshold = 0.9
        #****CHANGED*****
        # ckpt = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\model\train\experiments\floodai_retrain\7q0o34u1\checkpoints\epoch=9-step=19239.ckpt"
        ckpt_rawstring = r"Z:/1NEW_DATA/1data/2interim/TESTS/lightning_logs/version_0/checkpoints/epoch=1-step=19239.ckpt"
        ckpt = Path(ckpt_rawstring)

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

if __name__ == '__main__':
    main()
