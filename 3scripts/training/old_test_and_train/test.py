import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as Func
from torchvision import transforms

# import rasterio
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from utils import simplex, one_hot, class2one_hot, one_hot2dist

from PIL import Image

from tqdm import tqdm
from torch import Tensor, einsum
from operator import itemgetter, mul
from functools import partial

pl.utilities.seed.seed_everything(seed=42, workers=False)

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
            self.inputs = ['vv']
        else:
            self.inputs = inputs

        self.input_map_dlr = {
            'vv':0, 'vh': 1, 'dem':2, 'slope':3, 'mask':4, 'analysis extent':5
        }
        self.input_map_unosat = {
            'vv':0, 'vh': 1, 'grd':2, 'dem':3, 'slope':4, 'mask':5, 'analysis extent':6
        }

        self.mean = [-1476.4767353809093] # For dlr dataset
        self.std = [1020.6359514352469]
        self.imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

        # self.dist_transform = dist_map_transform([1,1], 2)


    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.sample_list)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        tile = tiff.imread(osp.join(self.tile_root, filename))

        input_idx = []
        for input in self.inputs:
            input_idx.append(self.input_map_unosat[input])

        model_input = tile[:,:,input_idx]
        # val_mask = tile[:,:,4]
        val_mask = tile[:,:,5]
        
        # onehot = class2one_hot(torch.from_numpy(val_mask).to(torch.int64), K=2)
        # dist = one_hot2dist(onehot.cpu().numpy().transpose(1,0,2), resolution=[1,1])

        model_input = Func.to_tensor(model_input/255)
        model_input_norm = Func.normalize(model_input.repeat(3,1,1), self.imagenet_stats[0], self.imagenet_stats[1])
        
        val_mask = Func.to_tensor(val_mask)
        # dist = torch.from_numpy(dist)

        # return [model_input_norm.float(), val_mask.float(), dist.float()]
        return [model_input_norm.float(), val_mask.float()]
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
    

def acc_camvid(input, target, threshold=0.5):
    'define pixel-level accuracy'
    target = target.squeeze(1)
    mask = target
    mask = target != 2
    # return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    return ((input>threshold)[mask]==target[mask]).float().mean()

def acc_background(input, target, threshold=0.5):
    'define pixel-level accuracy just for background'
    mask = target != 1
    # return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    return ((input>threshold)[mask]==target[mask]).float().mean()

def acc_flood(input, target, threshold=0.5):
    'define pixel-level accuracy just for flood'
    mask = target != 0
    # return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
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


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class SegmentFlood(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.save_hyperparameters()
        # self.lovasz_loss = smp.losses.LovaszLoss(mode='binary')
        self.boundary_loss = BoundaryLoss()
        # self.dice_loss = smp.losses.DiceLoss(mode='binary')

    def forward(self,batch):
        return self.model(batch)
    
    def training_step(self,batch, batch_idx):
        image, mask= batch
        logits = self.forward(image)

        logits = logits.softmax(1)
        total_loss = 0

        class_weight = torch.ones_like(mask)
        class_weight[mask==0] = 0.2

        # flood_loss0 = F.binary_cross_entropy(logits, mask.squeeze(1), weight=class_weight)
        # flood_loss0 = F.cross_entropy(logits, mask.squeeze(1).long(), weight=torch.Tensor([0.2,1]).cuda())

        b_loss = self.boundary_loss(logits, mask.long().squeeze(1))

        # flood_loss_1 = self.lovasz_loss(logits, mask)
        # total_loss = flood_loss0 + flood_loss_1

        # total_loss = flood_loss0 + b_loss*0.2
        total_loss = b_loss

        self.log('train_loss', total_loss)
        lr= self._get_current_lr()
        self.log('lr',lr)
        return total_loss
    
    def _get_current_lr(self):
        lr= [x["lr"] for x in self.optimizer.param_groups]
        return torch.Tensor([lr]).cuda()
    
    # def validation_step(self,batch, batch_idx):
    #     image, mask= batch
    #     logits = self.forward(image)
    #     result = {}
    #     flood_loss = F.binary_cross_entropy_with_logits(logits, mask)

    #     tp, fp, fn, tn = smp.metrics.get_stats(logits, mask.long(), mode='binary', threshold=0.5)
    #     iou = smp.metrics.iou_score(tp ,fp, fn, tn)

    #     result["loss"] = flood_loss
    #     result["iou"] = iou.mean()
    #     self.log('iou', iou.mean(), prog_bar=True)

    #     return result

    def validation_step(self,batch, batch_idx):
        image, mask= batch
        logits = self.forward(image)
        logits = logits.softmax(1)
        result = {}
        flood_loss = F.cross_entropy(logits, mask.squeeze(1).long(), weight=torch.Tensor([0.2,1]).cuda())

        tp, fp, fn, tn = smp.metrics.get_stats(logits[:,1:], mask.long(), mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp ,fp, fn, tn)

        result["loss"] = flood_loss
        result["iou"] = iou.mean()
        self.log('iou', iou.mean(), prog_bar=True)

        return result
    
    def validation_epoch_end(self, outputs):
        self.log('epoch',self.trainer.current_epoch)

        avg_iou = find_average(outputs, 'iou')
        self.log('iou', avg_iou, prog_bar=True)

        return {'avg iou': avg_iou}

    def test_step(self, batch, batch_idx):
        image, mask, dist= batch
        logits = self.forward(image)

        pd = logits.cpu().numpy()
        gt = mask.cpu().numpy()

        # for pd_i, gt_i in zip(pd, gt):
        #     plt.subplot(121)
        #     plt.imshow(pd_i[0])
        #     plt.subplot(122)
        #     plt.imshow(gt_i[0])
        #     plt.show()

        return {"results": None}
    
    def configure_optimizers(self):
        params=[x for x in self.model.parameters() if x.requires_grad]
        self.optimizer= torch.optim.AdamW(params,lr=1e-3)
        # scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=8,eta_min=0.6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[8,], gamma=0.1)
        dict_val= {"optimizer":self.optimizer,"lr_scheduler": scheduler}
        return dict_val

def main():
    datacube_tile_root = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\unosat_ai_v2\tiles'

    train_list = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\unosat_ai_v2\train_list.txt"
    test_list = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\unosat_ai_v2\test_list.txt"
    val_list = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\unosat_ai_v2\val_list.txt"


    bs = 32
    max_epoch = 10
    in_channels = 3
    inputs = ['grd']

    train_dataset = FloodDataset(train_list, datacube_tile_root, stage='train', inputs=inputs)
    train_dl = DataLoader(train_dataset, batch_size=bs, num_workers=12, shuffle=True)

    test_dataset = FloodDataset(test_list, datacube_tile_root, stage='test', inputs=inputs)
    test_dl = DataLoader(test_dataset, batch_size=bs, num_workers=12)

    val_dataset = FloodDataset(val_list, datacube_tile_root, stage='val', inputs=inputs)
    val_dl = DataLoader(val_dataset, batch_size=bs, num_workers=12)

    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=2, pretrained=True)

    experiment_name = 'unet_unosat-ai-dataset_grd-{}_epoch-{}_{}'.format(in_channels, max_epoch, 'boundary_loss')
    print('[{}]'.format(experiment_name))
    wandb_logger = WandbLogger(
        project="floodai_retrain",
        name=experiment_name)
    trainer= pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epoch,
        accelerator='gpu', 
        devices=1, 
        fast_dev_run=False,
        num_sanity_val_steps=0,
        default_root_dir=r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\model\train\experiments")
    
    threshold = 0.9

    ckpt = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\model\train\experiments\floodai_retrain\7q0o34u1\checkpoints\epoch=9-step=19239.ckpt"
    floodsegment = SegmentFlood.load_from_checkpoint(ckpt, model=model, accelerator='gpu')
    floodsegment = floodsegment.cuda()
    floodsegment.eval()
            
    # trainer.test(model=floodsegment, dataloaders=test_dl)

    preds, gts = [], []
    cnt = 0
    tps, fps, fns, tns = [], [], [], []
    nsds = []
    for sample in tqdm(test_dl, total=len(test_dl)):
        image, mask = sample
        # im = Func.normalize(image.repeat(1,3,1,1)/255, mean=imagenet_stats[0], std=imagenet_stats[1])

        with torch.no_grad():
            logits = floodsegment(image.cuda())
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


if __name__ == '__main__':
    main()
