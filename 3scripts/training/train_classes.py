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
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pathlib import Path
import tifffile as tiff

class FloodDataset(Dataset):
    def __init__(self, tile_list, tile_root, stage='train', inputs=None):
        with open(tile_list, 'r') as _in:
            sample_list = _in.readlines()
        self.sample_list = [t[:-1] for t in sample_list]
        
        if stage == 'train':
            self.tile_root = Path(tile_root, 'train')
        elif stage == 'test':
            self.tile_root = Path(tile_root, 'test')
        elif stage == 'val':
            self.tile_root = Path(tile_root, 'val')
    
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
        tile = tiff.imread(Path(self.tile_root, filename))

################### SPECIFY THE INPUTS HERE ##############################

        input_idx = []
        for input in self.inputs:
            input_idx.append(self.input_map_unosat[input])

        model_input = tile[:,:,input_idx]

        
        # onehot = class2one_hot(torch.from_numpy(val_mask).to(torch.int64), K=2)
        # dist = one_hot2dist(onehot.cpu().numpy().transpose(1,0,2), resolution=[1,1])

        model_input = Func.to_tensor(model_input)
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
    
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, classes, kernel_size=3, padding=1)  # Output with 2 channels

        # Optional: Use an upsampling layer to match the output size with the ground truth
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        
        # Optional: Use upsampling if necessary to match the size
        x = self.upsample(x)
        return x
