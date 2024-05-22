import numpy as np
import sys
import pandas as pd
import re
import json
import os
import torch
import glob
import pylibjpeg
import pydicom
import random
import platform
import ast
from PIL import Image, ImageStat
import cv2
import imagesize
import copy
import collections
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn import functional as F
from utils import get_logger, set_pandas_display
from bounding_box import bounding_box as bbfn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool
from utils import prepare_mask, fetch_filename
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")

tqdm.pandas()
set_pandas_display()
logger = get_logger('Dataset', 'INFO')

def batch_to_device(batch, device):
    batch_dict = {}
    for key in batch:
        if key == "transformations":
            batch_dict[key] = batch[key]
        else:
            batch_dict[key] = batch[key].to(device)
    return batch_dict

def cropdim(fname, threshold=0):
    flatImage = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    try:
        return [rows[0], cols[0], rows[-1] + 1, cols[-1] + 1]
    except:
        return [np.nan] * 4

# %timeit cv2.resize(img, (512,512,), interpolation = cv2.INTER_CUBIC)
# batch = [self.__getitem__(i) for i in range(4)]
def create_train_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, interpolation = 1, p=1), # interpolation = cv2.INTER_CUBIC,
        A.VerticalFlip(p=cfg.vflip ),
        A.HorizontalFlip(p=cfg.hflip ),
        A.RandomContrast(limit=cfg.RandomContrast, p=0.75),
        A.ShiftScaleRotate(shift_limit=cfg.shift_limit,
                           scale_limit=cfg.scale_limit,
                           value = 0,
                           rotate_limit=cfg.rotate_limit,
                           p=0.75,
                           border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=cfg.p_grid_distortion),
        
        A.Cutout(num_holes=cfg.holes,
                 max_h_size=int(cfg.hole_size*cfg.imagesize[0]),
                 max_w_size=int(cfg.hole_size*cfg.imagesize[1]),
                 fill_value=0,
                 always_apply=True, p=0.75),
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ],
    additional_targets=dict((f'image{i}', 'image') for i in range(500)))


def create_valid_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, p=1), # interpolation = cv2.INTER_CUBIC,
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ],
    additional_targets=dict((f'image{i}', 'image') for i in range(500)))

def create_valid_transforms_no_norm(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, p=1), # interpolation = cv2.INTER_CUBIC,
        ToTensorV2()
        ],
    additional_targets=dict((f'image{i}', 'image') for i in range(500)))

from torch.nn.functional import pad
def collate_fn(batch):
    
    maxlen = max([ len(b['mask']) for b in batch ] )
    
    # batch = [b for b in batch if b is not None]
    
    cols = 'StudyUID label'.split()
    batchout = dict((k,torch.stack([b[k] for b in batch])) for k in cols)
    batchout 
    
    batchout['imageidx'] = torch.cat([b['mask'].long() * t for t,b in enumerate(batch)])
    batchout['image'] = torch.cat([b['image'] for b in batch])
    batchout['transformations'] = [b['transformations'] for b in batch]

    return batchout

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

'''
df = pd.read_csv(cfg.train_df)
mode= "train"
cfg.norm_on_cuda = False
class self:
    1
self = CustomDataset(df, cfg, mode = mode)

batch = [self.__getitem__(i) for i in tqdm(range(0, 40, 20))]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
'''

'''
        https://arxiv.org/pdf/2010.13336.pdf
        1) Soft tissue window (w1 = 300; c1 = 80); 
        2) Standard bone window (w2 = 1800; c2 = 500); 
        3) Gross bone window (w3 = 650; c3 = 400).
'''

class CustomDataset(Dataset):

    #def __init__(self, df, mode, tokenizer, max_pos, smoothbreaks = False):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.metadf = pd.read_csv(cfg.meta_df).set_index('StudyInstanceUID')
        self.bbdf = pd.read_csv(cfg.bbox_df).set_index('StudyInstanceUID')
        self.bbadf = pd.read_csv(cfg.bboxall_df).set_index('StudyInstanceUID')
        
        if platform.system() == 'Darwin':
            local_dirs = glob.glob(f'{cfg.data_folder}/*')
            local_dirs = [i.split('/')[-1] for i in local_dirs]
            self.df = self.df[df.StudyInstanceUID.isin(local_dirs)].reset_index(drop = True)
        self.windows = self.cfg.windows #[ (500, 1800), (400, 650), (80, 300)]
        if mode!='test':
            self.df = self.df[~self.df.StudyInstanceUID.isin(self.cfg.drop_scans)]
            self.df = self.df[self.df.StudyInstanceUID.isin(self.bbdf.index)]
        
        # Get the bounding boxes
        grp_cols='StudyInstanceUID slice_numbers'.split()
        '''
        self.bbadf = \
        self.bbadf.reset_index().groupby(grp_cols)['x0 y0'.split()].min().join(
            self.bbadf.reset_index().groupby(grp_cols)['x1 y1'.split()].max()
            ).join(
                self.bbadf.reset_index().groupby(grp_cols)['has_bbox'].mean()
                ).reset_index()
        '''
        self.bbadf = \
            self.bbadf.reset_index()\
                .groupby(grp_cols, as_index=False, group_keys=False)\
                ['x0 y0 x1 y1 has_bbox'.split()].min()
        # Roll the min or max in window
        gg = self.bbadf.groupby('StudyInstanceUID', as_index=False, group_keys=False)
        gg1 = gg['x0 y0'.split()].rolling(24*3, 1, center = True).min()
        gg2 = gg['x1 y1 has_bbox'.split()].rolling(24*3, 1, center = True).max()
        ggout = pd.concat([gg1,gg2], axis=1)['x0 y0 x1 y1 has_bbox'.split()].values
        self.bbadf['_x0 _y0 _x1 _y1 _has_bbox'.split()] = ggout
        
        # Drop the low probas and roll forward
        idx = (self.bbadf['_has_bbox']<0.5).values
        self.bbadf.loc[idx, '_x0 _y0 _x1 _y1'.split()] = np.nan
        gg = self.bbadf.groupby('StudyInstanceUID', as_index=False, group_keys=False)
        ggfill = \
        pd.concat([ \
            pd.concat([gg['_x0 _y0'.split()].ffill(), \
                       gg['_x0 _y0'.split()].bfill()]).min(level=0), \
            pd.concat([gg['_x1 _y1'.split()].ffill(), \
                       gg['_x1 _y1'.split()].bfill()]).max(level=0)], axis=1)
        self.bbadf.loc[idx, '_x0 _y0 _x1 _y1'.split()] = ggfill.loc[idx].values
        
        self.bbadf['_x0 _y0 _x1 _y1'.split()] = self.bbadf['_x0 _y0 _x1 _y1'.split()] * np.array([*cfg.imagesize]*2)
        # self.bbadf['_x0 _y0 _x1 _y1'.split()] = self.bbadf['_x0 _y0 _x1 _y1'.split()].round().astype(int)
        
        # Ffill & bfill missing slices
        self.bbadf = self.bbadf['StudyInstanceUID slice_numbers _x0 _y0 _x1 _y1'.split()]
        idxdf = self.df['StudyInstanceUID slice_number'.split()]
        idxdf.columns = 'StudyInstanceUID slice_numbers'.split()
        self.bbadf = pd.merge(idxdf, self.bbadf, on = 'StudyInstanceUID slice_numbers'.split(), how='outer')
        self.bbadf = self.bbadf.sort_values(['StudyInstanceUID', 'slice_numbers'])
        self.bbadf['_x0 _y0 _x1 _y1'.split()] = \
            self.bbadf.groupby('StudyInstanceUID', as_index=False, group_keys=False)\
                    ['_x0 _y0 _x1 _y1'.split()].apply(lambda x: x.ffill().bfill()).values
        
        # Finally replace and add some buffer
        self.bbadf = self.bbadf.set_index('StudyInstanceUID')
        self.bbadf['_x0 _y0'.split()] = (self.bbadf['_x0 _y0'.split()] - 25).clip(0, 9999)
        self.bbadf['_x1 _y1'.split()] = (self.bbadf['_x1 _y1'.split()] + 25).clip(0, cfg.imagesize[0])
        self.bbadf['_x0 _y0 _x1 _y1'.split()] = self.bbadf['_x0 _y0 _x1 _y1'.split()].round().astype(int)

        
        '''
        self.bbadf.filter(like='x1')[:5000].plot(figsize = (30,5))
        self.bbadf.filter(like='y1')[:5000].plot(figsize = (30,5))
        self.bbadf.filter(like='x0')[:5000].plot(figsize = (30,5))
        self.bbadf.filter(like='y0')[:5000].plot(figsize = (30,5))
        '''
        
        
        if mode=='train':
            self.transform = create_train_transforms(cfg)
            #self.transform = create_valid_transforms(cfg)
        else:
            if self.cfg.norm_on_cuda:
                self.transform = create_valid_transforms_no_norm(cfg)
            else:
                self.transform = create_valid_transforms(cfg)

        self.df = self.df.reset_index(drop = True)
        self.df['slice_number_start'] = \
            self.df.groupby('StudyInstanceUID')['slice_number'].transform(max) - (cfg.sequence_len+1)
            
        self.norm_mean = torch.tensor(self.cfg.norm_mean)
        self.norm_std = torch.tensor(self.cfg.norm_std)
        
        self.df = self.df.set_index('StudyInstanceUID')
        self.ids = self.df.index.unique().tolist()
        idx = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
    
        idd = self.ids[idx]
        dfseq = self.df.loc[idd]
        bb_s = self.bbdf.loc[idd]
        bb_as = self.bbadf.loc[idd]
        image_folder = dfseq['image_folder'].iloc[0]
        
        # Narrow in where we have +ve bboxes
        slnums = dfseq.slice_number.values
        slpos_from = np.where(slnums == bb_s.slnum_from)[0] - 2
        slpos_to   = np.where(slnums == bb_s.slnum_to)[0] + 2
        slpos_from = max((0, slpos_from.item()))
        slpos_to = min((len(slnums), slpos_to.item()))
        # randomly shift channels in same image
        slpos_to += np.random.randint(3)
        dfseq = dfseq.iloc[slpos_from:slpos_to]
        
        
        label = self.metadf.loc[idd, ['patient_overall'] + self.cfg.target].values
        
        dfseq = dfseq.sort_values('slice_number').reset_index().set_index('slice_number')
        dfseq = dfseq.iloc[np.random.randint(self.cfg.dicom_steps)::self.cfg.dicom_steps]
        dfseq = dfseq.reset_index().set_index('StudyInstanceUID')
        # We will be grouping images in 3 channels per image
        dfseq = dfseq.iloc[:3*(len(dfseq)//3)]
        
        # Cropped bounding box
        bb_as = bb_as.set_index('slice_numbers')['_x0 _y0 _x1 _y1'.split()]
        bb_as = bb_as.loc[dfseq.slice_number.values]
        
        '''
        # Square the bbox
        bb = bb_s['x0 y0 x1 y1'.split()].values
        w,h = bb[2]-bb[0], bb[3]-bb[1]
        w_pad, h_pad = h - min(h,w), w - min(h,w)
        mat_pad = np.array([-w_pad//2, -h_pad//2, w_pad//2, h_pad//2])
        x0,y0,x1,y1 = bb = (mat_pad + bb).clip(0, 512)
        '''
        
        sumls = []
        imgls = []
        zposarr = []
        img_cropls = []
        bbls = []
        cropping_coords = []
        for idd,slcnum in dfseq.slice_number.items():
            dcmfile = f'{image_folder}/Image-{int(slcnum)}.dcm'
            X_orig, zpos = self.read_dicom(dcmfile)
            sumls.append(X_orig.sum())
            #X_crop = X_orig[y0:y1,x0:x1]
            imgls.append(X_orig)
            zposarr.append(zpos)
            bbls.append(bb_as.loc[slcnum].values)
            if len(imgls) == 3:
                x0,y0,x1,y1 = \
                 np.concatenate([np.stack(bbls)[:,:2].min(0), 
                                 np.stack(bbls)[:,2:].max(0)])
                img_crops = np.stack(imgls)
                img_crops = torch.from_numpy(img_crops)
                img_crops = img_crops[:, y0:y1,x0:x1]
                img_crops = F.interpolate(img_crops.unsqueeze(0), 
                                              size = self.cfg.imagesize, 
                                              mode=self.cfg.interpolation_mode).squeeze(0)
                img_crops = [self.windowsfn(m.numpy(), self.windows) for m in img_crops]
                img_cropls += img_crops
                imgls = []
                cropping_coords.append([x0,y0,x1,y1])
                

        
        imgls = [cv2.merge(img_cropls[i:i + 3]) for i in range(0, len(img_cropls), 3)]
        # print(len(dfseq), len(dfseq)//3, len(imgls))
        del img_crops, img_cropls
        
        aug_input = dict((f'image{t}', i) for t,i in enumerate(imgls))
        aug_input['image'] = imgls[0]
        aug_output = self.transform(**aug_input)
        zposarr = np.array(zposarr).astype(float)
        zdirn = (zposarr[1:] - zposarr[:-1]).mean()
        
        out = {'image': torch.stack([aug_output[f'image{t}'] for t,i in enumerate(imgls)])}
        out['label'] = torch.tensor(label)
        out['StudyUID'] = torch.tensor(int(idd.split('.')[-1]))
        
        # flip if they are going in the opposite direction
        if zdirn>0:
            out['image'] = torch.flip(out['image'], [0])
        
        out['mask'] = torch.ones(len(out['image']))
        
        out['transformations'] = {
            'cropping_coords': cropping_coords,
            'slice_numbers': dfseq.slice_number.values.tolist(),
            'z_positions': zposarr.tolist(),
            'image_shapes': [img.shape for img in imgls]
        }
        
        return out
    
    def read_dicom(self, dcmfile):
        if 'placeholder' in dcmfile:
            return np.zeros((512,512))-1000
        D = pydicom.dcmread(dcmfile)
        zpos = D.ImagePositionPatient[-1]
        # D.PhotometricInterpretation = 'YBR_FULL'
        m = float(D.RescaleSlope)
        b = float(D.RescaleIntercept)
        D = D.pixel_array.astype('float')*m
        D = D.astype('float')+b
        return D, zpos

    def list_dcm_dir(self, dcmfile):
        dcmfiles = sorted(glob.glob(f'{cfg.data_folder}/{idd}/*'))
        dcmnums = [int(i.split('/')[-1].split('.')[0]) for i in dcmfiles]
        dcmfiles = [x for _, x in sorted(zip(dcmnums, dcmfiles))]
        return dcmfiles
    
    def windowfn(self, img, WL=400, WW=1800):
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X
    
    def windowsfn(self, X, windows):
        return cv2.merge([self.windowfn(X, *w) for t, w in enumerate(windows)])

    def norm4d(self, x, max_pixel_value = 255.0):
        x1 = x.permute(0,2,3,1)
        if self.mode!='test':
            x1 = x1.float()
        else:
            x1 = x1.half()
        x1 -= self.norm_mean * max_pixel_value
        x1 *= torch.reciprocal(self.norm_std * max_pixel_value)
        x1 = x1.permute(0,3,1,2)
        return x1
