import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from .config import IMAGES, LABELS, OUT

YCOL=[f"label_{i}" for i in range(10)]
_CACHE=None
_CACHE_INDEX=None

def cached_image(filename):
    global _CACHE,_CACHE_INDEX
    cache=OUT/'cache'/'images_448.npy'; index=OUT/'cache'/'images_448_index.json'
    if cache.exists() and index.exists():
        if _CACHE is None:
            import json
            _CACHE=np.load(cache,mmap_mode='r'); _CACHE_INDEX=json.loads(index.read_text())
        return Image.fromarray(np.asarray(_CACHE[_CACHE_INDEX[filename]]))
    return Image.open(IMAGES/filename).convert('L')

class SquareCenterCrop:
    def __call__(self,img):
        w,h=img.size; s=min(w,h); return img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))

class XRVNormalize:
    def __call__(self,x): return (x*2.0-1.0)*1024.0

def image_transform(resolution,train,pretraining):
    geom=[transforms.RandomResizedCrop(resolution,scale=(.88,1.0)),transforms.RandomRotation(7)] if train else [SquareCenterCrop(),transforms.Resize((resolution,resolution))]
    if pretraining=='imagenet':
        return transforms.Compose(geom+[transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return transforms.Compose(geom+[transforms.Grayscale(1),transforms.ToTensor(),XRVNormalize()])

class ImageDataset(Dataset):
    def __init__(self,df,resolution,train,pretraining,projection_map):
        self.items=[]; self.t=image_transform(resolution,train,pretraining)
        for _,r in df.iterrows():
            y=r[YCOL].to_numpy(np.float32)
            for fn in str(r.image_filenames).split('|'): self.items.append((str(r.uid),fn,projection_map.get(fn,'Unknown'),y))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        uid,fn,view,y=self.items[i]
        im=cached_image(fn); x=self.t(im); im.close()
        return x,torch.from_numpy(y),uid,fn,view

class StudyDataset(Dataset):
    def __init__(self,df,resolution,train,pretraining,projection_map):
        self.rows=list(df.to_dict('records')); self.t=image_transform(resolution,train,pretraining); self.projection_map=projection_map
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; xs=[]; views=[]; fns=str(r['image_filenames']).split('|')
        for fn in fns:
            im=cached_image(fn); xs.append(self.t(im)); im.close()
            views.append(0 if self.projection_map.get(fn)=='Frontal' else (1 if self.projection_map.get(fn)=='Lateral' else 2))
        return xs,torch.tensor(views),torch.tensor([r[c] for c in YCOL],dtype=torch.float32),str(r['uid']),fns

def study_collate(batch):
    b=len(batch); vmax=max(len(x[0]) for x in batch); shape=batch[0][0][0].shape; images=torch.zeros((b,vmax,*shape)); mask=torch.zeros((b,vmax),dtype=torch.bool); views=torch.full((b,vmax),2,dtype=torch.long)
    ys=[]; uids=[]; names=[]
    for i,(xs,vs,y,uid,fns) in enumerate(batch):
        n=len(xs); images[i,:n]=torch.stack(xs); mask[i,:n]=True; views[i,:n]=vs; ys.append(y); uids.append(uid); names.append('|'.join(fns))
    return images,mask,views,torch.stack(ys),uids,names

class ImageBranch(nn.Module):
    def __init__(self,pretraining='imagenet',multiview=False):
        super().__init__(); self.pretraining=pretraining; self.multiview=multiview
        if pretraining=='imagenet':
            self.net=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT); n=1024
        elif pretraining=='cxr_chexpert':
            import torchxrayvision as xrv
            self.net=xrv.models.DenseNet(weights='densenet121-res224-chex'); n=self.net.classifier.in_features
        else: raise ValueError(pretraining)
        self.net.classifier=nn.Identity(); self.classifier=nn.Linear(n,len(LABELS))
        if multiview: self.view_embedding=nn.Embedding(3,8); self.gate=nn.Linear(n+8,1)
    def features(self,x):
        f=self.net.features(x); f=torch.relu(f); return torch.nn.functional.adaptive_avg_pool2d(f,(1,1)).flatten(1)
    def forward(self,x,mask=None,views=None):
        if not self.multiview: return self.classifier(self.features(x))
        b,v=x.shape[:2]; f=self.features(x.flatten(0,1)).reshape(b,v,-1); score=self.gate(torch.cat([f,self.view_embedding(views)],-1)).squeeze(-1); score=score.masked_fill(~mask,-1e4); weights=torch.softmax(score,1); pooled=(f*weights.unsqueeze(-1)).sum(1); return self.classifier(pooled)
    def set_phase(self,phase):
        for p in self.parameters(): p.requires_grad=False
        for p in self.classifier.parameters(): p.requires_grad=True
        if self.multiview:
            for module in (self.view_embedding,self.gate):
                for p in module.parameters(): p.requires_grad=True
        if phase in ('late','full'):
            for p in self.net.features.denseblock4.parameters(): p.requires_grad=True
            for p in self.net.features.norm5.parameters(): p.requires_grad=True
        if phase=='full':
            for p in self.net.features.parameters(): p.requires_grad=True

class AsymmetricLoss(nn.Module):
    def __init__(self,gamma_neg=4,gamma_pos=0,clip=.05): super().__init__(); self.gn=gamma_neg; self.gp=gamma_pos; self.clip=clip
    def forward(self,logits,y):
        p=torch.sigmoid(logits); pn=(1-p+self.clip).clamp(max=1); loss=y*torch.log(p.clamp(min=1e-8))*((1-p)**self.gp)+(1-y)*torch.log(pn.clamp(min=1e-8))*((1-pn)**self.gn); return -loss.mean()
