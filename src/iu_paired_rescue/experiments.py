import argparse
import json
import random
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision import models,transforms

from .audit import dev_data,state
from .config import CKPT,IMPROVED,LABELS,OUT,PROB,PROJECTIONS,V1,YCOL,setup
from src.iu_paired_improved.phase1 import atomic_json,full_metrics

CONFIGS={
 'P0_original':{'mode':'original'},'P1_clahe':{'mode':'clahe'},'P2_lung_roi':{'mode':'roi'},
 'P0_balanced':{'mode':'original','balanced':True},'P0_mixup':{'mode':'original','mixup':True},
 'P0_hardneg':{'mode':'original','hardneg':True},'findings_aligned_labels':{'mode':'original','aligned':True},
 'P2_roi_balanced':{'mode':'roi','balanced':True},'P2_roi_mixup':{'mode':'roi','mixup':True},'P2_roi_hardneg':{'mode':'roi','hardneg':True},
}

_ARRAYS={}
def array_for(mode):
    if mode not in _ARRAYS:
        path=IMPROVED/'cache'/'images_448.npy' if mode=='original' else OUT/'cache'/f"images_448_{'clahe' if mode=='clahe' else 'lung_roi'}.npy"; _ARRAYS[mode]=np.load(path,mmap_mode='r')
    return _ARRAYS[mode]
def index():
    return json.loads((IMPROVED/'cache'/'images_448_index.json').read_text())

class DS(Dataset):
    def __init__(self,df,mode,train):
        self.mode=mode; self.idx=index(); self.items=[]; geom=[transforms.RandomResizedCrop(224,scale=(.88,1)),transforms.RandomRotation(7)] if train else [transforms.Resize((224,224))]; self.t=transforms.Compose(geom+[transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        for _,r in df.iterrows():
            for fn in str(r.image_filenames).split('|'): self.items.append((str(r.uid),fn,r[YCOL].to_numpy(np.float32)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        uid,fn,y=self.items[i]; im=Image.fromarray(np.asarray(array_for(self.mode)[self.idx[fn]])); x=self.t(im); return x,torch.from_numpy(y),uid,fn

def model():
    m=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT); m.classifier=nn.Linear(m.classifier.in_features,10); return m
def set_phase(m,phase):
    for p in m.features.parameters(): p.requires_grad=False
    if phase in ('late','full'):
        for p in m.features.denseblock4.parameters(): p.requires_grad=True
        for p in m.features.norm5.parameters(): p.requires_grad=True
    if phase=='full':
        for p in m.features.parameters(): p.requires_grad=True

def metrics(y,p):
    th=np.array([max(np.arange(.01,1,.01),key=lambda t:f1_score(y[:,j],p[:,j]>=t,zero_division=0)) for j in range(10)]); m=full_metrics(y,p,th); return m,th

def infer(m,loader,device):
    rows=[]; m.eval()
    with torch.no_grad():
        for x,_,uids,fns in loader:
            p=torch.sigmoid(m(x.to(device))).cpu().numpy()
            for uid,fn,v in zip(uids,fns,p): rows.append([uid,fn,*v])
    im=pd.DataFrame(rows,columns=['uid','filename',*PROB]); return im,im.groupby('uid',as_index=False)[PROB].mean()

def training_loader(df,cfg,mode,batch=64,hard_scores=None):
    ds=DS(df,mode,True)
    if cfg.get('balanced'):
        Y=df[YCOL].to_numpy(); freq=Y.sum(0); study=1+np.minimum(2.0,(Y/np.maximum(freq,1)*len(df)/10).sum(1)); wmap=dict(zip(df.uid.astype(str),study)); weights=[wmap[x[0]] for x in ds.items]; sampler=WeightedRandomSampler(weights,len(weights),replacement=True); return DataLoader(ds,batch_size=batch,sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)
    if hard_scores is not None:
        weights=[1+min(2.0,hard_scores.get(x[0],0)*2) for x in ds.items]; sampler=WeightedRandomSampler(weights,len(weights),replacement=True); return DataLoader(ds,batch_size=batch,sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)
    return DataLoader(ds,batch_size=batch,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)

def train_fold(exp,fold,tag=None,seed_base=42):
    cfg=CONFIGS[exp]; tag=tag or exp; out=OUT/'model1'/tag/f'fold_{fold}'; ck=CKPT/'model1'/tag/f'fold_{fold}.pt'
    if (out/'val_predictions.csv').exists() and ck.exists(): return
    out.mkdir(parents=True,exist_ok=True); ck.parent.mkdir(parents=True,exist_ok=True); d=dev_data()
    if cfg.get('aligned'):
        aligned=pd.read_csv(OUT/'label_completeness'/'findings_aligned_labels.csv'); aligned.uid=aligned.uid.astype(str); d.uid=d.uid.astype(str); d=d.drop(columns=YCOL).merge(aligned,on='uid')
    tr=d[d.fold!=fold]; va=d[d.fold==fold]; device=torch.device('cuda'); seed=seed_base+fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); m=model().to(device); pos=tr[YCOL].sum().to_numpy(float); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr)-pos)/np.maximum(pos,1),dtype=torch.float32,device=device)); vl=DataLoader(DS(va,cfg['mode'],False),batch_size=64,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3); tl=training_loader(tr,cfg,cfg['mode']); best=-1; best_state=None; history=[]; epoch=0
    for phase,n,lr in [('head',2,1e-3),('late',8,1e-4),('full',8,3e-5)]:
        set_phase(m,phase); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=lr,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(1,n)); bad=0
        for ep in range(n):
            epoch+=1; m.train(); losses=[]
            for x,y,*_ in tl:
                x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                if cfg.get('mixup'):
                    lam=np.random.beta(.4,.4); perm=torch.randperm(len(x),device=device); xx=lam*x+(1-lam)*x[perm]; yy=lam*y+(1-lam)*y[perm]
                else: xx,yy=x,y
                with torch.amp.autocast('cuda'): loss=lossfn(m(xx),yy)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); losses.append(loss.item())
            _,vp=infer(m,vl,device); base=va[['uid']].copy(); base.uid=base.uid.astype(str); z=base.merge(vp,on='uid'); met,_=metrics(va[YCOL].to_numpy(),z[PROB].to_numpy()); history.append({'phase':phase,'epoch':epoch,'loss':np.mean(losses),'macro_auprc':met['macro_auprc'],'macro_auroc':met['macro_auroc'],'macro_f1':met['macro_f1']})
            if met['macro_auprc']>best: best=met['macro_auprc']; best_state={k:v.detach().cpu() for k,v in m.state_dict().items()}; bad=0
            else: bad+=1
            sched.step()
            if phase=='full' and bad>=3: break
        if best_state: m.load_state_dict(best_state)
    # Short hard-negative-aware stage: weights derived from TRAIN labels/predictions only; acceptance remains validation-only.
    if cfg.get('hardneg'):
        tr_eval=DataLoader(DS(tr,cfg['mode'],False),batch_size=64,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3); _,pred=infer(m,tr_eval,device); b=tr[['uid',*YCOL]].copy(); b.uid=b.uid.astype(str); q=b.merge(pred,on='uid'); ys=q[YCOL].to_numpy(); ps=q[PROB].to_numpy(); hard=((ps>.7)&(ys==0)).sum(1)/10; hs=dict(zip(q.uid,hard)); tl=training_loader(tr,cfg,cfg['mode'],hard_scores=hs); set_phase(m,'late'); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=2e-5,weight_decay=1e-4)
        for ep in range(2):
            epoch+=1; m.train(); losses=[]
            for x,y,*_ in tl:
                x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'): loss=lossfn(m(x),y)
                loss.backward(); opt.step(); losses.append(loss.item())
            _,vp=infer(m,vl,device); base=va[['uid']].copy(); base.uid=base.uid.astype(str); z=base.merge(vp,on='uid'); met,_=metrics(va[YCOL].to_numpy(),z[PROB].to_numpy()); history.append({'phase':'hard_negative','epoch':epoch,'loss':np.mean(losses),'macro_auprc':met['macro_auprc'],'macro_auroc':met['macro_auroc'],'macro_f1':met['macro_f1']})
            if met['macro_auprc']>best: best=met['macro_auprc']; best_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); torch.save({'state_dict':best_state,'config':cfg,'fold':fold},ck); image,vp=infer(m,vl,device); image.to_csv(out/'val_image_predictions.csv',index=False); vp.to_csv(out/'val_predictions.csv',index=False); pd.DataFrame(history).to_csv(out/'history.csv',index=False); atomic_json(out/'config.json',{'experiment':exp,**cfg,'fold':fold,'seed':seed,'selection':'validation macro AUPRC','development_only':True})

def summarize(exp,tag=None):
    tag=tag or exp
    d=dev_data(); p=pd.concat([pd.read_csv(OUT/'model1'/exp/f'fold_{f}'/'val_predictions.csv') for f in range(5)]); p.uid=p.uid.astype(str); base=d[['uid',*YCOL]].copy(); base.uid=base.uid.astype(str); z=base.merge(p,on='uid',validate='one_to_one'); m,th=metrics(z[YCOL].to_numpy(),z[PROB].to_numpy()); p.to_csv(OUT/'oof'/f'{exp}.csv',index=False) if (OUT/'oof').mkdir(parents=True,exist_ok=True) is None else None; atomic_json(OUT/'model1'/exp/'oof_metrics.json',{**m,'thresholds':dict(zip(LABELS,th.tolist()))}); return m

def run(exp,tag=None,seed_base=42):
    tag=tag or exp; setup(); state(f'train_{tag}')
    for f in range(5): train_fold(exp,f,tag,seed_base)
    m=summarize(tag); state('experiments_running',completed=tag); return m

def combined(base,p1,p2,name):
    d=dev_data(); a=pd.read_csv(OUT/'oof'/f'{p1}.csv'); b=pd.read_csv(OUT/'oof'/f'{p2}.csv'); a.uid=a.uid.astype(str); b.uid=b.uid.astype(str); z=a.merge(b,on='uid',suffixes=('_a','_b')); pa=np.column_stack([z[f'{c}_a'] for c in PROB]); pb=np.column_stack([z[f'{c}_b'] for c in PROB]); yb=d[['uid',*YCOL]].copy(); yb.uid=yb.uid.astype(str); y=yb.merge(z[['uid']],on='uid')[YCOL].to_numpy(); grid=[]
    for w in np.arange(0,1.01,.05):
        p=w*pa+(1-w)*pb; grid.append((metrics(y,p)[0]['macro_auprc'],float(w),p))
    _,w,p=max(grid,key=lambda x:x[0]); out=pd.DataFrame({'uid':z.uid,**{PROB[j]:p[:,j] for j in range(10)}}); out.to_csv(OUT/'oof'/f'{name}.csv',index=False); m,th=metrics(y,p); atomic_json(OUT/'model1'/name/'oof_metrics.json',{**m,'thresholds':dict(zip(LABELS,th.tolist())),'weight_first':w,'components':[p1,p2]}) if (OUT/'model1'/name).mkdir(parents=True,exist_ok=True) is None else None; return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',choices=list(CONFIGS)); ap.add_argument('--tag'); ap.add_argument('--seed-base',type=int,default=42); a=ap.parse_args(); run(a.experiment,a.tag,a.seed_base)
if __name__=='__main__': main()
