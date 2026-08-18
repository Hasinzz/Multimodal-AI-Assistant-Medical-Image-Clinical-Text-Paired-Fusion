import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .config import CKPT, LABELS, OUT, PROJECTIONS, SEED, V1_OUT, setup
from .modeling import AsymmetricLoss, ImageBranch, ImageDataset, StudyDataset, YCOL, study_collate
from .phase1 import atomic_json, registry, state

PROB=[f'prob_{i}' for i in range(10)]
CANDIDATES={
 'imnet224_bce_mean':dict(pretraining='imagenet',resolution=224,loss='bce',multiview=False),
 'cxr224_bce_mean':dict(pretraining='cxr_chexpert',resolution=224,loss='bce',multiview=False),
 'cxr224_asl_mean':dict(pretraining='cxr_chexpert',resolution=224,loss='asl',multiview=False),
 'cxr320_bce_mean':dict(pretraining='cxr_chexpert',resolution=320,loss='bce',multiview=False),
 'cxr384_bce_mean':dict(pretraining='cxr_chexpert',resolution=384,loss='bce',multiview=False),
 'cxr224_bce_gated':dict(pretraining='cxr_chexpert',resolution=224,loss='bce',multiview=True),
}

def load_dev():
    cohort=pd.read_csv(V1_OUT/'cohort.csv'); locked=pd.read_csv(OUT/'final_locked_split.csv'); folds=pd.read_csv(OUT/'development_folds.csv'); dev=locked[locked.split=='development'][['uid']].merge(cohort,on='uid',validate='one_to_one').merge(folds,on='uid',validate='one_to_one'); return dev

def score(y,p):
    vals=[]
    for j in range(10):
        try: vals.append(average_precision_score(y[:,j],p[:,j]))
        except ValueError: vals.append(np.nan)
    try: auc=roc_auc_score(y,p,average='macro')
    except ValueError: auc=np.nan
    th=np.array([max(np.arange(.05,.951,.025),key=lambda t:f1_score(y[:,j],p[:,j]>=t,zero_division=0)) for j in range(10)])
    return {'macro_auprc':float(np.nanmean(vals)),'macro_auroc':float(auc),'macro_f1':float(f1_score(y,p>=th,average='macro',zero_division=0))},th

def loaders(train,val,cfg,projection,batch):
    if cfg['multiview']:
        return (DataLoader(StudyDataset(train,cfg['resolution'],True,cfg['pretraining'],projection),batch_size=batch,shuffle=True,num_workers=8,pin_memory=True,collate_fn=study_collate,persistent_workers=True,prefetch_factor=3),
                DataLoader(StudyDataset(val,cfg['resolution'],False,cfg['pretraining'],projection),batch_size=batch,shuffle=False,num_workers=8,pin_memory=True,collate_fn=study_collate,persistent_workers=True,prefetch_factor=3))
    return (DataLoader(ImageDataset(train,cfg['resolution'],True,cfg['pretraining'],projection),batch_size=batch,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3),
            DataLoader(ImageDataset(val,cfg['resolution'],False,cfg['pretraining'],projection),batch_size=batch,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3))

def infer(model,loader,device,multiview):
    rows=[]; model.eval()
    with torch.no_grad():
        for batch in loader:
            if multiview:
                x,mask,views,_,uids,names=batch; logits=model(x.to(device),mask.to(device),views.to(device)); fns=names
            else:
                x,_,uids,fns,_=batch; logits=model(x.to(device))
            p=torch.sigmoid(logits).cpu().numpy()
            for uid,fn,v in zip(uids,fns,p): rows.append([uid,fn,*v])
    image=pd.DataFrame(rows,columns=['uid','filename',*PROB]); study=image.groupby('uid',as_index=False)[PROB].mean(); return image,study

def train_fold(candidate,fold,seed_base=42,tag=None):
    cfg=CANDIDATES[candidate].copy(); tag=tag or candidate; out=OUT/'model1'/tag/f'fold_{fold}'; ck=CKPT/'model1'/tag/f'fold_{fold}.pt'; pred=out/'val_predictions.csv'
    if ck.exists() and pred.exists(): return
    out.mkdir(parents=True,exist_ok=True); ck.parent.mkdir(parents=True,exist_ok=True); dev=load_dev(); train=dev[dev.fold!=fold]; val=dev[dev.fold==fold]; projection=pd.read_csv(PROJECTIONS).set_index('filename').projection.to_dict()
    seed=seed_base+fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); device=torch.device('cuda'); model=ImageBranch(cfg['pretraining'],cfg['multiview']).to(device)
    pos=train[YCOL].sum().to_numpy(float); pos_weight=torch.tensor((len(train)-pos)/np.maximum(pos,1),dtype=torch.float32,device=device); lossfn=nn.BCEWithLogitsLoss(pos_weight=pos_weight) if cfg['loss']=='bce' else AsymmetricLoss()
    batch=(32 if cfg['resolution']<=224 else (20 if cfg['resolution']==320 else 12)) if cfg['multiview'] else (64 if cfg['resolution']<=224 else (32 if cfg['resolution']==320 else 20))
    tl,vl=loaders(train,val,cfg,projection,batch); history=[]; global_best=-1; best_state=None; best_epoch=None; epoch_counter=0
    for phase,max_epochs,lr in [('head',2,1e-3),('late',8,1e-4),('full',8,3e-5)]:
        model.set_phase(phase); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.3,patience=1); bad=0
        for ep in range(max_epochs):
            epoch_counter+=1; model.train(); losses=[]
            for b in tl:
                if cfg['multiview']:
                    x,mask,views,y,_,_=b; x=x.to(device); mask=mask.to(device); views=views.to(device); y=y.to(device)
                else: x,y,*_=b; x=x.to(device); y=y.to(device); mask=views=None
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'): logits=model(x,mask,views) if cfg['multiview'] else model(x); loss=lossfn(logits,y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(loss.item())
            _,vp=infer(model,vl,device,cfg['multiview']); base=val[['uid']].copy(); base.uid=base.uid.astype(str); z=base.merge(vp,on='uid'); met,_=score(val[YCOL].to_numpy(),z[PROB].to_numpy()); scheduler.step(met['macro_auprc']); history.append({'phase':phase,'epoch':epoch_counter,'loss':float(np.mean(losses)),**met,'lr':opt.param_groups[0]['lr']})
            if met['macro_auprc']>global_best+1e-5: global_best=met['macro_auprc']; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; best_epoch=epoch_counter; bad=0
            else: bad+=1
            if bad>=3: break
        if best_state: model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'config':cfg,'candidate':candidate,'fold':fold,'best_epoch':best_epoch,'seed':seed},ck); model.load_state_dict(best_state); image,vp=infer(model,vl,device,cfg['multiview']); image.to_csv(out/'val_image_predictions.csv',index=False); vp.to_csv(pred,index=False); pd.DataFrame(history).to_csv(out/'history.csv',index=False); atomic_json(out/'config.json',{**cfg,'candidate':candidate,'fold':fold,'seed':seed,'best_epoch':best_epoch,'selection':'validation macro AUPRC','pretraining_provenance':pretraining_provenance(cfg)})

def pretraining_provenance(cfg):
    if cfg['pretraining']=='imagenet': return {'source':'torchvision DenseNet121_Weights.DEFAULT','expected_normalization':'ImageNet mean/std','architecture':'DenseNet-121','license':'torchvision upstream model license','native_resolution':224}
    return {'source':'torchxrayvision 1.5.2 densenet121-res224-chex','upstream_training':'CheXpert (not OpenI/IU)','paper':'Cohen et al., arXiv:2002.02497','expected_range':'single-channel approximately [-1024,1024]','architecture':'DenseNet-121','native_resolution':224,'license':'torchxrayvision package/upstream weight terms'}

def summarize(candidate,tag=None):
    tag=tag or candidate; dev=load_dev(); frames=[]
    for f in range(5): frames.append(pd.read_csv(OUT/'model1'/tag/f'fold_{f}'/'val_predictions.csv'))
    p=pd.concat(frames,ignore_index=True); p.uid=p.uid.astype(str); base=dev[['uid',*YCOL]].copy(); base.uid=base.uid.astype(str); z=base.merge(p,on='uid',validate='one_to_one'); met,th=score(z[YCOL].to_numpy(),z[PROB].to_numpy()); p.to_csv(OUT/'oof'/f'model1_{tag}.csv',index=False); atomic_json(OUT/'model1'/tag/'oof_metrics.json',{**met,'thresholds':dict(zip(LABELS,th.tolist()))}); return met

def image_oof(tag):
    p=pd.concat([pd.read_csv(OUT/'model1'/tag/f'fold_{f}'/'val_image_predictions.csv') for f in range(5)],ignore_index=True)
    return p.merge(pd.read_csv(PROJECTIONS)[['filename','projection']],on='filename',how='left')

def aggregate_view(p,strategy,weights=None):
    rows=[]
    for uid,g in p.groupby('uid'):
        front=g[g.projection=='Frontal'][PROB].mean().to_numpy(float) if (g.projection=='Frontal').any() else None
        lateral=g[g.projection=='Lateral'][PROB].mean().to_numpy(float) if (g.projection=='Lateral').any() else None
        mean=g[PROB].mean().to_numpy(float)
        if strategy=='mean': out=mean
        elif strategy=='frontal': out=front if front is not None else mean
        else: out=np.array([mean[j] if front is None and lateral is None else (lateral[j] if front is None else (front[j] if lateral is None else weights[j]*front[j]+(1-weights[j])*lateral[j])) for j in range(10)])
        rows.append([str(uid),*out])
    return pd.DataFrame(rows,columns=['uid',*PROB])

def analyze_views(tag):
    dev=load_dev(); base=dev[['uid',*YCOL]].copy(); base.uid=base.uid.astype(str); images=image_oof(tag); results={}; predictions={}
    for strategy in ['mean','frontal']:
        p=aggregate_view(images,strategy); z=base.merge(p,on='uid'); results[strategy]=score(z[YCOL].to_numpy(),z[PROB].to_numpy())[0]; predictions[strategy]=p
    paired=[]
    for uid,g in images.groupby('uid'):
        f=g[g.projection=='Frontal'][PROB].mean().to_numpy(float) if (g.projection=='Frontal').any() else g[PROB].mean().to_numpy(float)
        l=g[g.projection=='Lateral'][PROB].mean().to_numpy(float) if (g.projection=='Lateral').any() else f
        paired.append([str(uid),*f,*l])
    q=pd.DataFrame(paired,columns=['uid',*[f'f_{i}' for i in range(10)],*[f'l_{i}' for i in range(10)]]); z=base.merge(q,on='uid'); weights=[]
    for j in range(10): weights.append(max(np.arange(0,1.01,.05),key=lambda w:average_precision_score(z[YCOL[j]],w*z[f'f_{j}']+(1-w)*z[f'l_{j}'])))
    p=aggregate_view(images,'view_weighted',weights); z=base.merge(p,on='uid'); results['view_weighted']=score(z[YCOL].to_numpy(),z[PROB].to_numpy())[0]; predictions['view_weighted']=p
    for k,p in predictions.items(): p.to_csv(OUT/'oof'/f'model1_{tag}_{k}.csv',index=False)
    atomic_json(OUT/'model1'/tag/'view_strategy_metrics.json',results); atomic_json(OUT/'model1'/tag/'view_weights.json',dict(zip(LABELS,map(float,weights)))); return results,weights

def run_candidate(candidate,seed_base=42,tag=None):
    setup(); state(f'phase2_{tag or candidate}')
    for f in range(5): train_fold(candidate,f,seed_base,tag)
    met=summarize(candidate,tag); cfg=CANDIDATES[candidate]
    registry({'experiment_id':tag or candidate,'date/time':datetime.now(timezone.utc).isoformat(),'model':'DenseNet-121','pretraining':cfg['pretraining'],'image resolution':cfg['resolution'],'view strategy':'feature gated' if cfg['multiview'] else 'probability mean','loss':cfg['loss'],'optimizer':'AdamW','learning rate':'1e-3/1e-4/3e-5','scheduler':'ReduceLROnPlateau','epochs':'2+up to 8+up to 8 per fold','augmentation':'random resized crop .88-1.0; rotation 7; no flip','threshold method':'OOF per-label F1 grid .05:.025:.95','validation metrics':json.dumps(met),'checkpoint path':str(CKPT/'model1'/(tag or candidate)),'status':'completed','notes':'5-fold iterative study-wise DEVELOPMENT OOF; final holdout untouched'})
    state('phase2_running',completed=tag or candidate)

def select_and_seed():
    rows=[]
    for c in CANDIDATES:
        m=json.loads((OUT/'model1'/c/'oof_metrics.json').read_text()); rows.append({'candidate':c,**{k:m[k] for k in ['macro_auprc','macro_auroc','macro_f1']}})
    rank=pd.DataFrame(rows).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False); rank.to_csv(OUT/'model1'/'candidate_ranking.csv',index=False); best=rank.iloc[0].candidate
    view_strategy='gated' if CANDIDATES[best]['multiview'] else 'mean'; view_weights=None; single=json.loads((OUT/'model1'/best/'oof_metrics.json').read_text())
    if not CANDIDATES[best]['multiview']:
        vm,view_weights=analyze_views(best); view_strategy=max(vm,key=lambda k:(vm[k]['macro_auprc'],vm[k]['macro_auroc'],vm[k]['macro_f1'])); single=vm[view_strategy]
    for seed in [123,2026]: run_candidate(best,seed,tag=f'{best}_seed{seed}')
    frames=[]
    for tag in [best,f'{best}_seed123',f'{best}_seed2026']:
        p=(pd.read_csv(OUT/'oof'/f'model1_{tag}.csv') if view_strategy=='gated' else aggregate_view(image_oof(tag),view_strategy,view_weights)); p.uid=p.uid.astype(str); frames.append(p.set_index('uid')[PROB])
    ens=sum(frames)/3; ens.reset_index().to_csv(OUT/'oof'/'model1_best_ensemble.csv',index=False); dev=load_dev(); base=dev[['uid',*YCOL]].copy(); base.uid=base.uid.astype(str); z=base.merge(ens.reset_index(),on='uid'); em,th=score(z[YCOL].to_numpy(),z[PROB].to_numpy())
    use_ensemble=em['macro_auprc']>=single['macro_auprc']; chosen=f'{best}_ensemble3' if use_ensemble else best
    atomic_json(OUT/'model1'/'best_config.json',{'base_candidate':best,'final_choice':chosen,'ensemble_used':use_ensemble,'view_strategy':view_strategy,'view_weights':dict(zip(LABELS,map(float,view_weights))) if view_weights is not None else None,'single_oof_metrics':single,'ensemble_oof_metrics':em,'ensemble_thresholds':dict(zip(LABELS,th.tolist())),'selection_priority':['macro AUPRC','macro AUROC','macro F1'],'candidate_config':CANDIDATES[best]})
    state('phase2_complete',completed='model1_selection',best_image=chosen)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate',choices=list(CANDIDATES)+['all']); ap.add_argument('--select',action='store_true'); a=ap.parse_args()
    if a.select: select_and_seed()
    elif a.candidate=='all':
        for c in CANDIDATES: run_candidate(c)
        select_and_seed()
    else: run_candidate(a.candidate)
if __name__=='__main__': main()
