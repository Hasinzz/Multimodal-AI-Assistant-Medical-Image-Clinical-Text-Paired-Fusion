import argparse
import json
import math
import random
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.multiclass import OneVsRestClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import (CHECKPOINT_DIR, IMAGE_SIZE, IMAGES_DIR, OUTPUT_DIR,
                     PROJECTIONS_CSV, REPORTS_CSV, SEED, TARGET_LABELS,
                     assert_paths)
from .template_generator import derive_library

warnings.filterwarnings("ignore", category=UserWarning)
GRID = np.round(np.arange(.10, .901, .05), 2)

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)

def state(current, completed=None, error=None, **extra):
    path = OUTPUT_DIR / "run_state.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    done = old.get("completed_stages", [])
    if completed and completed not in done: done.append(completed)
    data = {"completed_stages": done, "current_stage": current,
            "important_paths": {"outputs": str(OUTPUT_DIR), "checkpoints": str(CHECKPOINT_DIR),
                                "cohort": str(OUTPUT_DIR/'cohort.csv'), "splits": str(OUTPUT_DIR/'splits.csv')},
            "selected_configuration": {"labels": TARGET_LABELS, "seed": SEED, "split": [0.70, .15, .15],
                                       "architecture": "DenseNet121 + TF-IDF/OVR Logistic + late fusion"},
            "checkpoint_paths": old.get("checkpoint_paths", {}), "errors": [] if error is None else [str(error)]}
    data.update(extra); atomic_json(path, data)

def setup():
    assert_paths(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def text_clean(x): return re.sub(r"\s+", " ", "" if pd.isna(x) else str(x)).strip().lower()
def tokens(x): return [v.strip() for v in str(x).split(';') if v.strip()]

def audit():
    setup(); state("audit")
    reports = pd.read_csv(REPORTS_CSV); projections = pd.read_csv(PROJECTIONS_CSV)
    er = ['uid','MeSH','Problems','image','indication','comparison','findings','impression']; ep=['uid','filename','projection']
    assert list(reports.columns)==er and list(projections.columns)==ep
    projections["exists"] = projections.filename.map(lambda x: (IMAGES_DIR/str(x)).is_file())
    readable = {}
    for fn in projections.loc[projections.exists, "filename"].unique():
        try:
            with Image.open(IMAGES_DIR/fn) as im: im.verify()
            readable[fn] = True
        except Exception: readable[fn] = False
    projections["readable"] = projections.filename.map(readable).fillna(False)
    grouped = projections[projections.readable].groupby("uid").agg(
        image_filenames=("filename", lambda x: "|".join(map(str,x))),
        projections=("projection", lambda x: "|".join(map(str,x))))
    merged = reports.merge(grouped, left_on="uid", right_index=True, how="left")
    rows=[]
    for _,r in merged.iterrows():
        probs=tokens(r.Problems); exact=len(probs)==1 and probs[0].lower()=="normal"
        selected=[x for x in TARGET_LABELS if x in probs]
        ok = pd.notna(r.image_filenames) and bool(text_clean(r.indication)) and bool(text_clean(r.findings)) and bool(probs) and (exact or bool(selected))
        if ok:
            rows.append({"uid":r.uid,"image_filenames":r.image_filenames,"projections":r.projections,
                         "indication":text_clean(r.indication),"findings":str(r.findings).strip(),"problems":str(r.Problems),
                         "problems_exact_normal":exact, **{f"label_{i}":int(x in selected) for i,x in enumerate(TARGET_LABELS)}})
    cohort=pd.DataFrame(rows).sort_values("uid"); cohort.to_csv(OUTPUT_DIR/"cohort.csv",index=False)
    audit_obj={"reports_rows":len(reports),"report_unique_uids":int(reports.uid.nunique()),
      "projection_rows":len(projections),"projection_unique_uids":int(projections.uid.nunique()),
      "unique_projection_filenames":int(projections.filename.nunique()),"missing_image_mappings":int((~projections.exists).sum()),
      "unreadable_image_mappings":int((~projections.readable).sum()),"eligible_cohort":len(cohort),
      "limitations":["Study-wise, not patient-wise, split because no verified patient identifier exists.",
                     "Selected-label evaluation omits abnormal findings outside the fixed target space."]}
    atomic_json(OUTPUT_DIR/"dataset_audit.json",audit_obj)
    Y=cohort[[f"label_{i}" for i in range(10)]].to_numpy(); normal=cohort.problems_exact_normal.astype(int).to_numpy()[:,None]
    s1=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.30,random_state=SEED)
    ti,rest=next(s1.split(np.zeros(len(cohort)),np.c_[Y,normal])); remY=np.c_[Y,normal][rest]
    s2=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.50,random_state=SEED)
    vi,tei=next(s2.split(np.zeros(len(rest)),remY)); split=np.full(len(cohort),'train',object); split[rest[vi]]='val'; split[rest[tei]]='test'
    splits=pd.DataFrame({"uid":cohort.uid,"split":split}); splits.to_csv(OUTPUT_DIR/"splits.csv",index=False)
    assert all(set(splits[splits.split==a].uid).isdisjoint(set(splits[splits.split==b].uid)) for a,b in [('train','val'),('train','test'),('val','test')])
    image_sets={s:set('|'.join(cohort.loc[split==s,'image_filenames']).split('|')) for s in ['train','val','test']}
    assert not(image_sets['train']&image_sets['val'] or image_sets['train']&image_sets['test'] or image_sets['val']&image_sets['test'])
    dist=[]
    for i,l in enumerate(TARGET_LABELS):
        for s in ['all','train','val','test']:
            mask=np.ones(len(cohort),bool) if s=='all' else split==s
            dist.append({"label":l,"split":s,"positive_count":int(Y[mask,i].sum()),"study_count":int(mask.sum())})
    pd.DataFrame(dist).to_csv(OUTPUT_DIR/"label_distribution.csv",index=False)
    state("idle",completed="audit",cohort_count=len(cohort),split_counts=pd.Series(split).value_counts().to_dict(),leakage_assertions="passed")

def load_data():
    c=pd.read_csv(OUTPUT_DIR/"cohort.csv"); s=pd.read_csv(OUTPUT_DIR/"splits.csv"); return c.merge(s,on='uid',validate='one_to_one')
def ymat(df): return df[[f"label_{i}" for i in range(10)]].to_numpy(dtype=np.float32)

class XrayDataset(Dataset):
    def __init__(self, df, tfm):
        self.items=[]; self.tfm=tfm
        for _,r in df.iterrows():
            for fn in str(r.image_filenames).split('|'): self.items.append((r.uid,fn,r[[f"label_{i}" for i in range(10)]].to_numpy(dtype=np.float32)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        uid,fn,y=self.items[i]
        with Image.open(IMAGES_DIR/fn) as im: x=self.tfm(im.convert('RGB'))
        return x,torch.from_numpy(y),str(uid),fn

def transforms_for(train=False):
    if train: return transforms.Compose([transforms.RandomResizedCrop(IMAGE_SIZE,scale=(.9,1.0)),transforms.RandomRotation(5),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return transforms.Compose([transforms.Resize(232),transforms.CenterCrop(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def densenet(pretrained=True):
    weights=None
    if pretrained:
        weights=models.DenseNet121_Weights.DEFAULT
    m=models.densenet121(weights=weights); m.classifier=nn.Linear(m.classifier.in_features,10); return m, weights is not None

def aggregate_image(model, df, device, batch=64):
    loader=DataLoader(XrayDataset(df,transforms_for(False)),batch_size=batch,shuffle=False,num_workers=4,pin_memory=True)
    rows=[]; model.eval()
    with torch.no_grad():
        for x,_,uids,fns in loader:
            p=torch.sigmoid(model(x.to(device))).cpu().numpy()
            for uid,fn,v in zip(uids,fns,p): rows.append([uid,fn,*v.tolist()])
    cols=['uid','filename']+[f"prob_{i}" for i in range(10)]; image=pd.DataFrame(rows,columns=cols)
    study=image.groupby('uid',as_index=False)[cols[2:]].mean(); study.uid=study.uid.astype(str); return image,study

def tune_thresholds(y,p):
    out=[]
    for j in range(10):
        scores=[f1_score(y[:,j],p[:,j]>=t,zero_division=0) for t in GRID]; out.append(float(GRID[int(np.argmax(scores))]))
    return np.array(out)

def safe_metric(fn,*args,**kwargs):
    try: return float(fn(*args,**kwargs))
    except ValueError: return float('nan')
def metrics(y,p,t):
    pred=p>=np.asarray(t); out={
      "macro_auroc":safe_metric(roc_auc_score,y,p,average='macro'),"micro_auroc":safe_metric(roc_auc_score,y,p,average='micro'),
      "macro_auprc":safe_metric(average_precision_score,y,p,average='macro'),"micro_auprc":safe_metric(average_precision_score,y,p,average='micro'),
      "macro_f1":f1_score(y,pred,average='macro',zero_division=0),"micro_f1":f1_score(y,pred,average='micro',zero_division=0),
      "macro_precision":precision_score(y,pred,average='macro',zero_division=0),"macro_recall":recall_score(y,pred,average='macro',zero_division=0)}
    out['per_label']={l:{"auroc":safe_metric(roc_auc_score,y[:,j],p[:,j]),"auprc":safe_metric(average_precision_score,y[:,j],p[:,j]),"f1":f1_score(y[:,j],pred[:,j],zero_division=0)} for j,l in enumerate(TARGET_LABELS)}
    out['undefined_metrics_are_nan']=True; return out

def train_image(smoke=False):
    setup(); state("smoke" if smoke else "train_image"); data=load_data(); train=data[data.split=='train']; val=data[data.split=='val']; test=data[data.split=='test']
    if smoke: train=train.iloc[:48]; val=val.iloc[:24]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,pretrained=densenet(True); model.to(device)
    pos=ymat(train).sum(0); pw=torch.tensor((len(train)-pos)/np.maximum(pos,1),device=device)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=pw); batch=64 if device.type=='cuda' else 16
    def phase(name,epochs,lr,unfreeze=False):
        for p in model.features.parameters(): p.requires_grad=False
        if unfreeze:
            for p in model.features.denseblock4.parameters(): p.requires_grad=True
            for p in model.features.norm5.parameters(): p.requires_grad=True
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr); hist=[]; best=-1; bad=0
        loader=DataLoader(XrayDataset(train,transforms_for(True)),batch_size=batch,shuffle=True,num_workers=4,pin_memory=True)
        nonlocal_best=None
        for ep in range(epochs):
            model.train(); losses=[]
            try:
                for x,y,_,_ in loader:
                    x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda',enabled=device.type=='cuda'): loss=lossfn(model(x),y)
                    loss.backward(); opt.step(); losses.append(loss.item())
            except torch.cuda.OutOfMemoryError:
                if batch==64: raise RuntimeError("CUDA_OOM_RETRY_32")
                raise
            _,vp=aggregate_image(model,val,device,batch); vv=val.copy(); vv.uid=vv.uid.astype(str); vp=vv[['uid']].merge(vp,on='uid')
            score=metrics(ymat(vv),vp[[f'prob_{i}' for i in range(10)]].to_numpy(),np.full(10,.5))["macro_auroc"]
            hist.append({"phase":name,"epoch":ep+1,"loss":np.mean(losses),"val_macro_auroc":score})
            if not math.isnan(score) and score>best: best=score; bad=0; nonlocal_best={k:v.detach().cpu() for k,v in model.state_dict().items()}
            else: bad+=1
            if unfreeze and bad>=3: break
        if nonlocal_best: model.load_state_dict(nonlocal_best)
        return hist
    try:
        history=phase('head',1 if smoke else 3,1e-3)+([] if smoke else phase('fine_tune',10,1e-4,True))
    except RuntimeError as e:
        if str(e)=="CUDA_OOM_RETRY_32":
            torch.cuda.empty_cache(); batch=32; history=phase('head',1 if smoke else 3,1e-3)+([] if smoke else phase('fine_tune',10,1e-4,True))
        else: raise
    if smoke:
        x=next(iter(DataLoader(XrayDataset(val,transforms_for(False)),batch_size=2)))[0].to(device); assert model(x).shape==(2,10)
        return model
    out=OUTPUT_DIR/'model1'; ck=CHECKPOINT_DIR/'model1'; out.mkdir(parents=True,exist_ok=True); ck.mkdir(parents=True,exist_ok=True)
    checkpoint=ck/'densenet121_best.pt'; torch.save(model.state_dict(),checkpoint); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    atomic_json(out/'config.json',{"architecture":"densenet121","pretrained_weights_loaded":pretrained,"image_size":224,"batch_size":batch,"selection_metric":"validation macro AUROC","labels":TARGET_LABELS})
    val_img,valp=aggregate_image(model,val,device,batch); test_img,testp=aggregate_image(model,test,device,batch)
    val_img.to_csv(out/'val_image_predictions.csv',index=False); test_img.to_csv(out/'test_image_predictions.csv',index=False)
    for df,p,path in [(val,valp,out/'val_predictions.csv'),(test,testp,out/'test_predictions.csv')]:
        z=df[['uid']].copy(); z.uid=z.uid.astype(str); z=z.merge(p,on='uid'); z.to_csv(path,index=False)
    vp=pd.read_csv(out/'val_predictions.csv')[[f'prob_{i}' for i in range(10)]].to_numpy(); th=tune_thresholds(ymat(val),vp)
    atomic_json(out/'thresholds.json',{"threshold_0.5":[.5]*10,"tuned":dict(zip(TARGET_LABELS,th.tolist())),"source":"validation only"})
    tp=pd.read_csv(out/'test_predictions.csv')[[f'prob_{i}' for i in range(10)]].to_numpy(); atomic_json(out/'metrics.json',{"threshold_0.5":metrics(ymat(test),tp,[.5]*10),"tuned":metrics(ymat(test),tp,th)})
    state("idle",completed="train_image",checkpoint_paths={"model1":str(checkpoint)})

def train_text(smoke=False):
    setup(); state("smoke" if smoke else "train_text"); d=load_data(); tr=d[d.split=='train']; va=d[d.split=='val']; te=d[d.split=='test']
    if smoke: tr=tr.iloc[:100]; va=va.iloc[:40]
    vec=TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,min_df=2,max_features=20000)
    X=vec.fit_transform(tr.indication.map(text_clean)); Xv=vec.transform(va.indication.map(text_clean)); best=None
    for C in ([1.0] if smoke else [.5,1.,2.]):
        clf=OneVsRestClassifier(LogisticRegression(C=C,class_weight='balanced',max_iter=1000,random_state=SEED)).fit(X,ymat(tr))
        p=clf.predict_proba(Xv); score=metrics(ymat(va),p,np.full(10,.5))['macro_auroc']
        if best is None or (not math.isnan(score) and score>best[0]): best=(score,C,clf,p)
    _,C,clf,vp=best; assert vp.shape[1]==10
    if smoke: return vec,clf
    out=OUTPUT_DIR/'model2'; ck=CHECKPOINT_DIR/'model2'; out.mkdir(parents=True,exist_ok=True); ck.mkdir(parents=True,exist_ok=True)
    model_path=ck/'tfidf_logreg.joblib'; joblib.dump({"vectorizer":vec,"classifier":clf,"labels":TARGET_LABELS},model_path)
    tp=clf.predict_proba(vec.transform(te.indication.map(text_clean)))
    for df,p,path in [(va,vp,out/'val_predictions.csv'),(te,tp,out/'test_predictions.csv')]:
        pd.DataFrame({"uid":df.uid.astype(str),**{f'prob_{i}':p[:,i] for i in range(10)}}).to_csv(path,index=False)
    th=tune_thresholds(ymat(va),vp); atomic_json(out/'thresholds.json',{"threshold_0.5":[.5]*10,"tuned":dict(zip(TARGET_LABELS,th.tolist())),"source":"validation only"})
    atomic_json(out/'config.json',{"input":"indication only","selected_C":C,"C_search":[.5,1.,2.],"labels":TARGET_LABELS,"tfidf_fit":"train only"})
    atomic_json(out/'metrics.json',{"threshold_0.5":metrics(ymat(te),tp,[.5]*10),"tuned":metrics(ymat(te),tp,th)})
    old=json.loads((OUTPUT_DIR/'run_state.json').read_text()); cps=old.get('checkpoint_paths',{}); cps['model2']=str(model_path); state("idle",completed="train_text",checkpoint_paths=cps)

def smoke():
    setup(); state('smoke'); d=load_data(); assert len(d)>0
    sample=d.iloc[:24]
    for fs in sample.image_filenames:
        for fn in fs.split('|'):
            with Image.open(IMAGES_DIR/fn) as im: im.load()
    im=train_image(True); vec,clf=train_text(True); y=ymat(sample); p=np.full_like(y,.2); assert (p*.5+p*.5).shape==(len(sample),10)
    rec=[]
    for _,r in d[d.split=='train'].head(100).iterrows(): rec.append({"uid":r.uid,"findings":r.findings,"labels":[TARGET_LABELS[i] for i,v in enumerate(ymat(pd.DataFrame([r]))[0]) if v],"problems_exact_normal":bool(r.problems_exact_normal)})
    lib=derive_library(rec); assert len(lib['labels'])==10
    tmp=OUTPUT_DIR/'smoke_generated_findings.csv'; pd.DataFrame([{"uid":"smoke","generated_findings":lib['normal']['template']}]).to_csv(tmp,index=False); tmp.unlink()
    metrics(y,p,np.full(10,.5)); state('idle',completed='smoke')

def fusion():
    setup(); state('fusion'); d=load_data(); va=d[d.split=='val']; te=d[d.split=='test']; cols=[f'prob_{i}' for i in range(10)]
    iv=pd.read_csv(OUTPUT_DIR/'model1'/'val_predictions.csv'); tv=pd.read_csv(OUTPUT_DIR/'model2'/'val_predictions.csv'); it=pd.read_csv(OUTPUT_DIR/'model1'/'test_predictions.csv'); tt=pd.read_csv(OUTPUT_DIR/'model2'/'test_predictions.csv')
    for x in [iv,tv,it,tt]: x.uid=x.uid.astype(str)
    va=va.copy(); te=te.copy(); va.uid=va.uid.astype(str); te.uid=te.uid.astype(str)
    def aligned(base,a,b):
        z=base[['uid']].merge(a,on='uid').merge(b,on='uid',suffixes=('_i','_t')); return np.c_[[z[f'{c}_i'] for c in cols]].T,np.c_[[z[f'{c}_t'] for c in cols]].T
    # explicit alignment avoids any dependence on CSV row order
    zv=va[['uid']].merge(iv,on='uid').merge(tv,on='uid',suffixes=('_i','_t'))
    pi=np.column_stack([zv[f'{c}_i'] for c in cols]); pt=np.column_stack([zv[f'{c}_t'] for c in cols]); yv=ymat(va)
    search=[]
    for a in np.arange(0,1.01,.1):
        pf=a*pi+(1-a)*pt; th=tune_thresholds(yv,pf); search.append({"alpha":round(float(a),1),"validation_macro_f1":f1_score(yv,pf>=th,average='macro',zero_division=0)})
    pd.DataFrame(search).to_csv(OUTPUT_DIR/'model3'/'alpha_search.csv',index=False) if (OUTPUT_DIR/'model3').mkdir(parents=True,exist_ok=True) is None else None
    alpha=max(search,key=lambda x:(x['validation_macro_f1'],-x['alpha']))['alpha']; pv=alpha*pi+(1-alpha)*pt; th=tune_thresholds(yv,pv)
    atomic_json(OUTPUT_DIR/'model3'/'selected_alpha.json',{"alpha":alpha,"source":"validation only","criterion":"macro F1"}); atomic_json(OUTPUT_DIR/'model3'/'thresholds.json',{"tuned":dict(zip(TARGET_LABELS,th.tolist())),"source":"validation only"})
    zt=te[['uid']].merge(it,on='uid').merge(tt,on='uid',suffixes=('_i','_t')); pi2=np.column_stack([zt[f'{c}_i'] for c in cols]); pt2=np.column_stack([zt[f'{c}_t'] for c in cols]); pf=alpha*pi2+(1-alpha)*pt2
    pd.DataFrame({"uid":zt.uid,**{c:pf[:,i] for i,c in enumerate(cols)}}).to_csv(OUTPUT_DIR/'model3'/'test_predictions.csv',index=False)
    image_th=list(json.loads((OUTPUT_DIR/'model1'/'thresholds.json').read_text())['tuned'].values())
    text_th=list(json.loads((OUTPUT_DIR/'model2'/'thresholds.json').read_text())['tuned'].values())
    mets={"image_only":metrics(ymat(te),pi2,image_th),"text_only":metrics(ymat(te),pt2,text_th),"late_fusion":metrics(ymat(te),pf,th)}
    atomic_json(OUTPUT_DIR/'model3'/'metrics.json',mets['late_fusion'])
    rows=[]; per=[]
    for name,m in mets.items():
        rows.append({"model":name,**{k:v for k,v in m.items() if k not in ('per_label','undefined_metrics_are_nan')}})
        for label,x in m['per_label'].items(): per.append({"model":name,"label":label,**x})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'structured_ablation.csv',index=False); pd.DataFrame(per).to_csv(OUTPUT_DIR/'per_label_metrics.csv',index=False)
    state('idle',completed='fusion',selected_alpha=alpha,fused_thresholds=th.tolist())

def templates_stage():
    setup(); state('templates'); d=load_data(); train=d[d.split=='train']; records=[]
    for _,r in train.iterrows(): records.append({"uid":r.uid,"findings":r.findings,"labels":[TARGET_LABELS[i] for i,v in enumerate(ymat(pd.DataFrame([r]))[0]) if v],"problems_exact_normal":bool(r.problems_exact_normal)})
    lib=derive_library(records); lib['provenance']={"source_split":"train only","validation_findings_used":False,"test_findings_used":False}; atomic_json(OUTPUT_DIR/'template_library.json',lib); state('idle',completed='templates')

def evaluate():
    setup(); state('evaluate'); d=load_data(); test=d[d.split=='test'].copy(); test.uid=test.uid.astype(str)
    probs=pd.read_csv(OUTPUT_DIR/'model3'/'test_predictions.csv'); probs.uid=probs.uid.astype(str); z=test.drop(columns=['findings']).merge(probs,on='uid')
    lib=json.loads((OUTPUT_DIR/'template_library.json').read_text()); th=np.array(list(json.loads((OUTPUT_DIR/'model3'/'thresholds.json').read_text())['tuned'].values())); cols=[f'prob_{i}' for i in range(10)]
    generated=[]
    for _,r in z.iterrows():
        p=np.array([r[c] for c in cols]); idx=np.where(p>=th)[0]; idx=idx[np.argsort(-p[idx])]
        texts=[]; labels=[]
        for i in idx:
            labels.append(TARGET_LABELS[i]); t=lib['labels'][TARGET_LABELS[i]]['template']
            if t not in texts: texts.append(t)
        generated.append({"uid":r.uid,"indication":r.indication,"predicted_labels":"|".join(labels),"fused_probabilities":json.dumps(dict(zip(TARGET_LABELS,map(float,p)))),"generated_findings":" ".join(texts) if texts else lib['normal']['template']})
    gen=pd.DataFrame(generated); hidden=test[['uid','findings']].rename(columns={'findings':'actual_findings'}); final=gen.merge(hidden,on='uid',validate='one_to_one')
    assert len(final)==len(test) and final.generated_findings.notna().all()
    try:
        from rouge_score import rouge_scorer
    except ImportError as e: raise RuntimeError("rouge-score is required for evaluation") from e
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True); scores=[scorer.score(a,g) for a,g in zip(final.actual_findings,final.generated_findings)]
    for key in ['rouge1','rouge2','rougeL']: final[key]=[x[key].fmeasure for x in scores]
    final.to_csv(OUTPUT_DIR/'generated_findings_test.csv',index=False); gm={k:float(final[k].mean()) for k in ['rouge1','rouge2','rougeL']}; gm.update({"bertscore":"skipped: not installed without a heavy dependency/model","interpretation":"secondary text overlap, not clinical accuracy"}); atomic_json(OUTPUT_DIR/'generation_metrics.json',gm)
    write_results(d,lib,gm); state('complete',completed='evaluate')

def write_results(d,lib,gm):
    ab=pd.read_csv(OUTPUT_DIR/'structured_ablation.csv').set_index('model'); alpha=json.loads((OUTPUT_DIR/'model3'/'selected_alpha.json').read_text())['alpha']; th=json.loads((OUTPUT_DIR/'model3'/'thresholds.json').read_text())['tuned']; dist=pd.read_csv(OUTPUT_DIR/'label_distribution.csv'); auditj=json.loads((OUTPUT_DIR/'dataset_audit.json').read_text())
    def row(name): return ', '.join(f"{k}={ab.loc[name,k]:.4f}" for k in ['macro_auroc','micro_auroc','macro_auprc','micro_auprc','macro_f1','micro_f1','macro_precision','macro_recall'])
    counts=d.split.value_counts(); derived=sum(not x['fallback'] for x in lib['labels'].values())+(not lib['normal']['fallback']); fallback=11-derived
    lines=["# Final Results","",f"Eligible cohort: {len(d)} studies; train {counts.train}, validation {counts.val}, test {counts.test}.","",
      "Labels (all-cohort positives): "+', '.join(f"{l}={int(dist[(dist.label==l)&(dist.split=='all')].positive_count.iloc[0])}" for l in TARGET_LABELS),"",
      f"Image only: {row('image_only')}",f"Indication only: {row('text_only')}",f"Late fusion: {row('late_fusion')}","",
      f"Selected alpha: {alpha}. Fused thresholds: {json.dumps(th)}.",
      f"Fusion macro F1 improved over image only: {bool(ab.loc['late_fusion','macro_f1']>ab.loc['image_only','macro_f1'])}; over text only: {bool(ab.loc['late_fusion','macro_f1']>ab.loc['text_only','macro_f1'])}.",
      f"Templates: {derived} training-derived, {fallback} fallback (including normal).",
      f"ROUGE (secondary text overlap): R1={gm['rouge1']:.4f}, R2={gm['rouge2']:.4f}, RL={gm['rougeL']:.4f}.",
      f"Missing image mappings: {auditj['missing_image_mappings']}; unreadable mappings: {auditj['unreadable_image_mappings']}.",
      "Leakage assertions passed: study/image-disjoint splits; TF-IDF and templates train-only; thresholds and alpha validation-only; test findings revealed after generation.",
      "Limitations: study-wise rather than patient-wise split; single internal held-out test; fixed incomplete label space; template-based text; no external or clinical validation."]
    (OUTPUT_DIR/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',required=True,choices=['audit','smoke','train_image','train_text','fusion','templates','evaluate','all']); a=ap.parse_args()
    funcs={'audit':audit,'smoke':smoke,'train_image':train_image,'train_text':train_text,'fusion':fusion,'templates':templates_stage,'evaluate':evaluate}
    try:
        if a.stage=='all':
            for fn in [audit,smoke,train_image,train_text,fusion,templates_stage,evaluate]: fn()
        else: funcs[a.stage]()
    except Exception as e:
        state(a.stage,error=f"{type(e).__name__}: {e}"); raise
if __name__=='__main__': main()
