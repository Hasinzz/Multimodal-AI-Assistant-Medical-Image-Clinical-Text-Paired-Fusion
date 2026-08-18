import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import CKPT, IMAGES, LABELS, OUT, PROJECTIONS, SEED, V1_CKPT, V1_OUT, setup

PROB = [f"prob_{i}" for i in range(10)]
YCOL = [f"label_{i}" for i in range(10)]
REGISTRY_FIELDS = ["experiment_id","date/time","model","pretraining","image resolution","view strategy","loss","optimizer","learning rate","scheduler","epochs","augmentation","text configuration","calibration","fusion method","threshold method","validation metrics","final-test metrics if final-test was legitimately run","checkpoint path","status","notes"]
ALIASES = {
 "Cardiomegaly":["cardiomegaly","cardiac enlargement","enlarged cardiac","enlarged heart"],
 "Pulmonary Atelectasis":["atelectasis","atelectatic"],
 "Calcified Granuloma":["calcified granuloma","calcified granulomata","granuloma"],
 "Cicatrix":["cicatrix","scar","scarring"], "Pleural Effusion":["pleural effusion","effusion"],
 "Atherosclerosis":["atherosclerosis","atherosclerotic","aortic calcification","calcified aorta"],
 "Airspace Disease":["airspace disease","airspace opacity","airspace opacities","airspace consolidation"],
 "Scoliosis":["scoliosis","scoliotic"], "Granulomatous Disease":["granulomatous disease","granulomatous"],
 "Nodule":["nodule","nodular"]}
NEG = re.compile(r"\b(no|not|without|negative for|no evidence of|absence of|free of)\b", re.I)

def atomic_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=True),encoding='utf-8'); tmp.replace(path)

def state(stage, completed=None, error=None, **kw):
    p=OUT/'run_state.json'; old=json.loads(p.read_text()) if p.exists() else {}; done=old.get('completed_stages',[])
    if completed and completed not in done: done.append(completed)
    obj={"completed_stages":done,"current_stage":stage,"locked_split":str(OUT/'final_locked_split.csv'),"development_folds":str(OUT/'development_folds.csv'),"v1_preserved":True,"final_test_evaluations":old.get('final_test_evaluations',0),"errors":[] if error is None else [str(error)]}; obj.update(kw); atomic_json(p,obj)

def registry(row):
    p=OUT/'EXPERIMENT_REGISTRY.csv'; exists=p.exists()
    with p.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=REGISTRY_FIELDS);
        if not exists: w.writeheader()
        w.writerow({k:row.get(k,'') for k in REGISTRY_FIELDS})

def safe(fn,*args,**kwargs):
    try: return float(fn(*args,**kwargs))
    except ValueError: return float('nan')

def markdown_table(df):
    cols=list(df.columns); rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows(): rows.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
    return '\n'.join(rows)

def full_metrics(y,p,thr):
    pred=p>=np.asarray(thr); per=[]
    for j,label in enumerate(LABELS):
        tn,fp,fn,tp=confusion_matrix(y[:,j],pred[:,j],labels=[0,1]).ravel()
        per.append({"label":label,"positives":int(y[:,j].sum()),"auroc":safe(roc_auc_score,y[:,j],p[:,j]),"auprc":safe(average_precision_score,y[:,j],p[:,j]),
          "f1":f1_score(y[:,j],pred[:,j],zero_division=0),"precision":precision_score(y[:,j],pred[:,j],zero_division=0),"recall":recall_score(y[:,j],pred[:,j],zero_division=0),
          "specificity":tn/(tn+fp) if tn+fp else np.nan,"accuracy":(tp+tn)/(tp+tn+fp+fn),"balanced_accuracy":balanced_accuracy_score(y[:,j],pred[:,j]),
          "predicted_positive_rate":float(pred[:,j].mean()),"true_prevalence":float(y[:,j].mean()),"false_positives":int(fp),"false_negatives":int(fn)})
    return {"label_wise_accuracy":float((pred==y).mean()),"exact_match_accuracy":float(np.all(pred==y,axis=1).mean()),
      "macro_balanced_accuracy":float(np.mean([x['balanced_accuracy'] for x in per])),"macro_auroc":safe(roc_auc_score,y,p,average='macro'),
      "micro_auroc":safe(roc_auc_score,y,p,average='micro'),"macro_auprc":safe(average_precision_score,y,p,average='macro'),
      "micro_auprc":safe(average_precision_score,y,p,average='micro'),"macro_f1":f1_score(y,pred,average='macro',zero_division=0),
      "micro_f1":f1_score(y,pred,average='micro',zero_division=0),"macro_precision":precision_score(y,pred,average='macro',zero_division=0),
      "macro_recall":recall_score(y,pred,average='macro',zero_division=0),"per_label":per}

def v1_diagnostic(cohort):
    old=pd.read_csv(V1_OUT/'splits.csv'); test=cohort.merge(old,on='uid'); test=test[test.split=='test'].copy(); test.uid=test.uid.astype(str); y=test[YCOL].to_numpy()
    specs={'V1 image':('model1','test_predictions.csv','thresholds.json'),'V1 text':('model2','test_predictions.csv','thresholds.json'),'V1 global fusion':('model3','test_predictions.csv','thresholds.json')}
    allm={}; per=[]
    for name,(folder,pfile,tfile) in specs.items():
        pr=pd.read_csv(V1_OUT/folder/pfile); pr.uid=pr.uid.astype(str); z=test[['uid']].merge(pr,on='uid'); p=z[PROB].to_numpy()
        tj=json.loads((V1_OUT/folder/tfile).read_text()); th=list(tj['tuned'].values()) if isinstance(tj['tuned'],dict) else tj['tuned']
        m=full_metrics(y,p,th); allm[name]=m
        for x in m['per_label']: per.append({"model":name,**x})
    pd.DataFrame(per).to_csv(OUT/'v1_per_label_diagnostics.csv',index=False); atomic_json(OUT/'v1_diagnostic_metrics.json',allm)
    prevalence=y.mean(0); allneg=float((1-y).mean()); exactneg=float(np.all(y==0,axis=1).mean())
    m=allm['V1 global fusion']; high=m['label_wise_accuracy']>=.95
    lines=["# V1 Diagnostic Report","",f"Evaluated the preserved V1 predictions on the original V1 test ({len(test)} studies).",
      "", "## Central diagnosis","",f"V1 fusion label-wise/Hamming accuracy is {m['label_wise_accuracy']:.4f} ({'>=95%' if high else '<95%'}). The all-negative baseline is {allneg:.4f}; therefore raw accuracy is dominated by negative labels and is not evidence of useful disease detection.",
      f"Exact-match accuracy is {m['exact_match_accuracy']:.4f} versus all-negative exact match {exactneg:.4f}; macro balanced accuracy is {m['macro_balanced_accuracy']:.4f}.",
      f"Meaningful V1 fusion metrics: macro AUROC {m['macro_auroc']:.4f}, macro AUPRC {m['macro_auprc']:.4f}, macro F1 {m['macro_f1']:.4f}, micro F1 {m['micro_f1']:.4f}.",
      "", "## Evidence-based bottlenecks","","- Severe imbalance makes raw accuracy misleading; label prevalence ranges from {:.2%} to {:.2%}.".format(prevalence.min(),prevalence.max()),
      "- The image branch dominates fusion (validation-selected alpha 0.9); indication-only signal is weak and can reduce thresholded macro F1.",
      "- V1 selected the image checkpoint by macro AUROC rather than macro AUPRC/F1 and did not calibrate branch probabilities.",
      "- V1 averaged image probabilities across views without explicit view modeling.",
      "- Rare-label false negatives and unstable thresholds are visible in the per-label table.",
      "", "## Required metric table","",markdown_table(pd.DataFrame([{"model":k,**{q:v[q] for q in ['label_wise_accuracy','exact_match_accuracy','macro_balanced_accuracy','macro_auroc','macro_auprc','macro_f1','micro_f1']}} for k,v in allm.items()])),
      "", "Full per-label AUROC, AUPRC, F1, precision, recall, specificity, accuracy, positive rates, prevalence, FP, and FN are saved in `v1_per_label_diagnostics.csv`. Accuracy is never interpreted without AUROC, AUPRC, and F1."]
    (OUT/'V1_DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def create_locked_split(cohort):
    y=cohort[YCOL].to_numpy(); normal=cohort.problems_exact_normal.astype(int).to_numpy()[:,None]; strat=np.c_[y,normal]
    ss=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=SEED); dev,final=next(ss.split(np.zeros(len(cohort)),strat))
    split=np.full(len(cohort),'development',object); split[final]='final_test'
    locked=pd.DataFrame({'uid':cohort.uid,'split':split}); locked.to_csv(OUT/'final_locked_split.csv',index=False)
    assert set(locked[locked.split=='development'].uid).isdisjoint(set(locked[locked.split=='final_test'].uid))
    image_sets={s:set('|'.join(cohort.loc[split==s,'image_filenames']).split('|')) for s in ['development','final_test']}; assert not image_sets['development']&image_sets['final_test']
    support=[]
    for s in ['development','final_test']:
        mask=split==s
        for j,l in enumerate(LABELS): support.append({'split':s,'label':l,'positive_count':int(y[mask,j].sum()),'studies':int(mask.sum()),'prevalence':float(y[mask,j].mean())})
    pd.DataFrame(support).to_csv(OUT/'locked_split_label_support.csv',index=False)
    devdf=cohort.iloc[dev].reset_index(drop=True); dy=np.c_[devdf[YCOL].to_numpy(),devdf.problems_exact_normal.astype(int).to_numpy()[:,None]]
    kf=MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); folds=np.empty(len(devdf),int)
    for fold,(_,vi) in enumerate(kf.split(np.zeros(len(devdf)),dy)): folds[vi]=fold
    pd.DataFrame({'uid':devdf.uid,'fold':folds}).to_csv(OUT/'development_folds.csv',index=False)
    fs=[]
    for f in range(5):
        for j,l in enumerate(LABELS): fs.append({'fold':f,'label':l,'positive_count':int(devdf.loc[folds==f,YCOL[j]].sum()),'studies':int((folds==f).sum())})
    pd.DataFrame(fs).to_csv(OUT/'development_fold_support.csv',index=False)
    return locked

def mention_state(text,label):
    low=text.lower(); aliases=ALIASES[label]; supports=False; negates=False
    for sent in re.split(r'(?<=[.!?])\s+|[\r\n]+',low):
        for alias in aliases:
            pos=sent.find(alias)
            if pos>=0:
                before=sent[max(0,pos-45):pos]
                if NEG.search(before): negates=True
                else: supports=True
    return supports,negates

def label_audit(cohort,locked):
    out=OUT/'label_audit'; out.mkdir(parents=True,exist_ok=True); dev=cohort.merge(locked,on='uid'); dev=dev[dev.split=='development']
    rows=[]
    for _,r in dev.iterrows():
        for j,l in enumerate(LABELS):
            positive=bool(r[YCOL[j]]); supports,negates=mention_state(str(r.findings),l)
            category='A_structured_positive_findings_support' if positive and supports else ('B_structured_positive_findings_negates' if positive and negates else ('C_structured_negative_findings_support' if not positive and supports else 'D_unclear_or_not_mentioned'))
            rows.append({'uid':r.uid,'label':l,'structured_positive':positive,'category':category,'findings':r.findings})
    detail=pd.DataFrame(rows); summary=detail.groupby(['label','category']).size().unstack(fill_value=0).reset_index()
    for c in ['A_structured_positive_findings_support','B_structured_positive_findings_negates','C_structured_negative_findings_support','D_unclear_or_not_mentioned']:
        if c not in summary: summary[c]=0
    pos=detail.groupby('label').structured_positive.sum(); summary['structured_positives']=summary.label.map(pos).astype(int)
    summary['positive_support_rate']=summary.A_structured_positive_findings_support/summary.structured_positives.replace(0,np.nan)
    summary['positive_negation_rate']=summary.B_structured_positive_findings_negates/summary.structured_positives.replace(0,np.nan)
    summary.to_csv(out/'label_consistency_summary.csv',index=False)
    examples=pd.concat([g.sort_values('uid').head(8) for _,g in detail[detail.category!='D_unclear_or_not_mentioned'].groupby(['label','category'])]); examples.to_csv(out/'label_consistency_examples.csv',index=False)
    severe=summary[(summary.positive_negation_rate>=.10)|(summary.C_structured_negative_findings_support>=summary.structured_positives*.25)].label.tolist()
    lines=["# Label Quality Report","",f"Scope: {len(dev)} DEVELOPMENT studies only. Locked final-test Findings were not inspected.","",
      "A conservative explicit alias dictionary and local pre-mention negation detector were used. This is an audit, not a relabeling procedure; primary ground truth remains the exact Problems tokens.","",
      markdown_table(summary),"",f"Labels flagged by the conservative heuristic for potentially severe inconsistency: {', '.join(severe) if severe else 'none'}.",
      "Structured-positive cases without a supporting mention are grouped as unclear, not automatically treated as errors. Structured-negative/supporting-text cases may reflect Problems omissions, alias ambiguity, or report context. Any cleaned-label experiment must remain separate."]
    (out/'LABEL_QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

class ImageSet(Dataset):
    def __init__(self,df):
        self.items=[]; self.t=transforms.Compose([transforms.Resize(232),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        for _,r in df.iterrows():
            for fn in str(r.image_filenames).split('|'): self.items.append((str(r.uid),fn))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        uid,fn=self.items[i]
        with Image.open(IMAGES/fn) as im: x=self.t(im.convert('RGB'))
        return x,uid,fn

def v1_model():
    m=models.densenet121(weights=None); m.classifier=nn.Linear(m.classifier.in_features,10); m.load_state_dict(torch.load(V1_CKPT/'model1'/'densenet121_best.pt',map_location='cpu',weights_only=True)); return m

def gradcam_audit(cohort,locked):
    figdir=OUT/'figures'/'gradcam_v1'; figdir.mkdir(parents=True,exist_ok=True); dev=cohort.merge(locked,on='uid'); dev=dev[dev.split=='development'].copy(); dev.uid=dev.uid.astype(str)
    device=torch.device('cuda'); model=v1_model().to(device).eval(); loader=DataLoader(ImageSet(dev),batch_size=64,num_workers=4,pin_memory=True); rows=[]
    with torch.no_grad():
        for x,uids,fns in loader:
            p=torch.sigmoid(model(x.to(device))).cpu().numpy()
            for u,fn,v in zip(uids,fns,p): rows.append([u,fn,*v])
    image=pd.DataFrame(rows,columns=['uid','filename',*PROB]); study=image.groupby('uid',as_index=False)[PROB].mean(); study.to_csv(OUT/'oof'/'v1_development_predictions.csv',index=False) if (OUT/'oof').mkdir(parents=True,exist_ok=True) is None else None
    z=dev.merge(study,on='uid'); th=json.loads((V1_OUT/'model1'/'thresholds.json').read_text())['tuned']; proj=pd.read_csv(PROJECTIONS).set_index('filename').projection.to_dict()
    chosen=[]
    for label in ['Cardiomegaly','Pulmonary Atelectasis','Pleural Effusion','Airspace Disease','Nodule']:
        j=LABELS.index(label); pred=z[PROB[j]]>=th[label]; truth=z[YCOL[j]].astype(bool)
        for kind,mask,ascending in [('TP',pred&truth,False),('FP',pred&~truth,False),('FN',~pred&truth,True)]:
            cand=z[mask].sort_values(PROB[j],ascending=ascending)
            if len(cand): chosen.append((label,kind,cand.iloc[0]))
    def make_cam(label,kind,r):
        fns=str(r.image_filenames).split('|'); fn=next((f for f in fns if proj.get(f)=='Frontal'),fns[0]); raw=Image.open(IMAGES/fn).convert('RGB'); t=ImageSet(pd.DataFrame([r])).t(raw).unsqueeze(0).to(device); acts=[]; grads=[]
        def capture(_m,_i,o):
            acts.append(o); o.register_hook(lambda g: grads.append(g))
        h1=model.features.denseblock4.register_forward_hook(capture); model.zero_grad(set_to_none=True); logit=model(t)[0,LABELS.index(label)]; logit.backward(); h1.remove()
        a=acts[0][0]; g=grads[0][0]; cam=torch.relu((g.mean((1,2))[:,None,None]*a).sum(0)); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); cam=torch.nn.functional.interpolate(cam[None,None],size=(raw.height,raw.width),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        arr=np.asarray(raw)/255.; overlay=np.clip(.55*arr+.45*plt.cm.jet(cam)[...,:3],0,1); plt.figure(figsize=(5,5)); plt.imshow(overlay); plt.axis('off'); plt.title(f"{label} {kind} p={float(r[PROB[LABELS.index(label)]]):.3f}"); path=figdir/f"{label.replace(' ','_')}_{kind}_{r.uid}.png"; plt.tight_layout(); plt.savefig(path,dpi=180,bbox_inches='tight'); plt.close(); return {'label':label,'case_type':kind,'uid':r.uid,'filename':fn,'probability':r[PROB[LABELS.index(label)]],'path':str(path)}
    meta=[make_cam(*x) for x in chosen]; pd.DataFrame(meta).to_csv(figdir/'gradcam_cases.csv',index=False)
    report=["# V1 Visual Shortcut Audit","",f"Generated {len(meta)} Grad-CAM examples from DEVELOPMENT studies only: confidence-ranked TP, FP, and FN where available for five priority labels.","",
      "These maps are diagnostic rather than causal localization evidence. The saved examples permit direct inspection of thoracic versus marker/border/corner emphasis. No crop or lung segmentation was introduced automatically.","",
      "Automated spatial concentration alone cannot reliably distinguish anatomy from shortcuts; the image files and case metadata are preserved for thesis review."]
    (figdir/'VISUAL_SHORTCUT_AUDIT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

def main():
    setup(); state('phase1_audit')
    try:
        cohort=pd.read_csv(V1_OUT/'cohort.csv'); v1_diagnostic(cohort); locked=create_locked_split(cohort); label_audit(cohort,locked); gradcam_audit(cohort,locked)
        registry({"experiment_id":"PHASE1_V1_AUDIT","date/time":datetime.now(timezone.utc).isoformat(),"model":"V1 audit","status":"completed","notes":"Preserved V1; locked seed-2026 holdout; created 5 development folds; label and Grad-CAM audits."})
        state('phase1_complete',completed='phase1_diagnostic',development_studies=int((locked.split=='development').sum()),final_test_studies=int((locked.split=='final_test').sum()),final_test_labels_accessed_for_optimization=False,final_test_findings_accessed=False)
    except Exception as e: state('phase1_failed',error=f'{type(e).__name__}: {e}'); raise

if __name__=='__main__': main()
