import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)

from .audit import dev_data,state
from .config import IMPROVED,LABELS,OUT,PROB,YCOL,setup
from .experiments import metrics
from src.iu_paired_improved.phase1 import atomic_json

def loadp(name):
    p=pd.read_csv(OUT/'oof'/f'{name}.csv'); p.uid=p.uid.astype(str); return p
def truth():
    d=dev_data()[['uid',*YCOL]]; d.uid=d.uid.astype(str); return d
def evaluate(name,p,y=None):
    yb=truth() if y is None else y; z=yb.merge(p,on='uid',validate='one_to_one'); m,th=metrics(z[YCOL].to_numpy(),z[PROB].to_numpy()); folder=OUT/'model1'/name; folder.mkdir(parents=True,exist_ok=True); p.to_csv(OUT/'oof'/f'{name}.csv',index=False); atomic_json(folder/'oof_metrics.json',{**m,'thresholds':dict(zip(LABELS,th.tolist()))}); return m
def average(names,name):
    fs=[loadp(x).set_index('uid')[PROB] for x in names]; p=(sum(fs)/len(fs)).reset_index(); return p,evaluate(name,p)
def blend(a,b,name):
    pa=loadp(a); pb=loadp(b); z=pa.merge(pb,on='uid',suffixes=('_a','_b')); ya=truth().merge(z[['uid']],on='uid')[YCOL].to_numpy(); A=np.column_stack([z[f'{c}_a'] for c in PROB]); B=np.column_stack([z[f'{c}_b'] for c in PROB]); rows=[]
    for w in np.arange(0,1.001,.025):
        p=w*A+(1-w)*B; m,_=metrics(ya,p); rows.append((m['macro_auprc'],m['macro_auroc'],m['macro_f1'],float(w),p))
    ap,au,f1,w,p=max(rows,key=lambda x:x[:3]); out=pd.DataFrame({'uid':z.uid,**{PROB[j]:p[:,j] for j in range(10)}}); m=evaluate(name,out); atomic_json(OUT/'model1'/name/'blend_config.json',{'first':a,'second':b,'weight_first':w,'selection':['macro AUPRC','macro AUROC','macro F1'],'development_only':True}); return out,m
def p4():
    v=loadp('V2_best'); c=loadp('P1_clahe'); r=loadp('P2_roi_ensemble3'); z=v.merge(c,on='uid',suffixes=('_v','_c')).merge(r,on='uid'); y=truth().merge(z[['uid']],on='uid')[YCOL].to_numpy(); V=np.column_stack([z[f'{x}_v'] for x in PROB]); C=np.column_stack([z[f'{x}_c'] for x in PROB]); R=z[PROB].to_numpy(); best=None
    for wv in np.arange(0,1.01,.1):
        for wc in np.arange(0,1.01-wv,.1):
            wr=1-wv-wc; p=wv*V+wc*C+wr*R; m,_=metrics(y,p); item=(m['macro_auprc'],m['macro_auroc'],m['macro_f1'],wv,wc,wr,p)
            if best is None or item[:3]>best[:3]: best=item
    *_,wv,wc,wr,p=best; out=pd.DataFrame({'uid':z.uid,**{PROB[j]:p[:,j] for j in range(10)}}); m=evaluate('P4_original_clahe_roi',out); atomic_json(OUT/'model1'/'P4_original_clahe_roi'/'blend_config.json',{'V2_original':wv,'CLAHE':wc,'lung_ROI':wr,'development_only':True}); return m
def fusion(image_name):
    im=loadp(image_name); tx=pd.read_csv(IMPROVED/'oof'/'model2_word_char.csv'); tx.uid=tx.uid.astype(str); z=im.merge(tx,on='uid',suffixes=('_i','_t')); y=truth().merge(z[['uid']],on='uid')[YCOL].to_numpy(); pi=np.column_stack([z[f'{c}_i'] for c in PROB]); pt=np.column_stack([z[f'{c}_t'] for c in PROB]); rows=[]
    for a in np.arange(0,1.001,.025):
        p=a*pi+(1-a)*pt; m,_=metrics(y,p); rows.append((m['macro_auprc'],m['macro_auroc'],m['macro_f1'],float(a),p,m))
    *_,a,p,m=max(rows,key=lambda x:x[:3]); out=OUT/'model3'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame({'uid':z.uid,**{PROB[j]:p[:,j] for j in range(10)}}).to_csv(out/'best_oof_predictions.csv',index=False); _,th=metrics(y,p); atomic_json(out/'best_fusion_config.json',{'image_source':image_name,'alpha_image':a,'thresholds':dict(zip(LABELS,th.tolist())),'oof_metrics':m,'development_only':True}); pd.DataFrame([{'alpha':x[3],'macro_auprc':x[0],'macro_auroc':x[1],'macro_f1':x[2]} for x in rows]).to_csv(out/'alpha_search.csv',index=False); return m,a
def binary(image_name):
    p=loadp(image_name); z=truth().merge(p,on='uid'); y=(z[YCOL].sum(axis=1)>0).astype(int).to_numpy(); prob=1-np.prod(1-z[PROB].to_numpy(),axis=1); grid=[]
    for t in np.arange(.01,1,.01): grid.append((f1_score(y,prob>=t),-abs(recall_score(y,prob>=t)-.8),t))
    t=max(grid)[2]; pred=prob>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); m={'task':'secondary normal vs any selected abnormal','threshold':float(t),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'sensitivity':recall_score(y,pred),'specificity':tn/(tn+fp),'f1':f1_score(y,pred),'auroc':roc_auc_score(y,prob),'auprc':average_precision_score(y,prob),'development_oof_only':True}; folder=OUT/'secondary_binary'; folder.mkdir(parents=True,exist_ok=True); atomic_json(folder/'metrics.json',m); pd.DataFrame({'uid':z.uid,'true_any_abnormal':y,'prob_any_abnormal':prob,'pred_any_abnormal':pred.astype(int)}).to_csv(folder/'oof_predictions.csv',index=False); return m
def main():
    setup(); state('ensemble_fusion_analysis'); _,rm=average(['P2_lung_roi','P2_lung_roi_seed123','P2_lung_roi_seed2026'],'P2_roi_ensemble3'); _,combo=blend('V2_best','P2_roi_ensemble3','V2_plus_roi_ensemble'); pm=p4()
    names=['V2_best','P1_clahe','P2_lung_roi','P3_original_roi','P4_original_clahe_roi','P0_balanced','P0_hardneg','P0_mixup','P2_roi_ensemble3','V2_plus_roi_ensemble']; rows=[]
    # V2 metrics are reconstructed using the identical rescue metric implementation.
    evaluate('V2_best',loadp('V2_best'))
    for n in names:
        m=json.loads((OUT/'model1'/n/'oof_metrics.json').read_text()); rows.append({'experiment':n,**{k:m[k] for k in ['macro_auroc','macro_auprc','macro_f1','micro_f1','macro_balanced_accuracy','label_wise_accuracy']}})
    rank=pd.DataFrame(rows).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False); rank.to_csv(OUT/'MODEL1_RESCUE_RANKING.csv',index=False); best=rank.iloc[0].experiment; fm,a=fusion(best); bm=binary(best); atomic_json(OUT/'BEST_RESCUE_CONFIG.json',{'best_image_oof':best,'image_metrics':rank.iloc[0].to_dict(),'fusion_alpha':a,'fusion_metrics':fm,'secondary_binary':bm,'selection_scope':'DEVELOPMENT OOF only','final_test_evaluated':False}); state('analysis_complete',completed='ensemble_fusion_binary',best_image=best,best_fusion_alpha=a)
if __name__=='__main__': main()
