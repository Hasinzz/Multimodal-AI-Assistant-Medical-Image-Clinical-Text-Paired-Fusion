import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

from .config import LABELS, OUT, setup
from .modeling import YCOL
from .phase1 import atomic_json, full_metrics, registry, state
from .phase2 import PROB, load_dev

def logit(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); total=0
    for a,b in zip(edges[:-1],edges[1:]):
        m=(p>=a)&(p<(b if b<1 else b+1e-9))
        if m.any(): total+=m.mean()*abs(p[m].mean()-y[m].mean())
    return float(total)

def align():
    d=load_dev(); base=d[['uid','fold',*YCOL]].copy(); base.uid=base.uid.astype(str)
    ip=pd.read_csv(OUT/'oof'/'model1_best_ensemble.csv'); tp=pd.read_csv(OUT/'oof'/'model2_word_char.csv'); ip.uid=ip.uid.astype(str); tp.uid=tp.uid.astype(str)
    z=base.merge(ip,on='uid').merge(tp,on='uid',suffixes=('_image','_text'),validate='one_to_one'); y=z[YCOL].to_numpy(); pi=np.column_stack([z[f'{c}_image'] for c in PROB]); pt=np.column_stack([z[f'{c}_text'] for c in PROB]); return z,y,pi,pt

def crossfit_platt(y,p,folds):
    out=np.zeros_like(p); final=[]
    for j in range(10):
        for f in range(5):
            tr=folds!=f; va=folds==f; m=LogisticRegression(C=1,max_iter=1000).fit(logit(p[tr,j])[:,None],y[tr,j]); out[va,j]=m.predict_proba(logit(p[va,j])[:,None])[:,1]
        m=LogisticRegression(C=1,max_iter=1000).fit(logit(p[:,j])[:,None],y[:,j]); final.append({'coef':float(m.coef_[0,0]),'intercept':float(m.intercept_[0])})
    return out,final

def apply_platt(p,params):
    out=np.zeros_like(p)
    for j,x in enumerate(params): out[:,j]=1/(1+np.exp(-(x['coef']*logit(p[:,j])+x['intercept'])))
    return out

def tune_thresholds(y,p):
    grid=np.arange(.01,1,.01); return np.array([max(grid,key=lambda t:f1_score(y[:,j],p[:,j]>=t,zero_division=0)) for j in range(10)])

def metric(y,p):
    th=tune_thresholds(y,p); return full_metrics(y,p,th),th

def save_strategy(name,p,y,extra=None):
    folder=OUT/'model3'/name; folder.mkdir(parents=True,exist_ok=True); pd.DataFrame({**{PROB[j]:p[:,j] for j in range(10)}}).to_csv(folder/'oof_predictions.csv',index=False); m,th=metric(y,p); atomic_json(folder/'oof_metrics.json',m); atomic_json(folder/'thresholds.json',dict(zip(LABELS,th.tolist()))); atomic_json(folder/'config.json',extra or {}); return m,th

def plot_calibration(y,raw,cal,name):
    fig,axes=plt.subplots(2,5,figsize=(16,6),sharex=True,sharey=True)
    for j,ax in enumerate(axes.flat):
        for p,label in [(raw,'raw'),(cal,'calibrated')]:
            xs=[]; ys=[]
            for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                m=(p[:,j]>=a)&(p[:,j]<b)
                if m.any(): xs.append(p[m,j].mean()); ys.append(y[m,j].mean())
            ax.plot(xs,ys,marker='o',label=label)
        ax.plot([0,1],[0,1],'k--',lw=.7); ax.set_title(LABELS[j],fontsize=8)
    axes.flat[0].legend(fontsize=7); fig.tight_layout(); path=OUT/'calibration'/'plots'; path.mkdir(parents=True,exist_ok=True); fig.savefig(path/f'{name}_reliability.png',dpi=180); plt.close(fig)

def main():
    setup(); state('phase4_calibration_fusion'); z,y,pi,pt=align(); folds=z.fold.to_numpy(); ci,ipar=crossfit_platt(y,pi,folds); ct,tpar=crossfit_platt(y,pt,folds)
    caldir=OUT/'calibration'; caldir.mkdir(parents=True,exist_ok=True); rows=[]
    for branch,raw,cal in [('image',pi,ci),('text',pt,ct)]:
        for status,p in [('raw',raw),('calibrated',cal)]: rows.append({'branch':branch,'status':status,'mean_brier':float(np.mean([brier_score_loss(y[:,j],p[:,j]) for j in range(10)])),'mean_ece':float(np.mean([ece(y[:,j],p[:,j]) for j in range(10)]))})
        plot_calibration(y,raw,cal,branch)
    pd.DataFrame(rows).to_csv(caldir/'metrics.csv',index=False); atomic_json(caldir/'calibrators.json',{'method':'per-label Platt scaling cross-fitted for OOF assessment; final parameters fit on all DEVELOPMENT OOF predictions','image':ipar,'text':tpar})
    candidates={}; candidates['image_only']=save_strategy('image_only',ci,y,{'calibrated':True})[0]
    # Uncalibrated global fusion, including the V1 alpha=0.9 reference on new OOF data.
    global_grid=[]
    for a in np.arange(0,1.001,.025):
        p=a*pi+(1-a)*pt; ap=float(np.mean([average_precision_score(y[:,j],p[:,j]) for j in range(10)])); global_grid.append((ap,float(a)))
    a=max(global_grid)[1]; pg=a*pi+(1-a)*pt; candidates['global_fusion']=save_strategy('global_fusion',pg,y,{'alpha':a,'calibrated':False,'selection':'OOF macro AUPRC'})[0]
    pd.DataFrame([{'alpha':x[1],'macro_auprc':x[0]} for x in global_grid]).to_csv(OUT/'model3'/'global_fusion'/'alpha_search.csv',index=False)
    save_strategy('v1_alpha_reference',.9*pi+.1*pt,y,{'alpha':.9,'calibrated':False,'note':'V1 alpha evaluated on development OOF only'})
    cgrid=[]
    for a in np.arange(0,1.001,.025):
        p=a*ci+(1-a)*ct; cgrid.append((float(np.mean([average_precision_score(y[:,j],p[:,j]) for j in range(10)])),float(a)))
    ca=max(cgrid)[1]; candidates['calibrated_global']=save_strategy('calibrated_global',ca*ci+(1-ca)*ct,y,{'alpha':ca,'calibrated':True,'selection':'OOF macro AUPRC'})[0]
    alphas=[]
    for j in range(10): alphas.append(max(np.arange(0,1.001,.025),key=lambda a:average_precision_score(y[:,j],a*ci[:,j]+(1-a)*ct[:,j])))
    pp=ci*np.array(alphas)+(ct*(1-np.array(alphas))); candidates['per_label_fusion']=save_strategy('per_label_fusion',pp,y,{'alphas':dict(zip(LABELS,map(float,alphas))),'calibrated':True})[0]; atomic_json(OUT/'model3'/'per_label_fusion'/'per_label_alpha.json',dict(zip(LABELS,map(float,alphas))))
    selective=[]; benefits={}
    for j in range(10):
        gain=[]
        for f in range(5):
            m=folds==f; gain.append(average_precision_score(y[m,j],pp[m,j])-average_precision_score(y[m,j],ci[m,j]))
        qualified=sum(g>0 for g in gain)>=3 and np.mean(gain)>0; selective.append(alphas[j] if qualified else 1.0); benefits[LABELS[j]]={'per_fold_auprc_gain':list(map(float,gain)),'text_permitted':bool(qualified)}
    ps=ci*np.array(selective)+ct*(1-np.array(selective)); candidates['selective_fusion']=save_strategy('selective_fusion',ps,y,{'alphas':dict(zip(LABELS,map(float,selective))),'benefit_rule':'positive AUPRC gain in >=3/5 folds and positive mean gain','benefits':benefits})[0]
    # Cross-fitted learned late-fusion stacking with only branch logits as features.
    stack_versions={}; stack_models={}
    for C in [.1,1.,10.]:
        po=np.zeros_like(ci); finals=[]
        for j in range(10):
            X=np.c_[logit(ci[:,j]),logit(ct[:,j])]
            for f in range(5):
                tr=folds!=f; va=folds==f; m=LogisticRegression(C=C,max_iter=1000).fit(X[tr],y[tr,j]); po[va,j]=m.predict_proba(X[va])[:,1]
            finals.append(LogisticRegression(C=C,max_iter=1000).fit(X,y[:,j]))
        stack_versions[C]=po; stack_models[C]=finals
    bestC=max(stack_versions,key=lambda C:np.mean([average_precision_score(y[:,j],stack_versions[C][:,j]) for j in range(10)])); pst=stack_versions[bestC]; candidates['stacking']=save_strategy('stacking',pst,y,{'C':bestC,'features':['calibrated image logit','calibrated text logit'],'assessment':'cross-fitted on DEVELOPMENT OOF'})[0]; joblib.dump({'C':bestC,'models':stack_models[bestC]},OUT/'model3'/'stacking'/'stacker.joblib')
    rank=pd.DataFrame([{'strategy':k,**{q:v[q] for q in ['macro_auprc','macro_auroc','macro_f1','micro_f1','label_wise_accuracy','macro_balanced_accuracy']}} for k,v in candidates.items()]).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False); rank.to_csv(OUT/'model3'/'fusion_ranking.csv',index=False)
    fused=rank[rank.strategy!='image_only'].iloc[0]; best=fused.strategy; fused_dict=json.loads(fused.to_json()); cfg=json.loads((OUT/'model3'/best/'config.json').read_text()); thresholds=json.loads((OUT/'model3'/best/'thresholds.json').read_text()); atomic_json(OUT/'model3'/'best_fusion_config.json',{'strategy':best,'development_metrics':fused_dict,'config':cfg,'thresholds':thresholds,'calibrators':str(caldir/'calibrators.json'),'selection_priority':['macro AUPRC','macro AUROC','macro F1'],'final_test_used':False})
    registry({'experiment_id':'fusion_comparison','date/time':datetime.now(timezone.utc).isoformat(),'model':'decision-level late fusion','calibration':'cross-fitted per-label Platt','fusion method':'global, calibrated global, per-label, selective, cross-fitted logistic stacking','threshold method':'per-label F1 grid 0.01-0.99 on DEVELOPMENT OOF','validation metrics':json.dumps(fused_dict),'status':'completed','notes':'Final holdout untouched; stacker trained only on branch OOF predictions.'})
    state('phase4_complete',completed='calibration_fusion_selection',best_fusion=best)
if __name__=='__main__': main()
