import json
import math
import random
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch import nn
from torch.utils.data import DataLoader

from .config import CKPT, LABELS, OUT, PROJECTIONS, V1_OUT, setup
from .modeling import ImageBranch, ImageDataset, YCOL
from .phase1 import atomic_json, full_metrics, registry, state
from .phase2 import PROB, aggregate_view, infer
from .phase3 import clean, vectorizer
from src.iu_paired.template_generator import (ALIASES, UNSUPPORTED, clean_sentence,
                                               concepts, derive_library, negated, sentences)

def load_partitions():
    cohort=pd.read_csv(V1_OUT/'cohort.csv'); locked=pd.read_csv(OUT/'final_locked_split.csv'); dev_ids=locked[locked.split=='development'][['uid']]; final_ids=locked[locked.split=='final_test'][['uid']]
    dev=dev_ids.merge(cohort,on='uid',validate='one_to_one'); final_hidden=final_ids.merge(cohort,on='uid',validate='one_to_one'); return dev,final_hidden

def projection_map(): return pd.read_csv(PROJECTIONS).set_index('filename').projection.to_dict()

def train_image_seed(dev,final_inputs,seed,epochs=(2,8,6)):
    cfg=json.loads((OUT/'model1'/'best_config.json').read_text()); base=cfg['candidate_config']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); device=torch.device('cuda'); model=ImageBranch(base['pretraining'],False).to(device); proj=projection_map()
    pos=dev[YCOL].sum().to_numpy(float); pw=torch.tensor((len(dev)-pos)/np.maximum(pos,1),dtype=torch.float32,device=device); lossfn=nn.BCEWithLogitsLoss(pos_weight=pw); ds=ImageDataset(dev,base['resolution'],True,base['pretraining'],proj); loader=DataLoader(ds,batch_size=64,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3); history=[]; scaler=torch.amp.GradScaler('cuda')
    for (phase,n,lr) in zip(['head','late','full'],epochs,[1e-3,1e-4,3e-5]):
        model.set_phase(phase); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(n,1))
        for ep in range(n):
            model.train(); losses=[]
            for x,y,*_ in loader:
                x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'): loss=lossfn(model(x),y)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5); scaler.step(opt); scaler.update(); losses.append(loss.item())
            history.append({'seed':seed,'phase':phase,'phase_epoch':ep+1,'loss':float(np.mean(losses)),'lr':opt.param_groups[0]['lr']}); sched.step()
    ck=CKPT/'model1'/f'final_seed{seed}.pt'; ck.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'config':base,'seed':seed,'epochs':epochs,'development_only':True},ck)
    vl=DataLoader(ImageDataset(final_inputs,base['resolution'],False,base['pretraining'],proj),batch_size=64,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3); image,_=infer(model,vl,device,False); image=image.merge(pd.read_csv(PROJECTIONS)[['filename','projection']],on='filename',how='left'); study=aggregate_view(image,cfg['view_strategy'],list(cfg['view_weights'].values()) if cfg['view_weights'] else None); return image,study,history,ck

def train_final_branches(dev,final_inputs):
    out1=OUT/'model1'; all_study=[]; histories=[]; checkpoints=[]
    for seed in [42,123,2026]:
        image,study,h,ck=train_image_seed(dev,final_inputs,seed); image.to_csv(out1/f'final_test_image_predictions_seed{seed}.csv',index=False); study.to_csv(out1/f'final_test_predictions_seed{seed}.csv',index=False); all_study.append(study.set_index('uid')[PROB]); histories+=h; checkpoints.append(str(ck))
    ens=sum(all_study)/len(all_study); ens.reset_index().to_csv(out1/'final_test_predictions.csv',index=False); pd.DataFrame(histories).to_csv(out1/'final_training_history.csv',index=False)
    best=json.loads((out1/'best_config.json').read_text()); best['final_checkpoints']=checkpoints; best['final_training_epochs']={'head':2,'late':8,'full':6}; atomic_json(out1/'best_config.json',best)
    # Final text model on all DEVELOPMENT indications only.
    tc=json.loads((OUT/'model2'/'best_config.json').read_text()); vec=vectorizer(tc['kind']); x=vec.fit_transform(dev.indication.map(clean)); from sklearn.linear_model import LogisticRegression; from sklearn.multiclass import OneVsRestClassifier
    clf=OneVsRestClassifier(LogisticRegression(C=tc['C'],class_weight='balanced',max_iter=1500,random_state=2026)).fit(x,dev[YCOL].to_numpy()); p=clf.predict_proba(vec.transform(final_inputs.indication.map(clean))); model_path=CKPT/'model2'/'final_tfidf_logreg.joblib'; model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'vectorizer':vec,'classifier':clf,'labels':LABELS,'config':tc,'development_only':True},model_path); tp=pd.DataFrame({'uid':final_inputs.uid.astype(str),**{PROB[j]:p[:,j] for j in range(10)}}); tp.to_csv(OUT/'model2'/'final_test_predictions.csv',index=False); tc['final_checkpoint']=str(model_path); atomic_json(OUT/'model2'/'best_config.json',tc)
    return ens.reset_index(),tp,checkpoints,model_path

def derive_generator(dev):
    records=[]
    for _,r in dev.iterrows(): records.append({'uid':r.uid,'findings':r.findings,'labels':[LABELS[j] for j in range(10) if r[YCOL[j]]], 'problems_exact_normal':bool(r.problems_exact_normal)})
    base=derive_library(records); banks={}
    for label in LABELS:
        cand=[]
        for r in records:
            if label not in r['labels']: continue
            for s in sentences(r['findings']):
                if any(a in s.lower() for a in ALIASES[label]) and not negated(s,ALIASES[label]) and not UNSUPPORTED.search(s) and len(concepts(s))==1 and not any(ch.isdigit() for ch in s): cand.append({'template':s,'source_uid':str(r['uid'])})
        seen=set(); safe=[]
        for x in sorted(cand,key=lambda x:(abs(len(x['template'].split())-9),len(x['template']),x['template'].lower(),x['source_uid'])):
            k=x['template'].lower()
            if k not in seen: seen.add(k); safe.append(x)
        banks[label]=safe[:10] or [base['labels'][label]]
    normals=[]
    for r in records:
        if r['problems_exact_normal']:
            s=clean_sentence(r['findings'])
            if s and not concepts(s) and not UNSUPPORTED.search(s) and 'xxxx' not in s.lower() and not any(ch.isdigit() for ch in s): normals.append({'template':s,'source_uid':str(r['uid'])})
    seen=set(); normalbank=[]
    for x in sorted(normals,key=lambda x:(len(x['template']),x['template'].lower(),x['source_uid'])):
        if x['template'].lower() not in seen: seen.add(x['template'].lower()); normalbank.append(x)
    lib={'source':'complete DEVELOPMENT Findings only; locked final-test Findings hidden','pathology_banks':banks,'normal_bank':normalbank[:25] or [base['normal']], 'selection':'deterministic concise safe phrase; no laterality/severity/measurements; contradiction filtering','fallbacks':sum(bool(x[0].get('fallback',False)) for x in banks.values())+bool((normalbank[:1] or [base['normal']])[0].get('fallback',False))}
    out=OUT/'generator'; out.mkdir(parents=True,exist_ok=True); atomic_json(out/'template_library_v2.json',lib); return lib

def fuse_and_generate(image,text,final_inputs,lib):
    cfg=json.loads((OUT/'model3'/'best_fusion_config.json').read_text()); image.uid=image.uid.astype(str); text.uid=text.uid.astype(str); base=final_inputs[['uid','indication']].copy(); base.uid=base.uid.astype(str); z=base.merge(image,on='uid').merge(text,on='uid',suffixes=('_image','_text'),validate='one_to_one'); pi=np.column_stack([z[f'{c}_image'] for c in PROB]); pt=np.column_stack([z[f'{c}_text'] for c in PROB])
    if cfg['strategy']=='global_fusion': pf=cfg['config']['alpha']*pi+(1-cfg['config']['alpha'])*pt
    elif cfg['strategy'] in ('per_label_fusion','selective_fusion'):
        a=np.array(list(cfg['config']['alphas'].values())); pf=a*pi+(1-a)*pt
    else: raise RuntimeError(f"Frozen final strategy {cfg['strategy']} requires unsupported final application")
    th=np.array(list(cfg['thresholds'].values())); rows=[]
    for i,r in z.iterrows():
        idx=np.where(pf[i]>=th)[0]; idx=idx[np.argsort(-pf[i,idx])]; labels=[LABELS[j] for j in idx]; texts=[]
        for j in idx:
            t=lib['pathology_banks'][LABELS[j]][0]['template']
            if t not in texts: texts.append(t)
        generated=' '.join(texts) if texts else lib['normal_bank'][0]['template']
        rows.append({'uid':r.uid,'indication':r.indication,'predicted_labels':'|'.join(labels),'fused_probabilities':json.dumps(dict(zip(LABELS,map(float,pf[i])))),'generated_findings':generated})
    pred=pd.DataFrame({'uid':z.uid,**{PROB[j]:pf[:,j] for j in range(10)}}); pred.to_csv(OUT/'model3'/'final_test_predictions.csv',index=False); gen=pd.DataFrame(rows); gen.to_csv(OUT/'generator'/'generated_findings_locked_before_reveal.csv',index=False); return z,pi,pt,pf,gen

def bootstrap(y,models,thresholds,n=1000):
    rng=np.random.default_rng(2026); rows=[]; names=list(models)
    def basic(yy,pp,th):
        try: au=roc_auc_score(yy,pp,average='macro')
        except ValueError: au=np.nan
        ap=np.mean([average_precision_score(yy[:,j],pp[:,j]) for j in range(10)]); f1=f1_score(yy,pp>=np.asarray(th),average='macro',zero_division=0); return au,ap,f1
    for name,p in models.items():
        vals=[]
        for _ in range(n):
            ix=rng.integers(0,len(y),len(y)); vals.append(basic(y[ix],p[ix],thresholds[name]))
        a=np.array(vals); est=basic(y,p,thresholds[name]); rows.append({'comparison':name,'metric':'macro_auroc','estimate':est[0],'ci_low':np.nanpercentile(a[:,0],2.5),'ci_high':np.nanpercentile(a[:,0],97.5)}); rows.append({'comparison':name,'metric':'macro_auprc','estimate':est[1],'ci_low':np.nanpercentile(a[:,1],2.5),'ci_high':np.nanpercentile(a[:,1],97.5)}); rows.append({'comparison':name,'metric':'macro_f1','estimate':est[2],'ci_low':np.nanpercentile(a[:,2],2.5),'ci_high':np.nanpercentile(a[:,2],97.5)})
    # Paired differences use the same locked final studies.
    for other in ['improved_image','improved_text']:
        vals=[]
        for _ in range(n):
            ix=rng.integers(0,len(y),len(y)); f=basic(y[ix],models['improved_fusion'][ix],thresholds['improved_fusion']); o=basic(y[ix],models[other][ix],thresholds[other]); vals.append((f[0]-o[0],f[1]-o[1],f[2]-o[2]))
        a=np.array(vals); est=np.array(basic(y,models['improved_fusion'],thresholds['improved_fusion']))-np.array(basic(y,models[other],thresholds[other])); rows += [{'comparison':f'fusion_minus_{other}','metric':'macro_auroc_difference','estimate':est[0],'ci_low':np.nanpercentile(a[:,0],2.5),'ci_high':np.nanpercentile(a[:,0],97.5)},{'comparison':f'fusion_minus_{other}','metric':'macro_auprc_difference','estimate':est[1],'ci_low':np.nanpercentile(a[:,1],2.5),'ci_high':np.nanpercentile(a[:,1],97.5)},{'comparison':f'fusion_minus_{other}','metric':'macro_f1_difference','estimate':est[2],'ci_low':np.nanpercentile(a[:,2],2.5),'ci_high':np.nanpercentile(a[:,2],97.5)}]
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_confidence_intervals.csv',index=False)

def figures(y,pi,pt,pf,metrics,history,gen):
    out=OUT/'figures'; out.mkdir(parents=True,exist_ok=True)
    h=pd.read_csv(history); fig,ax=plt.subplots();
    for seed,g in h.groupby('seed'): ax.plot(np.arange(len(g))+1,g.loss,label=str(seed))
    ax.set(xlabel='Epoch',ylabel='Training loss',title='Final image ensemble training curves'); ax.legend(); fig.tight_layout(); fig.savefig(out/'model1_training_curves.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(16,6));
    for j,ax in enumerate(axes.flat): fpr,tpr,_=roc_curve(y[:,j],pf[:,j]); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'k--',lw=.5); ax.set_title(LABELS[j],fontsize=8)
    fig.tight_layout(); fig.savefig(out/'roc_curves_per_label.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(16,6));
    for j,ax in enumerate(axes.flat): pr,re,_=precision_recall_curve(y[:,j],pf[:,j]); ax.plot(re,pr); ax.set_title(LABELS[j],fontsize=8)
    fig.tight_layout(); fig.savefig(out/'pr_curves_per_label.png',dpi=180); plt.close(fig)
    per=pd.DataFrame(metrics['improved_fusion']['per_label']); per.plot(x='label',y=['auroc','auprc','f1'],kind='bar',figsize=(14,6)); plt.tight_layout(); plt.savefig(out/'per_label_performance.png',dpi=180); plt.close()
    pd.DataFrame({'label':LABELS,'prevalence':y.mean(0)}).plot(x='label',y='prevalence',kind='bar',legend=False,figsize=(12,5)); plt.tight_layout(); plt.savefig(out/'label_prevalence.png',dpi=180); plt.close()
    pd.DataFrame([{'model':k,'macro_auroc':v['macro_auroc'],'macro_auprc':v['macro_auprc'],'macro_f1':v['macro_f1']} for k,v in metrics.items()]).plot(x='model',kind='bar',figsize=(10,5)); plt.tight_layout(); plt.savefig(out/'image_text_fusion_comparison.png',dpi=180); plt.close()
    a=json.loads((OUT/'model3'/'best_fusion_config.json').read_text()); alphas=[a['config'].get('alpha',np.nan)]*10 if 'alpha' in a['config'] else list(a['config']['alphas'].values()); pd.DataFrame({'label':LABELS,'alpha_image':alphas}).plot(x='label',y='alpha_image',kind='bar',legend=False,figsize=(12,5)); plt.tight_layout(); plt.savefig(out/'alpha_plot.png',dpi=180); plt.close()
    pd.DataFrame({'label':LABELS,'threshold':list(a['thresholds'].values())}).plot(x='label',y='threshold',kind='bar',legend=False,figsize=(12,5)); plt.tight_layout(); plt.savefig(out/'threshold_plot.png',dpi=180); plt.close()
    conf=[]; th=np.array(list(a['thresholds'].values())); pred=pf>=th
    for j,l in enumerate(LABELS):
        tn,fp,fn,tp=confusion_matrix(y[:,j],pred[:,j],labels=[0,1]).ravel(); conf.append({'label':l,'tn':tn,'fp':fp,'fn':fn,'tp':tp})
    pd.DataFrame(conf).to_csv(out/'confusion_summary.csv',index=False)
    # Non-cherry-picked representative cases by structured per-study F1: best, median, worst.
    case=[]
    for i in range(len(y)): case.append(f1_score(y[i],pred[i],zero_division=0))
    order=np.argsort(case); picks=[('failure',order[0]),('partial',order[len(order)//2]),('strong',order[-1])]; ex=[]
    for kind,i in picks:
        r=gen.iloc[i]; ex.append({'case_type':kind,'uid':r.uid,'study_structured_f1':case[i],'generated_findings':r.generated_findings,'actual_findings':r.actual_findings})
    pd.DataFrame(ex).to_csv(out/'representative_generated_findings_examples.csv',index=False)

def final_evaluation(final_hidden,z,pi,pt,pf,gen):
    # This is the single point where locked Problems and Findings are revealed.
    truth=final_hidden[['uid','findings',*YCOL]].copy(); truth.uid=truth.uid.astype(str); ordered=z[['uid']].merge(truth,on='uid',validate='one_to_one'); y=ordered[YCOL].to_numpy(); best=json.loads((OUT/'model3'/'best_fusion_config.json').read_text()); fth=list(best['thresholds'].values()); ith=list(json.loads((OUT/'model1'/'best_config.json').read_text())['ensemble_thresholds'].values()); tc=json.loads((OUT/'model2'/json.loads((OUT/'model2'/'best_config.json').read_text())['kind']/'best_config.json').read_text()); tth=list(json.loads((OUT/'model2'/tc['kind']/f"metrics_C{tc['C']:g}.json").read_text())['thresholds'].values())
    metrics={'improved_image':full_metrics(y,pi,ith),'improved_text':full_metrics(y,pt,tth),'improved_fusion':full_metrics(y,pf,fth)}
    atomic_json(OUT/'model1'/'final_metrics.json',metrics['improved_image']); atomic_json(OUT/'model2'/'final_metrics.json',metrics['improved_text']); atomic_json(OUT/'model3'/'final_metrics.json',metrics['improved_fusion'])
    actual=ordered[['uid','findings']].rename(columns={'findings':'actual_findings'}); finalgen=gen.merge(actual,on='uid',validate='one_to_one'); scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True); scores=[scorer.score(a,g) for a,g in zip(finalgen.actual_findings,finalgen.generated_findings)]
    for k in ['rouge1','rouge2','rougeL']: finalgen[k]=[s[k].fmeasure for s in scores]
    finalgen.to_csv(OUT/'generator'/'generated_findings_test.csv',index=False); gm={k:float(finalgen[k].mean()) for k in ['rouge1','rouge2','rougeL']}; gm.update({'BERTScore':'skipped: heavy model not already available','radiology_semantic_metrics':'skipped after dependency practicality check','interpretation':'ROUGE is secondary overlap, not clinical accuracy'}); atomic_json(OUT/'generator'/'generation_metrics.json',gm)
    baseline={'label_wise_accuracy':float((1-y).mean()),'exact_match_accuracy':float(np.all(y==0,axis=1).mean())}; atomic_json(OUT/'all_negative_baseline.json',baseline); bootstrap(y,{'improved_image':pi,'improved_text':pt,'improved_fusion':pf},{'improved_image':ith,'improved_text':tth,'improved_fusion':fth}); figures(y,pi,pt,pf,metrics,OUT/'model1'/'final_training_history.csv',finalgen); return metrics,gm,baseline

def comparison(metrics):
    v1=json.loads((OUT/'v1_diagnostic_metrics.json').read_text()); rows=[]
    mapping={'V1 image':'V1 image','V1 text':'V1 text','V1 global fusion':'V1 global fusion'}
    cols=['label_wise_accuracy','exact_match_accuracy','macro_balanced_accuracy','macro_auroc','micro_auroc','macro_auprc','micro_auprc','macro_f1','micro_f1','macro_precision','macro_recall']
    for name,key in mapping.items(): rows.append({'model':name,'evaluation_split':'original V1 test (344; not directly paired with improved test)',**{c:v1[key][c] for c in cols}})
    for key,name in [('improved_image','Improved image'),('improved_text','Improved text'),('improved_fusion','Improved fusion')]: rows.append({'model':name,'evaluation_split':'new locked seed-2026 final test (352)',**{c:metrics[key][c] for c in cols}})
    table=pd.DataFrame(rows); table.to_csv(OUT/'FINAL_COMPARISON.csv',index=False)
    plot=table[table.model.isin(['V1 global fusion','Improved fusion'])].set_index('model')[['macro_auroc','macro_auprc','macro_f1']]; plot.plot(kind='bar',figsize=(8,5)); plt.ylabel('Metric value'); plt.title('V1 reference vs improved (different test splits)'); plt.tight_layout(); plt.savefig(OUT/'figures'/'v1_vs_improved_comparison.png',dpi=180); plt.close()

def report(metrics,gm,baseline):
    v1=json.loads((OUT/'v1_diagnostic_metrics.json').read_text())['V1 global fusion']; im=metrics['improved_image']; tx=metrics['improved_text']; fu=metrics['improved_fusion']; besti=json.loads((OUT/'model1'/'best_config.json').read_text()); bestt=json.loads((OUT/'model2'/'best_config.json').read_text()); bestf=json.loads((OUT/'model3'/'best_fusion_config.json').read_text()); la=pd.read_csv(OUT/'label_audit'/'label_consistency_summary.csv'); rank=pd.read_csv(OUT/'model1'/'candidate_ranking.csv'); fr=pd.read_csv(OUT/'model3'/'fusion_ranking.csv'); ci=pd.read_csv(OUT/'bootstrap_confidence_intervals.csv')
    lines=["# Final Improved Results","","## 1. Current V1 diagnosis",f"V1 fusion on its original test: macro AUROC {v1['macro_auroc']:.4f}, AUPRC {v1['macro_auprc']:.4f}, F1 {v1['macro_f1']:.4f}. Its raw label-wise accuracy {v1['label_wise_accuracy']:.4f} was imbalance-dominated.",
      "","## 2. Major bottlenecks","Weak/noisy Problems labels, rare positives, weak Indication signal, probability miscalibration, and simplistic view averaging were the main evidence-backed bottlenecks.",
      "","## 3. Label quality audit",f"Development-only consistency audit completed for all labels. Conservative positive-negation rates ranged {la.positive_negation_rate.min():.3f}–{la.positive_negation_rate.max():.3f}; no primary labels were rewritten.",
      "","## 4. Visual shortcut audit","Fifteen confidence-ranked Grad-CAM TP/FP/FN cases were saved for five priority labels. Maps are diagnostic, not causal localization; no segmentation/cropping was added without conclusive evidence.",
      "","## 5–9. Model-1 experiments",f"Winner: {besti['base_candidate']}, {besti['view_strategy']}, three-seed ensemble={besti['ensemble_used']}. CXR-specific pretraining did not help: its best candidate AUPRC was {rank[rank.candidate.str.startswith('cxr')].macro_auprc.max():.4f} versus ImageNet {rank.iloc[0].macro_auprc:.4f}. ASL did not help. Higher 320/384 resolution did not help. Learned gated multi-view did not help, while validation-learned view weighting improved the selected model. The three-seed ensemble improved OOF AUPRC/AUROC and was retained.",
      "","## 10. Model-2",f"Word+character TF-IDF won with C={bestt['C']}; development OOF macro AUPRC {bestt['metrics']['macro_auprc']:.4f}.",
      "","## 11–12. Calibration and fusion","Platt calibration strongly improved Brier/ECE but did not improve ranking metrics. Per-label weighted fusion and learned stacking did not beat global fusion. Indication benefited the aggregate global fusion, but per-label gains were insufficiently reproducible for the selective strategy to win.",
      "","## 13. Ensemble results",f"Selected OOF ensemble macro AUPRC {besti['ensemble_oof_metrics']['macro_auprc']:.4f} versus single {besti['single_oof_metrics']['macro_auprc']:.4f}.",
      "","## 14–16. Locked final-test results and accuracy",f"The locked 352-study final test was evaluated exactly once after freezing. Image: AUROC {im['macro_auroc']:.4f}, AUPRC {im['macro_auprc']:.4f}, F1 {im['macro_f1']:.4f}. Text: AUROC {tx['macro_auroc']:.4f}, AUPRC {tx['macro_auprc']:.4f}, F1 {tx['macro_f1']:.4f}. Fusion: AUROC {fu['macro_auroc']:.4f}, AUPRC {fu['macro_auprc']:.4f}, F1 {fu['macro_f1']:.4f}, micro F1 {fu['micro_f1']:.4f}.",f"Fusion label-wise accuracy {fu['label_wise_accuracy']:.4f}; exact match {fu['exact_match_accuracy']:.4f}; balanced accuracy {fu['macro_balanced_accuracy']:.4f}. All-negative label-wise baseline {baseline['label_wise_accuracy']:.4f}. >=95% label-wise accuracy achieved: {'YES' if fu['label_wise_accuracy']>=.95 else 'NO'}.",
      "","## 17. Generation",f"ROUGE-1/2/L: {gm['rouge1']:.4f}/{gm['rouge2']:.4f}/{gm['rougeL']:.4f}. This is text overlap, not clinical accuracy.",
      "","## 18. Statistical uncertainty","Paired bootstrap confidence intervals for improved fusion versus improved image/text are in `bootstrap_confidence_intervals.csv`. V1 and improved headline results use different test splits and are not presented as paired estimates.",
      "","## 19–20. Error analysis and limitations","Per-label tables, confusion summaries, non-cherry-picked generated examples, ROC/PR curves, and Grad-CAM cases are saved. Limitations: small internal study-wise dataset; no patient identifier; label noise; rare targets; upstream pretraining mismatch; no external/clinical validation.",
      "","## 21. Strongest defensible thesis claims",f"The development-selected pipeline uses 224-pixel ImageNet DenseNet-121 ensembles, view-aware probability aggregation, improved sparse Indication modeling, and global late fusion (alpha {bestf['config'].get('alpha')}). Fusion beat image-only on locked macro AUROC: {fu['macro_auroc']>im['macro_auroc']}; macro AUPRC: {fu['macro_auprc']>im['macro_auprc']}; macro F1: {fu['macro_f1']>im['macro_f1']}. Meaningful metrics improved over V1 references only descriptively because test splits differ. Main bottleneck remains rare, incomplete/noisy structured labels combined with weak Indication signal."]
    (OUT/'FINAL_IMPROVED_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    setup(); state('phase5_freeze_retrain',architecture_frozen=True,final_test_evaluations=0)
    dev,hidden=load_partitions(); # Hidden table is not queried for labels/findings until final_evaluation.
    inputs=hidden[['uid','image_filenames','indication']].copy()
    for c in YCOL: inputs[c]=0
    lib=derive_generator(dev); image,text,cks,textck=train_final_branches(dev,inputs); z,pi,pt,pf,gen=fuse_and_generate(image,text,inputs,lib)
    atomic_json(OUT/'FROZEN_FINAL_CONFIG.json',{'image':json.loads((OUT/'model1'/'best_config.json').read_text()),'text':json.loads((OUT/'model2'/'best_config.json').read_text()),'fusion':json.loads((OUT/'model3'/'best_fusion_config.json').read_text()),'generator':str(OUT/'generator'/'template_library_v2.json'),'frozen_before_final_label_reveal':True,'final_test_evaluations_before_this_point':0})
    metrics,gm,baseline=final_evaluation(hidden,z,pi,pt,pf,gen); comparison(metrics); report(metrics,gm,baseline)
    registry({'experiment_id':'LOCKED_FINAL_2026','date/time':datetime.now(timezone.utc).isoformat(),'model':'3-seed ImageNet DenseNet-121 ensemble + word/char TF-IDF + global late fusion','pretraining':'ImageNet','image resolution':224,'view strategy':'per-label frontal/lateral probability weights','loss':'BCE + DEVELOPMENT pos_weight','optimizer':'AdamW','learning rate':'1e-3/1e-4/3e-5','scheduler':'cosine per phase','epochs':'2 head + 8 late + 6 full','augmentation':'crop .88-1.0, rotation 7, no flip','text configuration':'word+char TF-IDF OVR LR C=.25; Indication only','calibration':'evaluated; uncalibrated selected by OOF','fusion method':'global alpha=.75 selected on OOF','threshold method':'per-label OOF F1 grid .01-.99','final-test metrics if final-test was legitimately run':json.dumps({k:metrics['improved_fusion'][k] for k in ['label_wise_accuracy','macro_balanced_accuracy','macro_auroc','macro_auprc','macro_f1','micro_f1']}),'checkpoint path':'|'.join(cks+[str(textck)]),'status':'completed','notes':'Single locked final-test evaluation after full freeze; generation saved before Findings reveal.'})
    state('complete',completed='locked_final_evaluation',final_test_evaluations=1,final_test_findings_revealed_after_generation=True,best_image_checkpoints=cks,best_text_checkpoint=str(textck),best_fusion_config=str(OUT/'model3'/'best_fusion_config.json'),final_report=str(OUT/'FINAL_IMPROVED_RESULTS.md'))
if __name__=='__main__': main()
