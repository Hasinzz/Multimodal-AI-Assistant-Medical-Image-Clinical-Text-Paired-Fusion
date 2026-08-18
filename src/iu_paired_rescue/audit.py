import csv
import json
import re
from datetime import datetime,timezone

import numpy as np
import pandas as pd

from .config import IMPROVED,LABELS,MISSING_LABEL_SUBSTANTIAL_RATE,OUT,V1,YCOL,setup
from src.iu_paired_improved.phase1 import ALIASES, NEG, atomic_json, markdown_table

def state(stage,completed=None,error=None,**kw):
    p=OUT/'run_state.json'; old=json.loads(p.read_text()) if p.exists() else {}; done=old.get('completed_stages',[])
    if completed and completed not in done: done.append(completed)
    obj={'completed_stages':done,'current_stage':stage,'selection_scope':'DEVELOPMENT/OOF only','previous_final_test_labels_used':False,'errors':[] if error is None else [str(error)]}; obj.update(kw); atomic_json(p,obj)

def dev_data():
    c=pd.read_csv(V1/'cohort.csv'); s=pd.read_csv(IMPROVED/'final_locked_split.csv'); f=pd.read_csv(IMPROVED/'development_folds.csv'); return s[s.split=='development'][['uid']].merge(c,on='uid',validate='one_to_one').merge(f,on='uid',validate='one_to_one')

def mention(text,label):
    supports=negates=False
    for sent in re.split(r'(?<=[.!?])\s+|[\r\n]+',str(text).lower()):
        for alias in ALIASES[label]:
            pos=sent.find(alias)
            if pos>=0:
                if NEG.search(sent[max(0,pos-45):pos]): negates=True
                else: supports=True
    return supports,negates

def histories():
    rows=[]
    for tag in ['imnet224_bce_mean','imnet224_bce_mean_seed123','imnet224_bce_mean_seed2026']:
        for fold in range(5):
            h=pd.read_csv(IMPROVED/'model1'/tag/f'fold_{fold}'/'history.csv'); best=h.loc[h.macro_auprc.idxmax()]; stop=h.iloc[-1]
            rows.append({'ensemble_seed_tag':tag,'fold':fold,'actual_epochs':len(h),'best_epoch':int(best.epoch),'stopping_epoch':int(stop.epoch),'best_macro_auprc':best.macro_auprc,'stopping_macro_auprc':stop.macro_auprc,'best_at_stopping_epoch':int(best.epoch)==int(stop.epoch),'last_3_slope':float(np.polyfit(np.arange(min(3,len(h))),h.macro_auprc.tail(3),1)[0]),'still_improving_at_termination':bool(int(best.epoch)==int(stop.epoch) or np.polyfit(np.arange(min(3,len(h))),h.macro_auprc.tail(3),1)[0]>0.001)})
    pd.DataFrame(rows).to_csv(OUT/'WINNING_HISTORY_AUDIT.csv',index=False)
    final=pd.read_csv(IMPROVED/'model1'/'final_training_history.csv').groupby('seed').size().reset_index(name='actual_final_retraining_epochs'); final['schedule']='2 head + 8 late-block + 6 full'; final.to_csv(OUT/'FINAL_ENSEMBLE_EPOCHS.csv',index=False)
    r=pd.DataFrame(rows); lines=['# Winning Model-1 History Audit','',markdown_table(final),'',f"OOF folds: {len(r)}; stopping epochs {r.stopping_epoch.min()}–{r.stopping_epoch.max()}; best epochs {r.best_epoch.min()}–{r.best_epoch.max()}.",f"Best at termination: {r.best_at_stopping_epoch.sum()}/{len(r)}. Operational still-improving flag: {r.still_improving_at_termination.sum()}/{len(r)}.",'','Conclusion: most folds peaked before termination, so the rescue does not blindly extend to 80 epochs. It keeps staged training with early stopping; longer training is conditional on rescue OOF curves.']; (OUT/'WINNING_HISTORY_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def label_audit():
    d=dev_data(); rows=[]
    aligned=d[['uid',*YCOL]].copy()
    for _,r in d.iterrows():
        for j,l in enumerate(LABELS):
            sup,neg=mention(r.findings,l); pos=bool(r[YCOL[j]])
            cat='problems_positive_findings_positive' if pos and sup else ('problems_positive_findings_negative' if pos and neg else ('problems_negative_findings_explicit_positive' if not pos and sup else 'not_mentioned_or_uncertain'))
            rows.append({'uid':r.uid,'label':l,'problems_positive':pos,'findings_positive':sup,'findings_negated':neg,'category':cat,'findings':r.findings})
            if not pos and sup: aligned.loc[aligned.uid==r.uid,YCOL[j]]=1
    detail=pd.DataFrame(rows); summary=detail.groupby(['label','category']).size().unstack(fill_value=0).reset_index()
    cats=['problems_positive_findings_positive','problems_positive_findings_negative','problems_negative_findings_explicit_positive','not_mentioned_or_uncertain']
    for c in cats:
        if c not in summary: summary[c]=0
    for i,r in summary.iterrows():
        j=LABELS.index(r.label); neg_count=int((d[YCOL[j]]==0).sum()); summary.loc[i,'problems_negative_count']=neg_count; summary.loc[i,'missing_positive_rate_among_problems_negative']=r.problems_negative_findings_explicit_positive/neg_count; summary.loc[i,'substantial_by_frozen_rule']=summary.loc[i,'missing_positive_rate_among_problems_negative']>=MISSING_LABEL_SUBSTANTIAL_RATE
    folder=OUT/'label_completeness'; folder.mkdir(parents=True,exist_ok=True); summary.to_csv(folder/'label_completeness_summary.csv',index=False)
    examples=pd.concat([g.sort_values('uid').head(12) for _,g in detail[detail.category!='not_mentioned_or_uncertain'].groupby(['label','category'])]); examples.to_csv(folder/'label_completeness_examples.csv',index=False); aligned.to_csv(folder/'findings_aligned_labels.csv',index=False)
    flagged=summary[summary.substantial_by_frozen_rule.astype(bool)].label.tolist(); lines=['# Label Completeness Audit','',f'Scope: {len(d)} DEVELOPMENT studies only. Frozen substantial threshold: {MISSING_LABEL_SUBSTANTIAL_RATE:.1%} of Problems-negative studies explicitly positive in Findings.','',markdown_table(summary),'',f"Substantial missing-label signal: {', '.join(flagged) if flagged else 'none'}.",'Primary Problems labels were not modified. `findings_aligned_labels.csv` is a separate experimental target created by frozen synonym and negation rules.']; (folder/'LABEL_COMPLETENESS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); atomic_json(folder/'audit_config.json',{'aliases':ALIASES,'negation_pattern':NEG.pattern,'substantial_threshold':MISSING_LABEL_SUBSTANTIAL_RATE,'development_only':True,'rules_frozen_before_rate_inspection':True,'flagged_labels':flagged})

def main():
    setup(); state('audit'); histories(); label_audit(); state('audit_complete',completed='history_and_label_audit')
if __name__=='__main__': main()
