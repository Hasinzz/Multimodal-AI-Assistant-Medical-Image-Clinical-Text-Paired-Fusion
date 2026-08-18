import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .config import CKPT, LABELS, OUT, setup
from .modeling import YCOL
from .phase1 import atomic_json, registry, state
from .phase2 import PROB, load_dev, score

def clean(x):
    import re
    return re.sub(r'\s+',' ',str(x)).strip().lower()

def vectorizer(kind):
    word=TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,min_df=2,max_features=30000)
    if kind=='word': return word
    char=TfidfVectorizer(lowercase=True,analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,min_df=2,max_features=40000)
    return FeatureUnion([('word',word),('char',char)])

def run_kind(kind):
    dev=load_dev(); out=OUT/'model2'/kind; ck=CKPT/'model2'/kind; out.mkdir(parents=True,exist_ok=True); ck.mkdir(parents=True,exist_ok=True); rows=[]
    for C in [.25,.5,1.,2.,4.]:
        parts=[]
        for fold in range(5):
            tr=dev[dev.fold!=fold]; va=dev[dev.fold==fold]; vec=vectorizer(kind); x=vec.fit_transform(tr.indication.map(clean)); xv=vec.transform(va.indication.map(clean)); clf=OneVsRestClassifier(LogisticRegression(C=C,class_weight='balanced',max_iter=1500,random_state=2026)).fit(x,tr[YCOL].to_numpy()); p=clf.predict_proba(xv)
            parts.append(pd.DataFrame({'uid':va.uid.astype(str),**{PROB[j]:p[:,j] for j in range(10)}})); joblib.dump({'vectorizer':vec,'classifier':clf,'labels':LABELS,'C':C,'fold':fold},ck/f'C{C:g}_fold{fold}.joblib')
        oof=pd.concat(parts,ignore_index=True); base=dev[['uid',*YCOL]].copy(); base.uid=base.uid.astype(str); z=base.merge(oof,on='uid',validate='one_to_one'); met,th=score(z[YCOL].to_numpy(),z[PROB].to_numpy()); oof.to_csv(out/f'oof_C{C:g}.csv',index=False); atomic_json(out/f'metrics_C{C:g}.json',{**met,'thresholds':dict(zip(LABELS,th.tolist()))}); rows.append({'C':C,**met})
    rank=pd.DataFrame(rows).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False); rank.to_csv(out/'C_ranking.csv',index=False); best=float(rank.iloc[0].C); src=out/f'oof_C{best:g}.csv'; pd.read_csv(src).to_csv(OUT/'oof'/f'model2_{kind}.csv',index=False); atomic_json(out/'best_config.json',{'kind':kind,'C':best,'metrics':rank.iloc[0].to_dict(),'input':'indication only','negation_retained':True})
    registry({'experiment_id':f'text_{kind}','date/time':datetime.now(timezone.utc).isoformat(),'model':'TF-IDF + OVR Logistic Regression','text configuration':kind+' TF-IDF; word(1,2), char_wb(3,5) when applicable; indication only','validation metrics':json.dumps(rank.iloc[0].to_dict()),'checkpoint path':str(ck),'status':'completed','notes':'5-fold DEVELOPMENT OOF C grid; no Findings/Impression/Problems input; final holdout untouched'})

def main():
    setup(); state('phase3_text')
    for kind in ['word','word_char']: run_kind(kind)
    candidates=[]
    for k in ['word','word_char']:
        x=json.loads((OUT/'model2'/k/'best_config.json').read_text()); candidates.append({'kind':k,**x['metrics']})
    rank=pd.DataFrame(candidates).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False); rank.to_csv(OUT/'model2'/'candidate_ranking.csv',index=False); best=rank.iloc[0]['kind']; cfg=json.loads((OUT/'model2'/best/'best_config.json').read_text()); atomic_json(OUT/'model2'/'best_config.json',cfg); state('phase3_complete',completed='model2_selection',best_text=best)
if __name__=='__main__': main()
