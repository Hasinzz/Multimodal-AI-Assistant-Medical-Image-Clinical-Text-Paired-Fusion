import json

import pandas as pd

from .config import OUT, V1_OUT
from .modeling import YCOL
from .phase1 import atomic_json, full_metrics
from .phase2 import PROB

def main():
    cohort=pd.read_csv(V1_OUT/'cohort.csv'); split=pd.read_csv(OUT/'final_locked_split.csv'); truth=split[split.split=='final_test'][['uid']].merge(cohort,on='uid',validate='one_to_one'); truth.uid=truth.uid.astype(str)
    paths={'image':OUT/'model1'/'final_test_predictions.csv','text':OUT/'model2'/'final_test_predictions.csv','fusion':OUT/'model3'/'final_test_predictions.csv'}; rows=[]; results={}
    for name,path in paths.items():
        p=pd.read_csv(path); p.uid=p.uid.astype(str); z=truth.merge(p,on='uid',validate='one_to_one'); m=full_metrics(z[YCOL].to_numpy(),z[PROB].to_numpy(),[.5]*10); results[name]=m; atomic_json(path.parent/'final_metrics_threshold_0.5.json',m); rows.append({'model':name,**{k:m[k] for k in ['label_wise_accuracy','exact_match_accuracy','macro_balanced_accuracy','macro_auroc','macro_auprc','macro_f1','micro_f1']}})
    pd.DataFrame(rows).to_csv(OUT/'FINAL_THRESHOLD_0.5_METRICS.csv',index=False)
    report=OUT/'FINAL_IMPROVED_RESULTS.md'; text=report.read_text(encoding='utf-8'); marker='## 17. Generation'
    addition=("## Fixed threshold 0.5 check\n\n"
              f"At threshold 0.5, fusion label-wise accuracy was {results['fusion']['label_wise_accuracy']:.4f}, exact match {results['fusion']['exact_match_accuracy']:.4f}, balanced accuracy {results['fusion']['macro_balanced_accuracy']:.4f}, macro F1 {results['fusion']['macro_f1']:.4f}, and micro F1 {results['fusion']['micro_f1']:.4f}. It did not reach 95%. The primary OOF-tuned-threshold result also did not reach 95%, and the all-negative baseline remained higher in raw accuracy while having zero positive recall.\n\n")
    if addition not in text: report.write_text(text.replace(marker,addition+marker),encoding='utf-8')
if __name__=='__main__': main()
