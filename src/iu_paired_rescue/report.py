import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import state
from .config import IMPROVED,LABELS,OUT,setup
from .analyze import evaluate,loadp
from src.iu_paired_improved.phase1 import atomic_json

def m(path): return json.loads(Path(path).read_text())
def vals(x): return {k:x[k] for k in ['macro_auroc','macro_auprc','macro_f1','micro_f1','macro_balanced_accuracy','label_wise_accuracy']}

def main():
    setup(); state('reporting')
    # Ensure the final selected rescue image/fusion metadata reflects the last controlled comparison.
    selected_image='V2_plus_roi_balanced_ensemble'; image=m(OUT/'model1'/selected_image/'oof_metrics.json'); fusion=m(OUT/'model3'/'best_fusion_config.json'); binary=m(OUT/'secondary_binary'/'metrics.json')
    experiments=['V2_best','P1_clahe','P2_lung_roi','P3_original_roi','P4_original_clahe_roi','P0_balanced','P0_hardneg','P0_mixup','findings_aligned_labels','P2_roi_balanced','P2_roi_hardneg','P2_roi_mixup','P2_roi_ensemble3','P2_roi_balanced_ensemble3','V2_plus_roi_ensemble','V2_plus_roi_balanced_ensemble']
    rows=[]; per=[]
    for name in experiments:
        path=OUT/'model1'/name/'oof_metrics.json'
        if not path.exists(): continue
        x=m(path); rows.append({'experiment':name,**vals(x)})
        for q in x.get('per_label',[]): per.append({'experiment':name,**q})
    rows.append({'experiment':'best_rescue_fusion',**vals(fusion['oof_metrics'])});
    for q in fusion['oof_metrics'].get('per_label',[]): per.append({'experiment':'best_rescue_fusion',**q})
    pd.DataFrame(rows).sort_values(['macro_auprc','macro_auroc','macro_f1'],ascending=False).to_csv(OUT/'ALL_EXPERIMENT_RESULTS.csv',index=False); pd.DataFrame(per).to_csv(OUT/'ALL_PER_LABEL_RESULTS.csv',index=False)
    audit=pd.read_csv(OUT/'label_completeness'/'label_completeness_summary.csv'); hist=pd.read_csv(OUT/'WINNING_HISTORY_AUDIT.csv'); finalep=pd.read_csv(OUT/'FINAL_ENSEMBLE_EPOCHS.csv'); v2fusion=m(IMPROVED/'model3'/'best_fusion_config.json')['development_metrics']; roi=m(OUT/'model1'/'P2_lung_roi'/'oof_metrics.json'); clahe=m(OUT/'model1'/'P1_clahe'/'oof_metrics.json'); bal=m(OUT/'model1'/'P2_roi_balanced'/'oof_metrics.json'); hard=m(OUT/'model1'/'P2_roi_hardneg'/'oof_metrics.json'); mix=m(OUT/'model1'/'P2_roi_mixup'/'oof_metrics.json'); aligned=m(OUT/'model1'/'findings_aligned_labels'/'oof_metrics.json'); p3=m(OUT/'model1'/'P3_original_roi'/'oof_metrics.json'); p4=m(OUT/'model1'/'P4_original_clahe_roi'/'oof_metrics.json')
    def line(name,x): return f"| {name} | {x['macro_auroc']:.4f} | {x['macro_auprc']:.4f} | {x['macro_f1']:.4f} | {x['micro_f1']:.4f} | {x['macro_balanced_accuracy']:.4f} | {x['label_wise_accuracy']:.4f} |"
    table=['| Experiment | Macro AUROC | Macro AUPRC | Macro F1 | Micro F1 | Balanced accuracy | Label-wise accuracy |','|---|---:|---:|---:|---:|---:|---:|']+[line(r['experiment'],r) for r in pd.DataFrame(rows).to_dict('records')]
    rates='\n'.join(f"- {r.label}: {int(r.problems_negative_findings_explicit_positive)} explicit missing positives; {float(r.missing_positive_rate_among_problems_negative):.2%} of Problems-negative cases" for _,r in audit.iterrows())
    lines=['# IU Paired Rescue Results','',
      'All selection and threshold/alpha tuning used the locked DEVELOPMENT cohort and its existing five OOF folds only. Previously reported final-test labels and Findings were not used or re-evaluated. V1/V2 artifacts were preserved.','',
      '## Winning-history audit','',f"Final V2 ensemble retraining ran exactly 16 epochs per seed (42, 123, 2026): 2 head + 8 late-block + 6 full, 48 aggregate seed-epochs. Across 15 winning OOF seed/fold runs, actual stopping ranged {hist.stopping_epoch.min()}–{hist.stopping_epoch.max()} epochs and best epochs ranged {hist.best_epoch.min()}–{hist.best_epoch.max()}.",f"Only {int(hist.best_at_stopping_epoch.sum())}/15 folds had their best macro AUPRC at the stopping epoch; {int(hist.still_improving_at_termination.sum())}/15 met the frozen still-improving rule. The histories therefore mostly plateaued/overfit. Rescue training was not extended blindly to 80 epochs.",'',
      '## Label completeness audit','',rates,'',f"Frozen >=2% rule flagged: {', '.join(audit[audit.substantial_by_frozen_rule.astype(str).str.lower()=='true'].label)}. A separate findings-aligned-label model was run and underperformed (macro AUPRC {aligned['macro_auprc']:.4f}); primary Problems labels were never rewritten.",'',
      '## All development OOF results','',*table,'',
      'Full per-label AUROC, AUPRC, F1, precision, recall, specificity, accuracy, prevalence, predicted-positive rate, FP, and FN are in `ALL_PER_LABEL_RESULTS.csv`.','',
      '## Controlled effects','',
      f"- CLAHE: macro AUPRC {clahe['macro_auprc']:.4f}; worse than V2 {m(OUT/'model1'/'V2_best'/'oof_metrics.json')['macro_auprc']:.4f}.",
      f"- Lung ROI only: macro AUPRC {roi['macro_auprc']:.4f}; did not beat full-image V2. Masks came from TorchXRayVision PSPNet, both lung fields plus a 10% safe margin; the full-image branch was retained in fusion.",
      f"- Original + ROI: macro AUPRC {p3['macro_auprc']:.4f}, showing small complementary benefit.",
      f"- Original + CLAHE + ROI: macro AUPRC {p4['macro_auprc']:.4f}; CLAHE received zero/near-zero weight at the development optimum and did not add meaningful benefit.",
      f"- Positive-aware sampling on ROI: standalone macro AUPRC {bal['macro_auprc']:.4f}; its three-seed ROI ensemble was weak alone but complementary to V2.",
      f"- Hard-negative ROI training: macro AUPRC {hard['macro_auprc']:.4f}; rejected because it did not improve selection metrics.",
      f"- MixUp ROI: macro AUPRC {mix['macro_auprc']:.4f}; rejected.",
      f"- Best rescue image: V2 + positive-aware ROI three-seed ensemble, macro AUROC {image['macro_auroc']:.4f}, macro AUPRC {image['macro_auprc']:.4f}, macro F1 {image['macro_f1']:.4f}.",'',
      '## Model-3 rescue','',f"Using the unchanged improved word+character TF-IDF branch, development retuning selected alpha={fusion['alpha_image']:.3f} image. Fusion macro AUROC {fusion['oof_metrics']['macro_auroc']:.4f}, macro AUPRC {fusion['oof_metrics']['macro_auprc']:.4f}, macro F1 {fusion['oof_metrics']['macro_f1']:.4f}, micro F1 {fusion['oof_metrics']['micro_f1']:.4f}. It slightly improves AUPRC over image-only but does not justify a claim of final-test improvement because no final test was run.",'',
      '## Secondary normal/abnormal task','',f"Separate DEVELOPMENT OOF task—normal versus any selected abnormal: accuracy {binary['accuracy']:.4f}, balanced accuracy {binary['balanced_accuracy']:.4f}, sensitivity {binary['sensitivity']:.4f}, specificity {binary['specificity']:.4f}, F1 {binary['f1']:.4f}, AUROC {binary['auroc']:.4f}, AUPRC {binary['auprc']:.4f}. This binary accuracy is not the 10-label accuracy.",'',
      '## Conclusion','',f"Best rescue development result: macro AUROC {fusion['oof_metrics']['macro_auroc']:.4f}, macro AUPRC {fusion['oof_metrics']['macro_auprc']:.4f}, macro F1 {fusion['oof_metrics']['macro_f1']:.4f}, micro F1 {fusion['oof_metrics']['micro_f1']:.4f}, balanced accuracy {fusion['oof_metrics']['macro_balanced_accuracy']:.4f}, label-wise accuracy {fusion['oof_metrics']['label_wise_accuracy']:.4f}. The gain over V2 development fusion macro AUPRC ({v2fusion['macro_auprc']:.4f}) is modest. No 95% claim is made and no final-test claim is made."]
    (OUT/'RESCUE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); atomic_json(OUT/'BEST_RESCUE_CONFIG.json',{'best_image':'V2_plus_roi_balanced_ensemble','best_image_metrics':image,'best_fusion':fusion,'secondary_binary':binary,'selection':'DEVELOPMENT OOF only','previous_final_test_used':False,'final_test_evaluated':False})
    registry=[]
    for r in rows: registry.append({'experiment_id':r['experiment'],'scope':'DEVELOPMENT OOF only','macro_auroc':r['macro_auroc'],'macro_auprc':r['macro_auprc'],'macro_f1':r['macro_f1'],'micro_f1':r['micro_f1'],'balanced_accuracy':r['macro_balanced_accuracy'],'label_wise_accuracy':r['label_wise_accuracy'],'status':'completed','final_test_metrics':''})
    pd.DataFrame(registry).to_csv(OUT/'EXPERIMENT_REGISTRY.csv',index=False); state('complete',completed='rescue_report',best_image='V2_plus_roi_balanced_ensemble',best_fusion_alpha=fusion['alpha_image'],final_test_evaluated=False,final_report=str(OUT/'RESCUE_RESULTS.md'))
if __name__=='__main__': main()
