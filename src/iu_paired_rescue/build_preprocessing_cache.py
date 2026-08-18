import json

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv

from .audit import state
from .config import IMPROVED,OUT,V1,setup

def main():
    setup(); state('preprocessing_cache'); out=OUT/'cache'; out.mkdir(parents=True,exist_ok=True); source=np.load(IMPROVED/'cache'/'images_448.npy',mmap_mode='r'); index=json.loads((IMPROVED/'cache'/'images_448_index.json').read_text()); names=sorted(index,key=index.get); n=len(names)
    clahe_path=out/'images_448_clahe.npy'; roi_path=out/'images_448_lung_roi.npy'; mask_path=out/'lung_masks_448.npy'
    if all(p.exists() for p in [clahe_path,roi_path,mask_path,out/'lung_roi_metadata.csv']): return
    clahe=np.lib.format.open_memmap(clahe_path,mode='w+',dtype=np.uint8,shape=(n,448,448)); roi=np.lib.format.open_memmap(roi_path,mode='w+',dtype=np.uint8,shape=(n,448,448)); masks=np.lib.format.open_memmap(mask_path,mode='w+',dtype=np.uint8,shape=(n,448,448)); alg=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); model=xrv.baseline_models.chestx_det.PSPNet().cuda().eval(); rows=[]
    for start in range(0,n,16):
        batch=np.asarray(source[start:start+16]); x=torch.from_numpy(batch.copy()).float().unsqueeze(1).cuda()/255.; x=(x*2-1)*1024
        with torch.no_grad(): logits=model(x); seg=logits.argmax(1).cpu().numpy()
        for k,img in enumerate(batch):
            i=start+k; m=np.isin(seg[k],[4,5]).astype(np.uint8); m=cv2.resize(m,(448,448),interpolation=cv2.INTER_NEAREST); ys,xs=np.where(m>0); valid=len(xs)>=448*448*.08
            if valid:
                x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max(); mx=max(12,int((x1-x0)*.10)); my=max(12,int((y1-y0)*.10)); x0=max(0,x0-mx); x1=min(447,x1+mx); y0=max(0,y0-my); y1=min(447,y1+my); crop=img[y0:y1+1,x0:x1+1]
            else: x0=y0=0; x1=y1=447; crop=img
            masks[i]=m*255; roi[i]=cv2.resize(crop,(448,448),interpolation=cv2.INTER_AREA); clahe[i]=alg.apply(img); rows.append({'filename':names[i],'mask_valid':valid,'mask_fraction':float(m.mean()),'x0':x0,'y0':y0,'x1':x1,'y1':y1,'fallback_full_image':not valid})
    clahe.flush(); roi.flush(); masks.flush(); pd.DataFrame(rows).to_csv(out/'lung_roi_metadata.csv',index=False); (out/'index.json').write_text(json.dumps(index),encoding='utf-8'); atomic={'segmenter':'torchxrayvision 1.5.2 chestx_det.PSPNet','weights':'pspnet_chestxray_best_model_4.pth','weight_sha256':'019B167EAC6B729FC1BB92BBBC185FC1730AAA65819F4E3FE718186CADC044FC','targets':['Left Lung','Right Lung'],'input':'center-square grayscale, [-1024,1024], model-resized to 512','mask':'argmax classes 4/5','roi':'both lung fields bbox + 10% margin (minimum 12 px); full-image fallback below 8% mask area','clahe':'OpenCV clipLimit=2.0 tileGridSize=8x8'}; (out/'preprocessing_provenance.json').write_text(json.dumps(atomic,indent=2),encoding='utf-8'); state('cache_complete',completed='preprocessing_cache',valid_masks=int(pd.DataFrame(rows).mask_valid.sum()),fallback_masks=int((~pd.DataFrame(rows).mask_valid).sum()))
if __name__=='__main__': main()
