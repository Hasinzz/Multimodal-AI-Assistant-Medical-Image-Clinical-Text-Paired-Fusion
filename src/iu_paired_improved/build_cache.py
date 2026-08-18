import json

import cv2
import numpy as np
import pandas as pd

from .config import IMAGES, OUT, V1_OUT, setup

def main():
    setup(); out=OUT/'cache'; out.mkdir(parents=True,exist_ok=True); target=out/'images_448.npy'; idxpath=out/'images_448_index.json'
    if target.exists() and idxpath.exists(): return
    cohort=pd.read_csv(V1_OUT/'cohort.csv'); names=sorted(set('|'.join(cohort.image_filenames).split('|'))); arr=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(len(names),448,448)); index={}
    for i,fn in enumerate(names):
        im=cv2.imread(str(IMAGES/fn),cv2.IMREAD_GRAYSCALE)
        if im is None: raise RuntimeError(f'unreadable {fn}')
        h,w=im.shape; s=min(h,w); im=im[(h-s)//2:(h+s)//2,(w-s)//2:(w+s)//2]; arr[i]=cv2.resize(im,(448,448),interpolation=cv2.INTER_AREA); index[fn]=i
    arr.flush(); idxpath.write_text(json.dumps(index),encoding='utf-8')
    (out/'README.txt').write_text('Deterministic uint8 grayscale center-square 448x448 cache of eligible IU images; source filenames mapped in images_448_index.json. Used only to eliminate repeated PNG decoding/resizing.\n',encoding='utf-8')
if __name__=='__main__': main()
