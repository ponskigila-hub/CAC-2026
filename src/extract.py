"""Rebuild regional source CSV from original annual Kaggle files."""
from pathlib import Path
import argparse, hashlib
import pandas as pd
from src.forecast import ROOT

def extract(folder):
    files=[Path(folder)/f'wfp_food_prices_global_{y}.csv' for y in range(2015,2026)]
    parts=[];audit=[]
    for f in files:
        x=pd.read_csv(f,low_memory=False);y=x[x.countryiso3.eq('IDN')];parts.append(y)
        with f.open('rb') as fd: digest=hashlib.file_digest(fd,'sha256').hexdigest()
        audit.append(dict(file=f.name,global_rows=len(x),indonesia_rows=len(y),sha256=digest))
    pd.concat(parts,ignore_index=True).to_csv(ROOT/'data/raw/indonesia.csv.gz',index=False,compression='gzip');pd.DataFrame(audit).to_csv(ROOT/'data/raw/source_manifest.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw_dir');extract(p.parse_args().raw_dir)
