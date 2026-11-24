"""Load only bundled, manifest-verified V9 artifacts (never uploaded pickle files)."""
import hashlib,json,re
from pathlib import Path
import joblib
DATA=Path(__file__).resolve().parents[1]/'data'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_manifest(data_dir=DATA):
    data_dir=Path(data_dir); m=json.loads((data_dir/'v9_manifest.json').read_text())
    if m.get('version')!='V9': raise ValueError('Incorrect manifest version.')
    return m


def verify_dataset(data_dir=DATA):
    data_dir=Path(data_dir); m=load_manifest(data_dir)
    if sha256(data_dir/'foodshock_asean_model_ready.csv')!=m['dataset_sha256']:
        raise ValueError('Dataset does not match the frozen V9 benchmark. Re-run src/run_v9.py.')
    return m


def load_bundle(fold=None,data_dir=DATA,verify_data=True):
    data_dir=Path(data_dir); m=verify_dataset(data_dir) if verify_data else load_manifest(data_dir)
    if fold is None: rel='v9_official_bundle.joblib'
    else:
        fold=str(fold)
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',fold) or fold not in m['cutoffs']: raise ValueError('Unknown model fold.')
        rel=f'fold_models/{fold}.joblib'
    path=data_dir/rel
    if m['model_sha256'].get(rel)!=sha256(path): raise ValueError('Model file does not match V9 manifest.')
    return joblib.load(path)


def verify_results(data_dir=DATA):
    data_dir=Path(data_dir); m=verify_dataset(data_dir)
    for name,digest in m.get('artifact_sha256',{}).items():
        if sha256(data_dir/name)!=digest: raise ValueError(f'Result artifact mismatch: {name}. Re-run the benchmark.')
    return m
