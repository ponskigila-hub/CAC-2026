"""Retrospective balancing study, keeping original local holdout and demand profiles."""
import zipfile,json
import numpy as np,pandas as pd,joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix
from .core import ROOT,skills,features,label,score
from .train import pipe,profiles,ovmat,measure
R=ROOT/'data/raw/balancing';O=ROOT/'results';SW='Software Development'

def prepare(d):
    with zipfile.ZipFile(R/'IT Job Description Role Classification Dataset wit.zip') as z:
        raw=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('10461.csv'))))
    mapping={'Network Engineering':'Infrastructure & Security','Cybersecurity':'Infrastructure & Security','Data Analytics':'Data & AI','Data Science':'Data & AI','UI/UX Design':'Design'}
    frames=[];audit=[]
    def add(name,t,career,text):
        q=pd.DataFrame({'title':t,'career':career,'text':text}).dropna(subset=['career']).copy();q=q[q.career.ne(SW)];n=len(q);q['skills']=q.text.map(skills);q['x']=q.skills.map(features);q=q[q.x.ne('')];recognized=len(q);q=q[~q.x.isin(d.x)].drop_duplicates(['career','x']);q['source']=name;q['group']=q.x;frames.append(q);audit.append({'source':name,'minority_candidates':n,'recognized':recognized,'novel_profiles_before_cross_source_dedup':len(q)})
    for path in sorted(R.glob('*.csv')):
        q=pd.read_csv(path);t='title' if 'title' in q else 'Job Title';x='skills' if 'skills' in q else 'Job Description';add(path.stem,q[t],q[t].map(label),q[x])
    add('International role dataset',raw.label,raw.label.map(mapping),raw.jobdescription)
    aug=pd.concat(frames,ignore_index=True).drop_duplicates(['career','x']);ambiguous=aug.groupby('x').career.nunique();aug=aug[~aug.x.isin(ambiguous[ambiguous>1].index)].copy()
    pd.DataFrame(audit).to_csv(O/'balance_data_audit.csv',index=False)
    aug.drop(columns='text').assign(skills=aug.skills.map('|'.join)).to_csv(ROOT/'data/processed/balance_augmentation.csv',index=False)
    return aug

METHODS=[('Original Hybrid',False,'none'),('Class weighting',False,'weight'),('Random oversampling',False,'over'),('Random undersampling',False,'under'),('Selective augmentation',True,'none'),('Augmentation + class weighting',True,'weight'),('Augmentation + oversampling',True,'over'),('Augmentation + undersampling',True,'under'),('Moderate class weighting',False,'moderate'),('Augmentation + moderate weighting',True,'moderate')]

def fit(train,aug,use_aug,method,seed):
    a=pd.concat([train,aug],ignore_index=True) if use_aug else train.reset_index(drop=True)
    counts=a.career.value_counts();rng=np.random.default_rng(seed)
    if method in ('over','under'):
        target=counts.max() if method=='over' else counts.min()
        ix=np.concatenate([np.concatenate([g.index.to_numpy(),rng.choice(g.index,target-len(g),replace=True)]) if len(g)<target else rng.choice(g.index,target,replace=False) for _,g in a.groupby('career')]);a=a.loc[ix].reset_index(drop=True)
    weights=None
    if method in ('weight','moderate'):
        weights=a.career.map(len(a)/(len(counts)*counts)).to_numpy();weights=weights**(.5 if method=='moderate' else 1);weights=weights/weights.mean()
    model=pipe(MultinomialNB(alpha=.1));model.fit(a.x,a.career,**({'multinomialnb__sample_weight':weights} if weights is not None else {}));return model,a.career.value_counts().to_dict()

def metrics(y,p,cls):return {**measure(y,p,cls),'balanced_accuracy':balanced_accuracy_score(y,cls[p.argmax(1)])}

def run():
    d=pd.read_csv(ROOT/'data/processed/jobs.csv');d['skills']=d.skills.str.split('|');dev=d[d.split.eq('development')].reset_index(drop=True);test=d[d.split.eq('test')].reset_index(drop=True);aug=prepare(d);cls=np.array(sorted(d.career.unique()));assert not set(aug.x)&set(d.x);assert not set(dev.x)&set(test.x)
    folds=list(StratifiedGroupKFold(4,shuffle=True,random_state=43).split(dev,dev.career,dev.group));rows=[];dist=[];reports=[];models={};preds={};prof=profiles(dev);ovtest=ovmat(prof,test.skills,cls)
    for name,use_aug,method in METHODS:
        oof=np.zeros((len(dev),len(cls)))
        for i,(a,b) in enumerate(folds):
            tr,val=dev.iloc[a],dev.iloc[b];m,_=fit(tr,aug,use_aug,method,42+i);oof[b]=.5*score(m,val.x)+.5*ovmat(profiles(tr),val.skills,cls)
        rows.append({'method':name,'split':'grouped CV',**metrics(dev.career,oof,cls)})
        print(name,rows[-1]['macro_f1'],flush=True)
        m,c=fit(dev,aug,use_aug,method,42);models[name]=m;dist.extend({'method':name,'career':k,'training_rows':v} for k,v in c.items());p=.5*score(m,test.x)+.5*ovtest;preds[name]=p
        rows.append({'method':name,'split':'test',**metrics(test.career,p,cls)});cr=classification_report(test.career,cls[p.argmax(1)],labels=cls,output_dict=True,zero_division=0);reports.extend({'method':name,'career':c,**cr[c]} for c in cls)
    table=pd.DataFrame(rows);winner=table[table.split.eq('grouped CV')].sort_values(['macro_f1','balanced_accuracy'],ascending=False).iloc[0].method
    table.to_csv(O/'balance_metrics.csv',index=False);pd.DataFrame(reports).to_csv(O/'balance_class_report.csv',index=False);pd.DataFrame(dist).to_csv(O/'balance_training_distribution.csv',index=False)
    alternative=table[(table.split.eq('grouped CV')) & (table.method.ne('Original Hybrid'))].sort_values(['macro_f1','balanced_accuracy'],ascending=False).iloc[0].method
    summary={'selected':winner,'alternative':alternative,'augmentation_profiles':len(aug),'augmentation_counts':aug.career.value_counts().to_dict(),'development':len(dev),'test':len(test),'note':'Retrospective comparison; original test already used in earlier work. Select on grouped CV only. Demand profiles remain original local development.'};(O/'balance_summary.json').write_text(json.dumps(summary,indent=2))
    b=joblib.load(ROOT/'models/bundle.joblib');b['models']['Balancing experiment']=models[alternative];b['balance_selected']=alternative;joblib.dump(b,ROOT/'models/bundle.joblib',compress=3)
    name='Balancing experiment';p=preds[alternative];m=pd.read_csv(O/'metrics.csv');m=m[m.model.ne(name)];m=pd.concat([m,table[table.method.eq(alternative)].drop(columns=['method','balanced_accuracy']).assign(model=name)],ignore_index=True);m.to_csv(O/'metrics.csv',index=False)
    report=pd.read_csv(O/'class_report.csv');report=report[report.model.ne(name)];report=pd.concat([report,pd.DataFrame(reports).query('method == @alternative').drop(columns='method').assign(model=name)],ignore_index=True);report.to_csv(O/'class_report.csv',index=False)
    pd.DataFrame(confusion_matrix(test.career,cls[p.argmax(1)],labels=cls),index=cls,columns=cls).to_csv(O/f'cm_{name}.csv')
    print(summary,flush=True);print(table.to_string(index=False),flush=True)
if __name__=='__main__':run()
