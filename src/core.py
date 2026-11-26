from pathlib import Path
import re, joblib, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
# Curated technical vocabulary; aliases are explicit, no role names as features.
ALIASES={
'Python':['python'],'Java':['java'],'JavaScript':['javascript','java script','js'],'TypeScript':['typescript'],'C#':['c#','c sharp'],'C++':['c++'],'PHP':['php'],'Go':['golang','go language'],'Ruby':['ruby'],'Kotlin':['kotlin'],'Swift':['swift'],'Dart':['dart'],'HTML':['html5','html'],'CSS':['css3','css'],'React':['react.js','reactjs','react js','react'],'React Native':['react native'],'Vue':['vue.js','vuejs','vue js','vue'],'Angular':['angularjs','angular js','angular'],'Next.js':['next.js','nextjs'],'Node.js':['node.js','nodejs','node js'],'Express':['express.js','expressjs'],'Laravel':['laravel'],'Django':['django'],'Flask':['flask'],'Spring':['spring boot','spring framework','spring'],'ASP.NET':['asp.net','.net','dotnet'],'Flutter':['flutter'],'Android SDK':['android sdk'],'iOS SDK':['ios sdk'],'SQL':['sql'],'MySQL':['mysql'],'PostgreSQL':['postgresql','postgres'],'MongoDB':['mongodb','mongo db'],'Redis':['redis'],'SQL Server':['sql server','mssql'],'Oracle':['oracle'],'REST API':['restful api','rest api','restful','rest apis','rest'],'GraphQL':['graphql'],'Git':['github','gitlab','git'],'Docker':['docker'],'Kubernetes':['kubernetes','k8s'],'AWS':['amazon web services','aws'],'Azure':['azure'],'GCP':['google cloud','gcp'],'Linux':['linux'],'CI/CD':['ci/cd','continuous integration','continuous delivery'],'Jenkins':['jenkins'],'Microservices':['microservices','microservice'],'Agile':['agile','scrum'],'Testing':['unit testing','software testing','automated testing','selenium','junit','testing'],'Figma':['figma'],'Adobe Photoshop':['photoshop'],'Adobe Illustrator':['illustrator'],'Adobe InDesign':['indesign'],'UI design':['user interface design','ui design'],'UX research':['user research','ux research'],'Prototyping':['prototyping','wireframe','wireframing'],'SEO':['seo','search engine optimization'],'SEM':['sem','search engine marketing'],'Google Analytics':['google analytics'],'Google Ads':['google ads','adwords'],'Social media':['social media'],'Content marketing':['content marketing'],'Copywriting':['copywriting'],'Meta Ads':['facebook ads','meta ads'],'Excel':['excel'],'Power BI':['power bi'],'Tableau':['tableau'],'Pandas':['pandas'],'Machine learning':['machine learning'],'TensorFlow':['tensorflow'],'PyTorch':['pytorch'],'Spark':['apache spark','spark'],'Airflow':['apache airflow','airflow'],'ETL':['etl'],'Data modeling':['data modeling','data modelling'],'WordPress':['wordpress'],'jQuery':['jquery'],'Bootstrap':['bootstrap'],'Tailwind':['tailwind'],'Firebase':['firebase'],'XML':['xml'],'JSON':['json'],'Jira':['jira'],'Odoo':['odoo'],'SAP':['sap'],'Cybersecurity':['cybersecurity','cyber security'],'Networking':['networking','tcp/ip'],'Communication':['communication skills','komunikasi'],'Teamwork':['teamwork','team work','kerja sama tim'],'Problem solving':['problem solving','problem-solving']}
PATTERNS={k:re.compile(r'(?<![\w])(?:'+ '|'.join(re.escape(v) for v in sorted(vals,key=len,reverse=True))+r')(?![\w])',re.I) for k,vals in ALIASES.items()}
def extract(text):
    text=str(text); found={k for k,p in PATTERNS.items() if p.search(text)}
    if 'React Native' in found and not re.search(r'\breact\b(?!\s+native)',text,re.I):found.discard('React')
    if 'SQL Server' in found and not re.search(r'\bsql\b(?!\s+server)',text,re.I):found.discard('SQL')
    return sorted(found)
def features(skills):return ' '.join('skill_'+s.encode('utf-8').hex() for s in sorted(set(skills)))
def role(title):
    t=str(title).lower()
    if re.search(r'bangkok|singapore based|relocation|malaysia based',t):return None
    for name,p in [('Full-stack',r'full[\s-]*stack'),('Front-end',r'front[\s-]*end|angular|react(?! native)|vue[.\s]'),('Back-end',r'back[\s-]*end'),('Mobile',r'\bmobile\b|\bandroid\b|\bios\b|flutter|react native'),('Digital marketing',r'digital market|digital advertis|seo specialist|social media specialist|performance market'),('Graphic design',r'graphic design|desain grafis|desainer grafis|visual design'),('Web development',r'web developer|web programmer|wordpress developer'),('Software engineering',r'software (engineer|developer)|\bprogrammer\b|java developer|\.net developer|dotnet developer|php developer|golang developer|it developer|c# developer')]:
        if re.search(p,t):return name
    return None


rules=[('Data & AI',r'data (analyst|scientist|engineer|architect)|machine learning|\bmlops\b|business intelligence|big data|data analytics'),('Infrastructure & Security',r'devops|cloud (engineer|architect)|network (engineer|administrator)|system[s]? administrator|security (engineer|analyst)|cybersecurity|database administrator|\bdba\b|it support|technical support'),('Design',r'graphic design|ui[/ ]?ux|ux designer|ui designer|visual design|desain grafis'),('Digital Marketing',r'digital market|seo specialist|social media specialist|performance market'),('Software Development',r'developer|programmer|software engineer|frontend|backend|front.end|back.end|full.stack|software development|software tester|software quality assurance')]
def label(t):
 for c,p in rules:
  if re.search(p,str(t),re.I):return c
 return None
extra={'Data analysis':r'data analytics|analisis data','Statistics':r'statistic|statistik','Data visualization':r'data visuali[sz]ation|visualisasi data','Hadoop':r'\bhadoop\b','Scikit-learn':r'scikit.learn|sklearn','LLM':r'large language model|\bllm\b','Cloud computing':r'cloud computing','Network security':r'network security|firewall','Windows Server':r'windows server','CCNA':r'\bccna\b','Active Directory':r'active directory','Adobe XD':r'adobe xd','Canva':r'\bcanva\b','Digital advertising':r'digital advertising|iklan digital','Content creation':r'content creation|content creator','R programming':r'\br programming\b|\br language\b','Data warehouse':r'data warehous','IT troubleshooting':r'troubleshooting','Virtualization':r'virtuali[sz]ation|vmware','Automation':r'automation|otomasi'}
def skills(t):return sorted(set(extract(t))|{k for k,p in extra.items() if re.search(p,str(t),re.I)})

ALIASES.update({k:[] for k in extra})
def load():return joblib.load(ROOT/'models/bundle.joblib')
def score(m,x):
    if hasattr(m,'predict_proba'):return m.predict_proba(x)
    v=m.decision_function(x);v=np.exp(v-v.max(axis=1,keepdims=True));return v/v.sum(axis=1,keepdims=True)
def overlap(profiles,skills,classes):
    s=set(skills);return np.array([sum(v for k,v in profiles[c].items() if k in s)/max(sum(profiles[c].values()),1e-12) for c in classes])
def recommend(bundle,skills,model=None):
    known=sorted(set(skills)&set(bundle['vocabulary']));name=model or bundle['best'];m=bundle['models'][bundle['base'] if name=='Hybrid' else name];classes=m.classes_
    if not known:return pd.DataFrame(columns=['Career','Model score','Skill coverage'])
    p=score(m,[features(known)])[0];o=overlap(bundle['profiles'],known,classes)
    if name in ('Hybrid','Balancing experiment'):p=bundle['alpha']*p+(1-bundle['alpha'])*(o/o.sum() if o.sum() else np.ones(len(o))/len(o))
    return pd.DataFrame({'Career':classes,'Model score':p,'Skill coverage':o}).sort_values('Model score',ascending=False,ignore_index=True)
def gap(bundle,career,skills):
    return pd.DataFrame([{'Skill':s,'Demand share':v,'Have':s in skills} for s,v in bundle['profiles'][career].items()]).sort_values('Demand share',ascending=False,ignore_index=True)
def explain(bundle,skills,career,model=None):
    base=recommend(bundle,skills,model).set_index('Career')['Model score'][career];rows=[]
    for s in skills:
        r=recommend(bundle,[k for k in skills if k!=s],model);rows.append({'Skill':s,'Score change when removed':base-r.set_index('Career')['Model score'][career] if len(r) else np.nan})
    return pd.DataFrame(rows).sort_values('Score change when removed',ascending=False)
# Specializations use explicit within-family rules and minimum development support.
SPEC_RULES={
'Data & AI':[('Data science',r'data scientist|machine learning engineer|ai engineer'),('Data engineering',r'data engineer|data architect|big data'),('Data analytics',r'data analyst|data analytics|business intelligence|bi analyst')],
'Infrastructure & Security':[('Cybersecurity',r'cyber\s*security|security (engineer|analyst)|penetration tester'),('Cloud & DevOps',r'devops|cloud (engineer|architect)'),('Network & systems',r'network|system[s]? administrator|database administrator|\bdba\b|it support|technical support')],
'Design':[('Graphic design',r'graphic design|desain grafis|visual design'),('UI/UX',r'ui[/ ]?ux|ux designer|ui designer')],
'Digital Marketing':[('Digital marketing',r'digital market|digital advertis|performance market'),('SEO',r'\bseo\b'),('Social media',r'social media')],
'Software Development':[('Full-stack',r'full[\s-]*stack'),('Front-end',r'front[\s-]*end|react developer|angular developer|vue developer'),('Back-end',r'back[\s-]*end|php developer|java developer|golang developer'),('Mobile',r'\bmobile\b|android|\bios\b|flutter|react native'),('QA & testing',r'software tester|software quality assurance'),('General software',r'web developer|web programmer|programmer|software engineer|software developer|developer')]}
def specialization(career,title):
    for name,p in SPEC_RULES.get(career,[]):
        if re.search(p,str(title),re.I):return name
    return None

def learning_rank(bundle,career,known):
    """Rank missing skills using selected development-only policy; coverage is separate."""
    if 'gap_matrix' in bundle:
        from .gap_rank import ranking
        return ranking(bundle,career,known)
    profile=bundle['profiles'][career];known=set(known);policy=bundle.get('gap_policy','frequency');rows=[]
    for skill,freq in profile.items():
        if skill in known:continue
        value=freq
        if policy!='frequency':
            counts=bundle['cooccurrence'][career];den=bundle['skill_counts'][career];cond=[counts.get(tuple(sorted((s,skill))),0)/max(den.get(s,0),1) for s in known if den.get(s,0)>=3];conditional=float(np.mean(cond)) if cond else freq;w=float(policy);value=w*conditional+(1-w)*freq
        rows.append({'Skill':skill,'Demand share':freq,'Learning priority score':value})
    return pd.DataFrame(rows,columns=['Skill','Demand share','Learning priority score']).sort_values('Learning priority score',ascending=False,ignore_index=True)
