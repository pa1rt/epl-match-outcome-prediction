import pandas as pd,numpy as np
from collections import defaultdict,deque
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path('/mnt/data')
d=pd.read_csv(p/'EPL_Combined_1900_Matches.csv');d['Date']=pd.to_datetime(d.Date,dayfirst=True);d=d.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
# use only prior dates: group fixtures by date, update history after feature extraction for entire date
hist=defaultdict(list); rows=[]
for date,g in d.groupby('Date',sort=True):
 for _,m in g.iterrows():
  r={'Date':date,'Season':m.Season,'HomeTeam':m.HomeTeam,'AwayTeam':m.AwayTeam,'FTR':m.FTR}
  for side,team in [('home',m.HomeTeam),('away',m.AwayTeam)]:
   previous=hist[team][-5:]
   r[f'{side}_n']=len(previous)
   for field in ['points','gf','ga','win']:
    r[f'{side}_{field}_5']=float(np.mean([z[field] for z in previous])) if previous else np.nan
  for field in ['points','gf','ga','win']:
   r[f'diff_{field}_5']=r[f'home_{field}_5']-r[f'away_{field}_5']
  rows.append(r)
 for _,m in g.iterrows():
  hg,ag=int(m.FTHG),int(m.FTAG)
  hist[m.HomeTeam].append({'points':3 if hg>ag else 1 if hg==ag else 0,'gf':hg,'ga':ag,'win':int(hg>ag)})
  hist[m.AwayTeam].append({'points':3 if ag>hg else 1 if hg==ag else 0,'gf':ag,'ga':hg,'win':int(ag>hg)})
f=pd.DataFrame(rows)
# drop matches lacking 5 previous matches per side; data only previous matches, across season boundaries
f=f[(f.home_n>=5)&(f.away_n>=5)].copy()
features=[c for c in f if c.startswith(('home_','away_','diff_')) and c not in ['HomeTeam','AwayTeam'] and c not in ['home_n','away_n']]
train=f[f.Date<'2025-08-01'];test=f[f.Date>='2025-08-01']
Xtr,Xte=train[features],test[features];ytr,yte=train.FTR,test.FTR
models={'Majority-class baseline':DummyClassifier(strategy='most_frequent'),'Multinomial logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)),'Random forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=5,random_state=42,n_jobs=-1)}
res=[]
for name,model in models.items():
 model.fit(Xtr,ytr);pred=model.predict(Xte)
 res.append({'Model':name,'Test accuracy':round(accuracy_score(yte,pred),4),'Macro F1':round(f1_score(yte,pred,average='macro'),4),'Correct':int((pred==yte).sum()),'Test matches':len(test)})
 print(name,res[-1]);print(classification_report(yte,pred,zero_division=0))
 if name=='Multinomial logistic regression':
  cm=confusion_matrix(yte,pred,labels=['H','D','A']);pd.DataFrame(cm,index=['Actual home','Actual draw','Actual away'],columns=['Pred home','Pred draw','Pred away']).to_csv(p/'EPL_Logistic_Confusion_Matrix.csv')
  fig,ax=plt.subplots(figsize=(6,4.6));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(3),['Home','Draw','Away']);ax.set_yticks(range(3),['Home','Draw','Away']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title('Logistic regression · 2025–26 test matches')
  for i in range(3):
   for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=13)
  fig.tight_layout();fig.savefig(p/'EPL_Confusion_Matrix.png',dpi=170);plt.close(fig)
pd.DataFrame(res).to_csv(p/'EPL_Model_Results.csv',index=False)
f.to_csv(p/'EPL_PreMatch_Features.csv',index=False)
fig,ax=plt.subplots(figsize=(8.8,4.8));names=['Baseline','Logistic regression','Random forest'];acc=[r['Test accuracy']*100 for r in res];bars=ax.bar(names,acc,color=['#8d99ae','#1f77b4','#4b9b81']);ax.set_ylim(0,65);ax.set_ylabel('Accuracy (%)');ax.set_title('Out-of-time test: 2025–26 Premier League season');ax.bar_label(bars,fmt='%.1f%%',padding=3);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(p/'EPL_Model_Comparison.png',dpi=180);plt.close(fig)
prs=Presentation(p/'DS670_Midterm_With_Real_EDA.pptx');print('slides',len(prs.slides))
# append clear real results
for title,bullets,img in [
 ('First predictive models — real results', [f"Chronological split: {len(train):,} training matches (2021–25); {len(test):,} test matches (2025–26)","Only prior-match statistics used: last-five-match goals, points, wins and goal differences",*[f"{r['Model']}: accuracy {r['Test accuracy']:.1%}; macro F1 {r['Macro F1']:.3f}" for r in res],"Preliminary models only; further tuning and analysis planned"],p/'EPL_Model_Comparison.png'),
 ('How predictions were evaluated', ['Features for each fixture use only matches played before that fixture date','Require five previous games for each team; initial fixtures without enough history are excluded','Train on earlier seasons and test on the full 2025–26 season','Macro F1 reflects performance across all three outcomes, including draws','Next: analyze draw errors, improve features, tune models without test leakage'],p/'EPL_Confusion_Matrix.png')]:
 slide=prs.slides.add_slide(prs.slide_layouts[6]);bg=slide.background.fill;bg.solid();bg.fore_color.rgb=RGBColor(11,23,43)
 tb=slide.shapes.add_textbox(Inches(.55),Inches(.35),Inches(12.2),Inches(.65));tf=tb.text_frame;tf.text=title;tf.paragraphs[0].font.size=Pt(29);tf.paragraphs[0].font.bold=True;tf.paragraphs[0].font.color.rgb=RGBColor(255,255,255)
 for i,b in enumerate(bullets):
  box=slide.shapes.add_textbox(Inches(.7),Inches(1.2+i*.66),Inches(6.3),Inches(.6));t=box.text_frame;t.word_wrap=True;t.text='• '+b;t.paragraphs[0].font.size=Pt(16);t.paragraphs[0].font.color.rgb=RGBColor(226,235,246)
 slide.shapes.add_picture(str(img),Inches(7.25),Inches(1.35),width=Inches(5.4))
prs.save(p/'DS670_Midterm_With_Models.pptx')
print('features',len(features),'train',len(train),'test',len(test))
