"""Reproducible, read-only audit and publication figures for the current results.

Run with uv run --with matplotlib --with pandas --with scipy python Figs/create_results_atlas.py.
PNG-only, no figure titles or grids, following Figs/README.md.
Results are first-valid-per-Filename to match the accompanying thesis tables.
Selection is descriptive on the test set, never a validation-selected claim.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Figs' / 'results_atlas'
OUT.mkdir(exist_ok=True)
ARTS = (3, 6, 8)
LABELS = ['Key case', 'Level 1', 'Level 2', 'Level 3']
COLORS = ['#882255', '#CC6677', '#DDCC77', '#4477AA']
MODELS = ['GPT-OSS', 'Llama 3.3']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                    'axes.spines.top': False, 'axes.spines.right': False,
                    'axes.grid': False, 'figure.facecolor': 'white',
                    'savefig.facecolor': 'white', 'axes.titlesize': 11})
CASES = {a: pd.read_pickle(ROOT / f'data/processed/article{a}/splits/comm_test.pkl')
         .set_index('Filename') for a in ARTS}
for c in CASES.values():
    assert c.index.is_unique and c.importance.isin([1, 2, 3, 4]).all()
RUNS, AUDIT, SNAPSHOT, SELECTED = {}, [], [], []


def metrics(y, p):
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y - 1, p - 1), 1)
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(4), where=cm.sum(1)>0)
    return dict(mae=float(np.mean(abs(y-p))),
                src=float(spearmanr(y, p).statistic) if len(set(p)) > 1 else np.nan,
                over=float(np.mean(p<y)), under=float(np.mean(p>y)),
                exact=float(np.mean(p==y)), low_share=float(np.mean(p==4)),
                balanced_recall=float(recall.mean()), low_recall=float(recall[3]),
                minority_recall=float(recall[:3].mean()))


def register(a, model, family, method, k, paths):
    paths = [ROOT / f'data/results/article{a}' / p for p in paths]
    path = next((p for p in paths if p.exists()), paths[0]) if paths else None
    key = (a, model, family, method, k)
    c = CASES[a]; y = c.importance.to_numpy(int)
    first, last, preds = {}, {}, defaultdict(set)
    invalid = badjson = duplicates = mismatch = 0
    digest = ''
    if path and path.exists():
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        seen = set()
        for line in raw.splitlines():
            try: r = json.loads(line)
            except (ValueError, UnicodeDecodeError): badjson += 1; continue
            fn, p = str(r.get('Filename')), r.get('prediction')
            duplicates += fn in seen; seen.add(fn)
            if fn in c.index and r.get('importance') != int(c.loc[fn, 'importance']): mismatch += 1
            if type(p) is not int or p not in (1, 2, 3, 4): invalid += 1; continue
            first.setdefault(fn, p); last[fn] = p; preds[fn].add(p)
    if family == 'Majority': first = last = {fn: 4 for fn in c.index}
    valid = len(set(first) & set(c.index)); extra = len(set(first)-set(c.index))
    complete = valid == len(c) and extra == 0 and mismatch == 0 and badjson == 0
    row = dict(article=a, model=model, family=family, method=method, k=k,
               source=str(path.relative_to(ROOT)) if path else 'majority label 4',
               sha256=digest, target=len(c), valid_unique=valid, complete=complete,
               invalid_attempts=invalid, duplicate_rows=duplicates,
               conflicting_cases=sum(len(s)>1 for s in preds.values()),
               extra_ids=extra, truth_mismatches=mismatch, malformed_lines=badjson)
    if complete or (family=='Baseline' and valid and extra==0 and mismatch==0 and badjson==0):
        index=[fn for fn in c.index if fn in first]
        y=c.loc[index,'importance'].to_numpy(int)
        p = np.array([first[fn] for fn in index]); q = np.array([last[fn] for fn in index])
        row.update(metrics(y, p)); latest = metrics(y, q)
        row.update(last_valid_mae=latest['mae'], last_valid_src=latest['src'])
        RUNS[key] = (y, p, row)
        SNAPSHOT.extend(dict(article=a,model=model,family=family,method=method,k=k,
                             Filename=fn,truth=int(t),prediction=int(v))
                        for fn,t,v in zip(index,y,p))
    AUDIT.append(row)


def collect():
    for a in ARTS:
        register(a,'Majority','Majority','Zero-shot',0,[])
        for model, tag in [('GPT-OSS','gpt-oss-120b'),('Llama 3.3','Llama-3.3-70B-Instruct')]:
            base = tag if model=='GPT-OSS' else tag+'-FP8'
            for method, prefix in [('Zero-shot','base'),('Few-shot','few_shot')]:
                register(a,model,'Baseline',method,0,[f'{prefix}_text1_test_{base}.jsonl'])
            for retr in ['faiss','bm25','tgn_kg']:
                for rr in [False,True]:
                    for k in [3,5,10]:
                        stem=f'retrieval_{retr}_k{k}'+('_rerank' if rr else '')+f'_text1_test_{tag}'
                        paths=[stem+'_static_rerun_20260925.jsonl']
                        # Article 3 temporal RR was rerun in-place with the AP-selected
                        # checkpoint (e.g. predict_rag job 10692186); old copies are .bak.
                        if not rr or (a==3 and retr=='tgn_kg'): paths.append(stem+'.jsonl')
                        register(a,model,'Experiment 1',retr+('+RR' if rr else ''),k,paths)
            for k in [3,5,10]:
                register(a,model,'Gold','gold',k,[f'retrieval_gold_k{k}_text1_test_{base}.jsonl',f'retrieval_gold_k{k}_text1_test_{tag}.jsonl'])
            ft = f'llama-v2-art{a}-e1' if model=='Llama 3.3' else f'gptoss-merged-art{a}-native'+('-a100' if a==3 else '')
            register(a,model,'Fine-tuned','Zero-shot',0,[f'base_text1_test_{ft}.jsonl'])
            for retr in ['faiss','tgn_kg'] if model=='Llama 3.3' else ['faiss']:
                for k in [3,5,10]:
                    stem=f'retrieval_{retr}_k{k}_text1_test_{ft}'
                    # Complete direct outputs supersede partial static reruns.
                    register(a,model,'Fine-tuned',retr,k,[stem+'.jsonl',stem+'_static_rerun_20260925.jsonl'])
        for k in [0,3,5,10]:
            stem='iterative_text1_test_'+(f'faiss_k{k}_' if k else '')+'gpt-oss-120b-1h100.jsonl'
            register(a,'GPT-OSS','Iterative','faiss' if k else 'Zero-shot',k,[stem])


def best(a, model, family='Experiment 1', method='faiss', criterion='mae'):
    eligible=[(key,v) for key,v in RUNS.items() if key[:3]==(a,model,family)
              and key[3]==method and np.isfinite(v[2][criterion])]
    if not eligible: return None
    key, value=min(eligible,key=lambda kv: (kv[1][2][criterion]*(1 if criterion=='mae' else -1),kv[0][4]))
    rec=dict(article=a,model=model,family=family,method=method,criterion=criterion,k=key[4],source=value[2]['source'])
    if rec not in SELECTED: SELECTED.append(rec)
    return key


def save(fig, name):
    fig.savefig(OUT / (name+'.png'),dpi=350,bbox_inches='tight')
    plt.close(fig)


def matrix(ax, key, heading):
    ax.set_title(heading, pad=10)
    if key not in RUNS:
        ax.text(.5,.5,'TBA\nIncomplete coverage',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off(); return
    y,p,r=RUNS[key]; cm=np.zeros((4,4),int);np.add.at(cm,(y-1,p-1),1)
    perc=np.divide(cm*100,cm.sum(1)[:,None],out=np.zeros((4,4)),where=cm.sum(1)[:,None]>0)
    ax.imshow(perc,vmin=0,vmax=100,cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{cm[i,j]}\n{perc[i,j]:.1f}%',ha='center',va='center',fontsize=8,
                    color='white' if perc[i,j]>52 else '#222222')
    ax.set_xticks(range(4),LABELS,rotation=35,ha='right',fontsize=8)
    ax.set_yticks(range(4),LABELS,fontsize=8)
    ax.set_xlabel('Predicted importance',fontsize=9); ax.set_ylabel('Actual importance',fontsize=9)
    src=f'{r["src"]:.3f}' if np.isfinite(r['src']) else 'undefined'
    ax.text(.5,-.47,f'MAE {r["mae"]:.3f} · SRC {src}\nOverestimated: {r["over"]:.1%} · n={len(y):,}/{r["target"]:,}',
            transform=ax.transAxes,ha='center',fontsize=9)


def confusions():
    for arts, suffix in [([3],'article3'),(ARTS,'all_articles')]:
        fig,axs=plt.subplots(len(arts),4,figsize=(13.4,4.9*len(arts)),squeeze=False)
        for i,a in enumerate(arts):
            for j,(model,meth) in enumerate([(m,c) for m in MODELS for c in ['Zero-shot','Few-shot']]):
                matrix(axs[i,j],(a,model,'Baseline',meth,0),f'Article {a} · {model}\n{meth}'+(' (FP8)' if model=='Llama 3.3' else ''))
        fig.subplots_adjust(wspace=.42,hspace=.8,bottom=.2 if len(arts)==1 else .08)
        save(fig,'baseline_confusions_'+suffix)
    for arts,suffix in [([3],'article3'),(ARTS,'all_articles')]:
        fig,axs=plt.subplots(len(arts),4,figsize=(13.4,4.9*len(arts)),squeeze=False)
        for i,a in enumerate(arts):
            key=best(a,'GPT-OSS'); k=key[4] if key else 10
            keys=[key,(a,'GPT-OSS','Iterative','faiss',k),(a,'GPT-OSS','Fine-tuned','faiss',k),
                  (a,'Llama 3.3','Fine-tuned','faiss',k)]
            for j,(key,heading) in enumerate(zip(keys,['GPT-OSS · FAISS','GPT-OSS · Iterative','GPT-OSS · Fine-tuned','Llama 3.3 · Fine-tuned'])):
                matrix(axs[i,j],key,f'Article {a} · {heading}\nFAISS k={k}'+(' · minimum MAE' if j==0 else ''))
        fig.subplots_adjust(wspace=.42,hspace=.8,bottom=.2 if len(arts)==1 else .08)
        save(fig,'approach_confusions_'+suffix)


def direction():
    fig,axs=plt.subplots(1,3,figsize=(13,4.7),sharex=True)
    for ax,a in zip(axs,ARTS):
        keys=[(a,m,'Baseline',c,0) for m in MODELS for c in ['Zero-shot','Few-shot']]
        names=[f'{m}\n{c} ({RUNS[k][2]["valid_unique"]}/{RUNS[k][2]["target"]})' for k,(m,c) in zip(keys,[(m,c) for m in MODELS for c in ['Zero-shot','Few-shot']])]
        left=np.zeros(len(keys))
        for field,color,label in [('over','#CC6677','Overestimated'),('exact','#4477AA','Correct'),('under','#DDCC77','Underestimated')]:
            v=[100*RUNS[k][2][field] if k in RUNS else 0 for k in keys]
            ax.barh(range(len(keys)),v,left=left,color=color,label=label)
            for n,x in enumerate(v):
                if x>7: ax.text(left[n]+x/2,n,f'{x:.1f}%',ha='center',va='center',fontsize=8)
            left+=v
        ax.set_yticks(range(len(keys)),names,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('Test cases (%)');ax.set_title(f'Article {a}')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.1,1,1));save(fig,'baseline_error_direction')


def performance():
    palette=['#4477AA','#EE6677','#228833','#AA3377','#66CCEE','#AA4499']
    methods=['faiss','faiss+RR','bm25','bm25+RR','tgn_kg','tgn_kg+RR']
    for model in MODELS:
        fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
        for j,a in enumerate(ARTS):
            for i,metric in enumerate(['mae','src']):
                ax=axs[i,j]
                for method,color in zip(methods,palette):
                    vals=[RUNS.get((a,model,'Experiment 1',method,k),(None,None,{}))[2].get(metric,np.nan) for k in [3,5,10]]
                    ax.plot([3,5,10],vals,'o--' if '+RR' in method else 'o-',color=color,
                            lw=1.6,ms=5,label=method.replace('tgn_kg','TGN').replace('faiss','FAISS').replace('bm25','BM25'))
                if metric=='mae':
                    majority=RUNS[(a,'Majority','Majority','Zero-shot',0)][2]['mae']
                    ax.scatter([2.5],[majority],marker='D',color='black',s=28,label='Majority reference')
                ax.set_xticks([3,5,10]); ax.set_ylabel('MAE ↓' if metric=='mae' else 'Spearman ρ ↑')
                if i==0: ax.set_title(f'Article {a}')
                else: ax.set_xlabel('Retrieved examples (k)')
                pass  # incomplete-cell annotation removed
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,.12,1,1));save(fig,'experiment1_'+('gptoss' if model=='GPT-OSS' else 'llama'))
    fig,axs=plt.subplots(2,3,figsize=(12,7.5))
    points=[]
    for j,a in enumerate(ARTS):
        keys=[(a,'GPT-OSS','Baseline','Zero-shot',0),best(a,'GPT-OSS'),best(a,'GPT-OSS','Iterative'),best(a,'GPT-OSS','Fine-tuned'),best(a,'Llama 3.3','Fine-tuned'),(a,'Majority','Majority','Zero-shot',0)]
        names=['GPT zero-shot','GPT FAISS','GPT iterative','GPT fine-tuned','Llama fine-tuned','Majority']
        for i,metric in enumerate(['mae','src']):
            ax=axs[i,j]
            for n,(key,name) in enumerate(zip(keys,names)):
                if key in RUNS:
                    r=RUNS[key][2];v=r[metric]
                    if np.isfinite(v):
                        ax.scatter(v,n,color=palette[n],s=42)
                        ax.annotate(f'{v:.3f}'+(f'  k={key[4]}' if key[4] else ''),(v,n),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
                    else: ax.text(.04,n,'Undefined (constant)',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
                else: ax.text(.04,n,'TBA',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
            ax.set_yticks(range(6),names if j==0 else []);ax.set_ylim(5.6,-.6)
            ax.set_xlabel('MAE ↓' if metric=='mae' else 'Spearman ρ ↑')
            if i==0: ax.set_title(f'Article {a}')
            ax.set_xlim((0,1.35) if metric=='mae' else (-.015,.31))
    fig.tight_layout();save(fig,'approach_performance')


def paired_and_distributions():
    fig,axs=plt.subplots(1,3,figsize=(12,4.7))
    rng=np.random.default_rng(42); rows=[]
    for ax,a in zip(axs,ARTS):
        base=(a,'GPT-OSS','Experiment 1','faiss',10)
        comparisons=[('Iterative',(a,'GPT-OSS','Iterative','faiss',10)),('GPT fine-tuned',(a,'GPT-OSS','Fine-tuned','faiss',10)),('Llama fine-tuned',(a,'Llama 3.3','Fine-tuned','faiss',10))]
        for j,(name,key) in enumerate(comparisons):
            if base not in RUNS or key not in RUNS:
                ax.text(.04,j,'TBA',transform=ax.get_yaxis_transform(),va='center');continue
            y,p,_=RUNS[base]; _,q,_=RUNS[key];d=abs(y-q)-abs(y-p)
            boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
            lo,hi=np.quantile(boot,[.025,.975]);mu=d.mean()
            ax.errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#4477AA',capsize=4)
            rows.append(dict(article=a,comparison=name,delta_mae=mu,ci_low=lo,ci_high=hi,n=len(d)))
        ax.set_yticks(range(3),[x[0] for x in comparisons]);ax.set_ylim(2.5,-.5)
        ax.set_xlim(-.32,.20)
        ax.set_xlabel('Δ MAE against GPT FAISS k=10\nNegative favours comparison');ax.set_title(f'Article {a}')
    fig.tight_layout();save(fig,'paired_mae_differences');pd.DataFrame(rows).to_csv(OUT/'paired_mae_bootstrap.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,5))
    dist=[]
    for ax,a in zip(axs,ARTS):
        y=CASES[a].importance.to_numpy(int)
        sets=[('Actual',y)]
        for name,key in [('Zero-shot',(a,'GPT-OSS','Baseline','Zero-shot',0)),('FAISS k=10',(a,'GPT-OSS','Experiment 1','faiss',10)),('Iterative k=10',(a,'GPT-OSS','Iterative','faiss',10)),('GPT FT k=10',(a,'GPT-OSS','Fine-tuned','faiss',10)),('Llama FT k=10',(a,'Llama 3.3','Fine-tuned','faiss',10))]:
            sets.append((name,RUNS[key][1] if key in RUNS else None))
        left=np.zeros(len(sets))
        for level,color in enumerate(COLORS,1):
            v=[100*np.mean(x==level) if x is not None else 0 for _,x in sets]
            ax.barh(range(len(sets)),v,left=left,color=color,label=LABELS[level-1])
            left+=v
            for (name,x),pct in zip(sets,v):dist.append(dict(article=a,condition=name,label=level,percent=pct if x is not None else np.nan))
        for j,(_,x) in enumerate(sets):
            if x is None:ax.text(3,j,'TBA',va='center')
        ax.set_yticks(range(len(sets)),[s[0] for s in sets]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_title(f'Article {a}');ax.set_xlabel('Cases (%)')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(COLORS,LABELS)],loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'prediction_distributions');pd.DataFrame(dist).to_csv(OUT/'prediction_distributions.csv',index=False)


def recall_tradeoff():
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.5))
    for ax,a in zip(axs,ARTS):
        for name,key,color in [('FAISS',(a,'GPT-OSS','Experiment 1','faiss',10),'#4477AA'),
                               ('Iterative',(a,'GPT-OSS','Iterative','faiss',10),'#228833'),
                               ('GPT FT',(a,'GPT-OSS','Fine-tuned','faiss',10),'#AA3377'),
                               ('Llama FT',(a,'Llama 3.3','Fine-tuned','faiss',10),'#EE6677'),
                               ('Majority',(a,'Majority','Majority','Zero-shot',0),'#555555')]:
            if key not in RUNS:continue
            r=RUNS[key][2]
            ax.scatter(100*r['low_recall'],100*r['minority_recall'],s=50,color=color,label=name)
            ax.annotate(name,(100*r['low_recall'],100*r['minority_recall']),xytext=(-5,7 if name!='Majority' else -14),
                        textcoords='offset points',ha='right',fontsize=8)
        ax.set_xlim(35,105);ax.set_ylim(-5,40);ax.set_title(f'Article {a}')
        ax.set_xlabel('Level 3 recall (%)');ax.set_ylabel('Mean recall across other classes (%)')
    fig.tight_layout();save(fig,'majority_minority_recall_tradeoff')


def citation_distributions():
    """Reconstruct outcome citations; never relabel communicated metadata as outcomes."""
    distribution=[];coverage=[];links=[];resolution=[];dateaudit=[]
    meta=pd.read_pickle(ROOT/'data/processed/metadata/unfiltered_metadata.pkl')
    meta=meta[meta.doctypebranch=='COMMUNICATEDCASES'].drop_duplicates('itemid').set_index('itemid')
    for a in ARTS:
        oc=pd.read_pickle(ROOT/f'data/processed/article{a}/outcome_cases.pkl')
        oc=oc.copy();oc['date']=pd.to_datetime(oc['date']); byfile=oc.set_index('File')
        appindex=defaultdict(list)
        for r in oc.to_dict('records'):
            for app in str(r['appno']).split(';'):appindex[app.strip()].append(r)
        cited={};match_count=ambiguous=unresolved=0
        for fn,q in CASES[a].iterrows():
            commdate=pd.to_datetime(meta.loc[fn,'kpdate'],errors='coerce') if fn in meta.index else pd.NaT
            dateaudit.append(dict(article=a,Filename=fn,stored_doc_date=str(q.doc_date),communication_kpdate=str(commdate),
                                  stored_after_communication=bool(pd.notna(commdate) and pd.Timestamp(q.doc_date)>commdate)))
            apps=set(str(q.appno).split(';'))
            cutoff=commdate if pd.notna(commdate) else pd.Timestamp(q.doc_date)
            matches={r['File']:r for app in apps for r in appindex.get(app.strip(),[]) if r['date']>=cutoff}
            match_count+=bool(matches);ambiguous+=len(matches)>1
            refs=set()
            for out in matches.values():
                for app in str(out.get('extractedappno','')).split(';'):
                    app=app.strip()
                    if '/' not in app or app in apps:continue
                    candidates=[r for r in appindex.get(app,[]) if r['date']<out['date'] and r['File'] not in matches]
                    if not candidates:
                        unresolved+=1
                        resolution.append(dict(article=a,query=fn,citing_document=out['File'],cited_appno=app,resolved_document=''))
                        continue
                    # Most recent pre-citing document for the cited application.
                    ref=max(candidates,key=lambda r:r['date']);refs.add(ref['File'])
                    resolution.append(dict(article=a,query=fn,citing_document=out['File'],cited_appno=app,resolved_document=ref['File']))
            cited[fn]=sorted(refs)
        gold=pd.read_pickle(ROOT/f'data/vectordb/article{a}/gold_results.pkl')
        groups={'Outcome citations':cited,'Current gold links (top 10)':{fn:gold.get(fn,[])[:10] for fn in CASES[a].index}}
        bestkey=best(a,'GPT-OSS');bestk=bestkey[4] if bestkey else 10
        for method,k in [('faiss',bestk),('faiss+RR',bestk),('tgn_kg',bestk)]:
            stem=method.replace('+RR','')+f'_k{k}'
            if '+RR' in method:stem+='_rerank-rerank_model_20260924_ap_selected'
            folder=ROOT/f'data/prompts/retrieval_v1/article{a}'/(stem+'_text1_test_ctx9600_out1000')
            found={}
            for p in sorted(folder.glob('part-*.jsonl')):
                for line in p.open():
                    r=json.loads(line);fn=r['Filename']
                    if fn in CASES[a].index: found[fn]=r['example_filenames']
            groups[method+f' k={k}']=found
        for group,mapping in groups.items():
            count=np.zeros(4,int);queryweights=np.zeros(4);nonempty=missingids=future=aftercomm=ownapp=0;resolved_unique=set()
            matchedweights=np.zeros(4);matchedn=0
            for fn in CASES[a].index:
                docs=list(dict.fromkeys(mapping.get(fn,[])));local=np.zeros(4,int)
                for doc in docs:
                    if doc not in byfile.index:missingids+=1;continue
                    label=int(byfile.loc[doc,'importance']);assert 1<=label<=4
                    local[label-1]+=1;resolved_unique.add(doc)
                    qdate=pd.Timestamp(CASES[a].loc[fn,'doc_date']);ddate=byfile.loc[doc,'date']
                    commdate=pd.to_datetime(meta.loc[fn,'kpdate'],errors='coerce') if fn in meta.index else pd.NaT
                    future+=int(ddate>=qdate)
                    aftercomm+=int(pd.notna(commdate) and ddate>=commdate)
                    selfapp=bool(set(str(CASES[a].loc[fn,'appno']).split(';')) & set(str(byfile.loc[doc,'appno']).split(';')))
                    ownapp+=int(selfapp)
                    links.append(dict(article=a,group=group,query=fn,outcome=doc,importance=label,
                                      query_date=str(qdate),outcome_date=str(ddate),communication_date=str(commdate),
                                      not_strictly_precommunication=bool(pd.notna(commdate) and ddate>=commdate),
                                      same_application_as_query=selfapp,
                                      not_strictly_prequery=bool(ddate>=qdate)))
                if local.sum(): nonempty+=1;queryweights+=local/local.sum()
                if local.sum() and cited.get(fn):matchedn+=1;matchedweights+=local/local.sum()
                count+=local
            for level in range(4):
                distribution.append(dict(article=a,group=group,label=level+1,count=count[level],
                    percent=100*count[level]/count.sum() if count.sum() else np.nan,
                    query_weighted_percent=100*queryweights[level]/nonempty if nonempty else np.nan,
                    matched_query_percent=100*matchedweights[level]/matchedn if matchedn else np.nan))
            coverage.append(dict(article=a,group=group,test_cases=len(CASES[a]),queries_present=len(mapping),
                queries_with_resolved_context=nonempty,document_occurrences=int(count.sum()),unique_documents=len(resolved_unique),
                missing_document_ids=missingids,not_strictly_prequery_occurrences=future,
                not_strictly_precommunication_occurrences=aftercomm,
                same_application_occurrences=ownapp,
                matched_query_n=matchedn,matched_outcome_queries=match_count if group=='Outcome citations' else '',
                multiple_outcome_queries=ambiguous if group=='Outcome citations' else '',
                unresolved_citation_app_occurrences=unresolved if group=='Outcome citations' else ''))
    df=pd.DataFrame(distribution);df.to_csv(OUT/'context_importance_distribution.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'context_coverage.csv',index=False)
    pd.DataFrame(links).to_csv(OUT/'context_links.csv',index=False)
    pd.DataFrame(resolution).to_csv(OUT/'citation_resolution.csv',index=False)
    pd.DataFrame(dateaudit).to_csv(OUT/'query_date_audit.csv',index=False)
    for field,suffix in [('percent','occurrence_weighted'),('query_weighted_percent','query_weighted'),('matched_query_percent','matched_queries')]:
        fig,axs=plt.subplots(3,2,figsize=(12,10.8),sharey=True)
        for i,a in enumerate(ARTS):
            sub=df[df.article==a]; groups=list(sub.group.unique())
            for j,names in enumerate([groups[:2],groups[2:]]):
                ax=axs[i,j]; width=.8/max(1,len(names))
                for n,name in enumerate(names):
                    v=sub[sub.group==name].sort_values('label')[field].to_numpy()
                    ax.bar(np.arange(4)-.4+(n+.5)*width,v,width=width*.94,
                           label=name.replace('faiss','FAISS').replace('bm25','BM25').replace('tgn_kg','TGN'),color=['#4477AA','#EE6677','#228833','#AA3377'][n])
                ax.set_xticks(range(4),LABELS);ax.set_ylim(0,100);ax.set_title(f'Article {a} · '+('Cited / gold context' if j==0 else 'Retrieved context'))
                if j==0:ax.set_ylabel('Context cases (%)')
                ax.legend(frameon=False,fontsize=8,ncol=2)
        fig.tight_layout();save(fig,'context_distributions_'+suffix)
    fig,axs=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    cov=pd.DataFrame(coverage)
    for ax,a in zip(axs,ARTS):
        sub=cov[(cov.article==a)&cov.group.str.contains('k=')]
        vals=100*sub.not_strictly_precommunication_occurrences/sub.document_occurrences
        bars=ax.bar(range(len(sub)),vals,color=['#4477AA','#EE6677','#228833'])
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=4,fontsize=9)
        ax.set_xticks(range(len(sub)),[s.replace('faiss','FAISS').replace('tgn_kg','TGN') for s in sub.group],rotation=20)
        ax.set_ylim(0,45);ax.set_title(f'Article {a}')
        if a==3:ax.set_ylabel('Context documents dated on/after\ncommunication date (%)')
    fig.tight_layout();save(fig,'retrieval_temporal_availability_audit')


def main():
    collect();confusions();direction();performance();paired_and_distributions();recall_tradeoff();citation_distributions()
    pd.DataFrame(AUDIT).to_csv(OUT/'run_audit.csv',index=False)
    pd.DataFrame(SNAPSHOT).to_csv(OUT/'evaluated_predictions.csv',index=False)
    pd.DataFrame(SELECTED).to_csv(OUT/'selections.csv',index=False)
    metadata=dict(audit_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                  retry_policy='first valid prediction in append order',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__,
                  test_sha256={a:hashlib.sha256((ROOT/f'data/processed/article{a}/splits/comm_test.pkl').read_bytes()).hexdigest() for a in ARTS})
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(pd.DataFrame(AUDIT).groupby(['family','model']).complete.agg(['sum','count']).to_string())


if __name__=='__main__': main()
