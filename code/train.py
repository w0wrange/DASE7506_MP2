"""Standalone final-recipe retraining: ten fold fits, two full fits, residual grid."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from training_core import dataset,subset,fit
from features import probabilities
from models import model_scores
from arithmetic_features import extract_arithmetic
from trace_arithmetic import extract_trace
from answer_aggregation import select_answers,summarize,paired
from release_utils import ROOT,write_json,freeze


def read(p):return json.loads(Path(p).read_text())
def choices(d,p,agg):return select_answers(p,d['valid'],d['answers'],agg)
def augmented(split):
    d=dataset(split);d['f']=np.concatenate([d['f'],extract_arithmetic(d['bank'],d['answers'])],axis=-1);return d
def logits(d,path):return model_scores(d['bank'],d['e'],d['f'],d['answers'],path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/reproduction');p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory; existing artifacts are never silently reused')
    out.mkdir(parents=True,exist_ok=True);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):run(out,a.device)


def run(out,device):
    start=time.perf_counter();recipe=read(ROOT/'configs/training.json');data=augmented('train');folds=read(ROOT/'configs/folds.json')['held_out_indices'];lm,nm=recipe['members'];ls=np.zeros_like(data['y']);ns=np.zeros_like(data['y']);ledger=[]
    write_json(out/'recipe.json',recipe)
    for fold,held in enumerate(folds):
        held=np.asarray(held,int);keep=np.setdiff1d(np.arange(len(data['y'])),held);tr,dev=subset(data,keep),subset(data,held)
        for label,c,epochs in [('linear',recipe['linear'],None),('neural',recipe['neural'],recipe['epochs'])]:
            path=out/f'folds/fold-{fold}/{label}';rec,_=fit(c,tr,None,path,seed=recipe['seed']+fold,fixed_epochs=epochs,device=device);ledger.append({'fold':fold,'member':label,**rec})
            (ls if label=='linear' else ns)[held]=logits(dev,path)
        print('Completed fold',fold,flush=True)
    np.save(out/'linear-oof.npy',ls);np.save(out/'neural-oof.npy',ns)
    for label,c,epochs in [('linear',recipe['linear'],None),('neural',recipe['neural'],recipe['epochs'])]:
        rec,_=fit(c,data,None,out/'full'/label,seed=recipe['seed'],fixed_epochs=epochs,device=device);ledger.append({'fold':'full','member':label,**rec})
    tp=lm['weight']*probabilities(ls/lm['temperature'],data['valid'])+nm['weight']*probabilities(ns/nm['temperature'],data['valid'])
    val=augmented('validation');vl=logits(val,out/'full/linear');vn=logits(val,out/'full/neural');vp=lm['weight']*probabilities(vl/lm['temperature'],val['valid'])+nm['weight']*probabilities(vn/nm['temperature'],val['valid'])
    tt=extract_trace(data['bank'],data['answers']);vt=extract_trace(val['bank'],val['answers']);base=choices(data,tp,{'mode':'candidate_max'});vb=choices(val,vp,{'mode':'candidate_max'});threshold=summarize(data,base)['hard_correct'];rows=[]
    for bonus,penalty in itertools.product(recipe['residual_grid'],repeat=2):
        for agg in recipe['aggregation_grid']:
            t=probabilities(np.log(np.maximum(tp,1e-8))+bonus*tt[...,0]-penalty*tt[...,1],data['valid']);v=probabilities(np.log(np.maximum(vp,1e-8))+bonus*vt[...,0]-penalty*vt[...,1],val['valid']);tc=choices(data,t,agg);vc=choices(val,v,agg);tm=summarize(data,tc)
            rows.append({'support_bonus':bonus,'taint_penalty':penalty,'aggregation':agg,'oof':tm,'validation':summarize(val,vc),'eligible':tm['hard_correct']>=threshold,'train_paired':paired(data,base,tc),'validation_paired':paired(val,vb,vc)})
    eligible=[x for x in rows if x['eligible']];selected=sorted(eligible,key=lambda x:(-x['validation']['correct'],-x['oof']['correct'],x['support_bonus']+x['taint_penalty']))[0]
    write_json(out/'residual-grid.json',rows);write_json(out/'selection.json',selected);write_json(out/'fit-ledger.json',ledger)
    ck=out/'checkpoint';ck.mkdir()
    for label in ('linear','neural'):
        (ck/label).mkdir()
        for n in ('model.json','weights.npz'):shutil.copy2(out/'full'/label/n,ck/label/n)
    write_json(ck/'config.json',{'members':recipe['members'],'support_bonus':selected['support_bonus'],'taint_penalty':selected['taint_penalty'],'aggregation':selected['aggregation']})
    freeze(ck,'Replayed final training recipe and train-OOF/validation residual selection; see DISCLOSURE.md for historical test-feedback context.')
    import platform,torch
    write_json(out/'resources.json',{'seconds':time.perf_counter()-start,'fits':len(ledger),'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'python':platform.python_version(),'torch':torch.__version__,'seed':recipe['seed'],'threads':4})
    print(json.dumps({'checkpoint':str(ck),'selected':selected,'training_seconds':time.perf_counter()-start},indent=2))


if __name__=='__main__':main()
