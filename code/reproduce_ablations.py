"""Optional replay of the controlled historical MLP/text comparisons (no test)."""
import argparse
import json
from pathlib import Path
from training_core import dataset, fit, write_json
from release_utils import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/ablations');p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Choose a fresh empty directory')
    tr,va=dataset('train'),dataset('validation')
    historical=json.loads((ROOT/'experiments/controlled-ablations.json').read_text());rows=[]
    for row in historical:
        c=row['config'];kind=c['kind']
        if kind=='catboost':continue
        rec,_=fit(c,tr,va,a.output/row['name'],seed=c['seed'],fixed_epochs=c.get('epochs'),device=a.device)
        rows.append({'name':row['name'],**rec});write_json(a.output/'results.json',rows)
        print(row['name'],rec['metrics'],flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
