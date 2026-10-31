"""Time fresh official prediction and independent scoring; no cached choices."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from release_utils import ROOT,asset_size,sha,verify,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoint');p.add_argument('--split',choices=('validation','test'),default='validation');p.add_argument('--output-dir',type=Path,default=ROOT/'evaluation');p.add_argument('--data-dir',type=Path,default=ROOT/'code/data');p.add_argument('--keep-validation-predictions',action='store_true');a=p.parse_args()
    ck=a.checkpoint.resolve();out=a.output_dir.resolve();verify(ck);size=asset_size(ck)
    if size>16*2**20:raise ValueError('Inference assets exceed16MiB')
    score=out/(a.split+'-score.json')
    if a.split=='test' and score.exists():raise ValueError('Final test JSON already exists; choose a distinct directory for reproduction')
    if a.split=='test' and a.keep_validation_predictions:raise ValueError('This release does not persist test predictions')
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4',BLIS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='mp2-official-') as tmp:
        pred=Path(tmp)/'predictions.json';start=time.perf_counter()
        subprocess.run([sys.executable,str(ROOT/'code/predict.py'),'--method',str(ROOT/'code/submission.py'),'--checkpoint',str(ck),'--split',a.split,'--data-dir',str(a.data_dir.resolve()),'--output',str(pred)],env=env,check=True);pt=time.perf_counter()-start
        start=time.perf_counter();subprocess.run([sys.executable,str(ROOT/'code/score_predictions.py'),'--split',a.split,'--data-dir',str(a.data_dir.resolve()),'--predictions',str(pred),'--output',str(score)],env=env,check=True);st=time.perf_counter()-start
        if a.keep_validation_predictions:shutil.copy2(pred,out/'validation-predictions.json')
    cpu=platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),cpu)
    elif sys.platform=='darwin':
        result=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True)
        if result.returncode==0:cpu=result.stdout.strip()
    record={'split':a.split,'prediction_process_seconds':pt,'scoring_process_seconds':st,'total_seconds':pt+st,'asset_uncompressed_bytes':size,'threads':4,'platform':platform.platform(),'cpu_model':cpu,'python':platform.python_version(),'checkpoint_freeze_sha256':sha(ck/'FREEZE.json'),'reference_cpu_eligibility':'Not established; Xeon Platinum8457C is authoritative.','test_predictions_retained':False,'official_scorer_sha256':sha(ROOT/'code/score_predictions.py'),'bank_manifest_sha256':sha(ROOT/'code/data/manifest.json')}
    write_json(out/(a.split+'-resources.json'),record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
