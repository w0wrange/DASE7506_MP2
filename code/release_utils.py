"""Release integrity and inference-asset accounting."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent.parent
INFERENCE=('submission.py','models.py','features.py','arithmetic_features.py','trace_arithmetic.py','answer_aggregation.py','common.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def asset_size(path):
    total=0
    for p in Path(path).rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:total+=sum(x.file_size for x in z.infolist())
        else:total+=p.stat().st_size
    return total
def freeze(path,selected_using):
    path=Path(path)
    value={'schema':1,'method':'code/submission.py','asset_sha256':{str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name!='FREEZE.json'},'source_sha256':{'code/'+n:sha(ROOT/'code'/n) for n in INFERENCE},'selected_using':selected_using,'disclosure':'See DISCLOSURE.md: repeated reference testing occurred; eligibility requires instructor review.'}
    write_json(path/'FREEZE.json',value);return value
def verify(path):
    path=Path(path);m=json.loads((path/'FREEZE.json').read_text())
    for n,d in m['asset_sha256'].items():
        if sha(path/n)!=d:raise ValueError('Changed checkpoint asset: '+n)
    for n,d in m['source_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError('Changed inference source: '+n)
    return m
