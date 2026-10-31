"""Install hash-verified official data before offline evaluation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

REVISION='bb801f4f2e7629d270df765ab12c333b2dc616bb'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path);p.add_argument('--splits',nargs='+',choices=('train','validation','test'),default=['train','validation','test']);a=p.parse_args()
    data=Path(__file__).resolve().parent/'data';manifest=json.loads((data/'manifest.json').read_text());names=[]
    for split in a.splits:names += [f'candidates/gsm8k_{split}.jsonl',f'candidates/gsm8k_{split}.npz',f'gsm8k_{split}.jsonl']
    missing=[n for n in names if not (data/n).exists() or hashlib.sha256((data/n).read_bytes()).hexdigest()!=manifest['sha256'][n]]
    if not missing:print('Requested official data already present and hash-verified.');return
    archive=None
    if a.source_dir is None:
        url='https://codeload.github.com/xudongwu-0/DASE7506-MP2/zip/'+REVISION
        request=urllib.request.Request(url,headers={'User-Agent':'DASE7506-MP2-reproduction'})
        with urllib.request.urlopen(request,timeout=180) as response:archive=zipfile.ZipFile(io.BytesIO(response.read()))
    try:
        for n in missing:
            blob=(a.source_dir/n).read_bytes() if a.source_dir else archive.read('DASE7506-MP2-'+REVISION+'/code/data/'+n)
            if hashlib.sha256(blob).hexdigest()!=manifest['sha256'][n]:raise ValueError('Official data hash mismatch: '+n)
            out=data/n;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_bytes(blob);tmp.replace(out)
    finally:
        if archive is not None:archive.close()
    print('Installed',len(missing),'hash-verified official files; no model download.')


if __name__=='__main__':main()
