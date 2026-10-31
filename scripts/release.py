"""Create/verify a content-addressed source manifest and clean deliverable ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from release_utils import asset_size, sha, verify, write_json


def paths():
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,check=True)
    return [ROOT/n for n in sorted(set(result.stdout.decode().strip('\0').split('\0'))) if n and (ROOT/n).is_file()]


def manifest():
    verify(ROOT/'checkpoint'); files={str(p.relative_to(ROOT)):sha(p) for p in paths() if p.name not in ('RELEASE.json','SUBMISSION.json')}
    source={n:d for n,d in files.items() if n.startswith(('code/','configs/','scripts/')) or n=='requirements.txt'}
    identity=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    record={'schema':1,'source_release_id':'sha256:'+identity,'checkpoint_freeze_sha256':sha(ROOT/'checkpoint/FREEZE.json'),'files_sha256':files,'mutable_form_draft':'SUBMISSION.json is excluded to allow the student to fill publication URLs; official score and frozen checkpoint remain hashed.','git_revision':'Student creates initial Git commit with their own identity; use that published full commit SHA in the course form.','eligibility':'Historical test feedback conflict disclosed; instructor review required. Reference CPU limit unverified.'}
    write_json(ROOT/'RELEASE.json',record)
    form=ROOT/'SUBMISSION.json'
    if form.exists():
        draft=json.loads(form.read_text());draft['source_release_id']=record['source_release_id'];write_json(form,draft)
    return record


def check():
    verify(ROOT/'checkpoint'); record=json.loads((ROOT/'RELEASE.json').read_text())
    for n,digest in record['files_sha256'].items():
        if sha(ROOT/n)!=digest:raise ValueError('Changed release file: '+n)
    print('Release hashes verified:',record['source_release_id']);return record


def package(destination):
    check();destination.mkdir(parents=True,exist_ok=True)
    repo=destination/'DASE7506_MP2_3036802038_repository.zip'
    with zipfile.ZipFile(repo,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in paths():z.write(p,'DASE7506_MP2/'+str(p.relative_to(ROOT)))
    checkpoint=destination/'checkpoint_submission_3036802038.zip'
    with zipfile.ZipFile(checkpoint,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted((ROOT/'checkpoint').rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    write_json(destination/'SHA256SUMS.json',{p.name:sha(p) for p in (repo,checkpoint)})
    print(json.dumps({'repository_zip':str(repo),'checkpoint_zip':str(checkpoint),'inference_uncompressed_bytes':asset_size(ROOT/'checkpoint')}))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('manifest','verify','package'));p.add_argument('--output',type=Path,default=ROOT/'dist');a=p.parse_args()
    if a.action=='manifest':print(manifest()['source_release_id'])
    elif a.action=='verify':check()
    else:package(a.output.resolve())


if __name__=='__main__':main()
