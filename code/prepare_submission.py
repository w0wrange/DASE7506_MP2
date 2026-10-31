"""Fill local form metadata after the student publishes; performs no network action."""
import argparse
import json
import re
from urllib.parse import urlparse
from release_utils import ROOT, verify, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-url',required=True);p.add_argument('--checkpoint-url',required=True);p.add_argument('--revision',required=True);a=p.parse_args()
    for value in (a.code_url,a.checkpoint_url):
        if urlparse(value).scheme!='https' or not urlparse(value).netloc:raise ValueError('Use a public HTTPS URL')
    if not re.fullmatch(r'[0-9a-f]{40}',a.revision):raise ValueError('Use the full 40-character Git commit SHA')
    verify(ROOT/'checkpoint');record=json.loads((ROOT/'SUBMISSION.json').read_text());score=json.loads((ROOT/'evaluation/test-score.json').read_text())
    record.update(code_url=a.code_url,checkpoint_url=a.checkpoint_url,code_revision=a.revision,score=score['accuracy'])
    write_json(ROOT/'SUBMISSION.json',record)
    print('Updated local SUBMISSION.json. The student must create the course GitHub issue to submit.')


if __name__=='__main__':main()
