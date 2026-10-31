"""Reproduce validation comparisons and small input-contract checks; never opens test."""
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
from training_core import dataset
from features import choose, probabilities
from arithmetic_features import extract_arithmetic
from trace_arithmetic import verify_synthetic
from models import model_scores
from submission import predict
from answer_aggregation import summarize, paired
from common import parse_answer
from release_utils import ROOT, verify, sha, write_json


def vote(answers):
    counts=Counter(a for a in answers if a is not None)
    return answers.index(max(counts,key=counts.get)) if counts else 0


def main():
    verify(ROOT/'checkpoint')
    d=dataset('validation'); ck=ROOT/'checkpoint'; config=json.loads((ck/'config.json').read_text())
    linear=model_scores(d['bank'],d['e'],d['f'],d['answers'],ck/'linear')
    f=np.concatenate([d['f'],extract_arithmetic(d['bank'],d['answers'])],axis=-1)
    neural=model_scores(d['bank'],d['e'],f,d['answers'],ck/'neural')
    lm,nm=config['members']
    p=lm['weight']*probabilities(linear/lm['temperature'],d['valid'])+nm['weight']*probabilities(neural/nm['temperature'],d['valid'])
    majority=[vote(answers) for answers in d['answers']]
    selected={'first_candidate':[0]*len(d['bank']),'parsed_majority':majority,
              'linear':choose(linear,d['valid'],d['answers']),
              'arithmetic_neural':choose(neural,d['valid'],d['answers']),
              'fusion_without_trace':choose(p,d['valid'],d['answers']),
              'final':predict(d['bank'],{'embeddings':d['e']},ck)}
    comparison={name:summarize(d,pred) for name,pred in selected.items()}
    comparison['trace_paired']=paired(d,selected['fusion_without_trace'],selected['final'])
    write_json(ROOT/'evaluation/validation-comparison.json',comparison)
    examples=[]
    for n,(base,final) in enumerate(zip(selected['fusion_without_trace'],selected['final'])):
        if len(examples)>=4:break
        if not d['hard'][n] or d['y'][n,final]>.5:continue
        examples.append({'split':'validation','index':n,'question':d['bank'][n]['question'],
                         'parsed_answers':d['answers'][n], 'final_choice':final,
                         'reference_numeric_answer':next(a for a,y in zip(d['answers'][n],d['y'][n]) if y>.5),
                         'selected_text':d['bank'][n]['candidates'][final]['text']})
    write_json(ROOT/'evaluation/validation-failures.json',examples)
    checks={}
    for indices in ([],[7],[22,3,100,9,44]):
        bank=[copy.deepcopy(d['bank'][i]) for i in indices]
        for i,b in enumerate(bank):b['id']='unseen-'+str(i)
        actual=predict(bank,{'embeddings':d['e'][indices]},ck)
        assert actual==[selected['final'][i] for i in indices]
        assert all(type(x) is int and 0<=x<4 for x in actual)
        checks[str(len(indices))]='IDs changed; reordered subsets preserve alignment; Python integer outputs'
    assert parse_answer('#### 1/2')==parse_answer('#### 0.5')
    assert choose(np.array([[0,1,2,3.]]),np.array([[0,1,1,0]],bool),[[None,'1/2','1/2',None]])==[1]
    assert choose(np.ones((1,4)),np.zeros((1,4),bool),[[None]*4])==[0]
    assert vote(['2','3','3','2'])==0
    assert vote([None,'3',None,'2'])==1
    upstream=json.loads((ROOT/'provenance/upstream.json').read_text())
    for name,expected in upstream['official_file_sha256'].items():assert sha(ROOT/'code'/name)==expected,name
    write_json(ROOT/'verification/interface.json',{'status':'passed','subset_checks':checks,
        'parser_equivalence':'1/2 equals 0.5','multiple_correct':'first valid duplicate selected',
        'no_valid_answers':'falls back to index0','arithmetic':verify_synthetic(),
        'official_files':'four upstream files byte-identical','checkpoint_freeze_sha256':sha(ck/'FREEZE.json'),
        'test_references_accessed':False})
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
