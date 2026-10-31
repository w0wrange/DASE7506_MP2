"""H7 probability residual correction by input-only arithmetic trace evidence."""
import json
from pathlib import Path
import numpy as np
from features import extract,probabilities
from arithmetic_features import extract_arithmetic
from trace_arithmetic import extract_trace
from models import model_scores
from answer_aggregation import select_answers


def predict(bank,arrays,checkpoint):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        if not bank:return []
        root=Path(checkpoint);c=json.loads((root/'config.json').read_text());e=np.asarray(arrays['embeddings'],np.float32);f,v,a=extract(bank,e)
        fa=np.concatenate([f,extract_arithmetic(bank,a)],axis=-1);p=np.zeros((len(bank),4),np.float64)
        for m in c['members']:
            s=model_scores(bank,e,fa if m['flavor']=='arithmetic' else f,a,root/m['path']);p+=m['weight']*probabilities(s/m['temperature'],v)
        tr=extract_trace(bank,a);s=np.log(np.maximum(p,1e-8))+c['support_bonus']*tr[...,0]-c['taint_penalty']*tr[...,1]
        out=probabilities(s,v)
        return select_answers(out,v,a,c['aggregation'])
