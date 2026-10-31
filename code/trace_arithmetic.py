"""Bounded arithmetic-trace recomputation; no reference access or ID features."""
import ast
from fractions import Fraction
import math
import re
import numpy as np
from arithmetic_features import normalize
from features import numeric_values

SIDE=r'[0-9.()+*/\-\s]+'
NAMES=('correction_support','own_final_tainted','direct_final_error','trace_errors','proposal_available','corrected_consensus')


def rational(expression,changes=None):
    tree=ast.parse(expression.strip(),mode='eval')
    if sum(1 for _ in ast.walk(tree))>50:raise ValueError('oversized')
    def walk(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):v=Fraction(str(n.value))
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):v=walk(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        elif isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,ast.Add):v=a+b
            elif isinstance(n.op,ast.Sub):v=a-b
            elif isinstance(n.op,ast.Mult):v=a*b
            else:v=a/b
        else:raise ValueError('unsupported')
        if abs(v)>10**15 or v.numerator.bit_length()>256 or v.denominator.bit_length()>256:raise ValueError('unbounded')
        return changes.get(v,v) if changes else v
    return walk(tree.body)


def materially_different(a,b,rhs):
    if a==b:return False
    delta=abs(float(a-b));decimal=re.fullmatch(r'[-+]?\d*\.(\d+)',rhs.strip())
    rounding=.50001*10**(-len(decimal.group(1))) if decimal else 0.
    return delta>max(rounding,1e-6,.001*max(abs(float(a)),abs(float(b))))


def proposal(question,text,answer):
    try:final=Fraction(answer)
    except (ValueError,TypeError,ZeroDivisionError):return None,0,0,0
    literals=set(numeric_values(question));changes={};errors=0;direct=0;final_target=None
    for line in normalize(text).splitlines():
        parts=re.split(r'=|≈',line)
        for i in range(len(parts)-1):
            left=re.search('('+SIDE+r')\s*$',parts[i]);right=re.match(r'\s*('+SIDE+')',parts[i+1])
            if not left or not right:continue
            lhs,rhs=left.group(1).strip(),right.group(1).strip()
            if not re.search(r'[*/]|\d\s*[+\-]\s*\d',lhs+rhs):continue
            try:a=rational(lhs);b=rational(rhs);fixed=rational(lhs,changes)
            except (SyntaxError,ValueError,TypeError,ZeroDivisionError,OverflowError,RecursionError):continue
            if materially_different(fixed,b,rhs):
                errors+=1
                if b==final:final_target=fixed;direct|=int(materially_different(a,b,rhs))
                if b not in literals and b!=0 and len(changes)<8:changes[b]=fixed
    target=changes.get(final,final_target)
    if errors>8 or target is None or target==final:return None,0,direct,errors
    return target,1,direct,errors


def extract_trace(bank,answers):
    rows=[]
    for row,aa in zip(bank,answers):
        props=[proposal(row['question'],c['text'],a) for c,a in zip(row['candidates'],aa)]
        corrected=[p[0] if p[0] is not None else Fraction(a) if a is not None else None for p,a in zip(props,aa)]
        rr=[]
        for a,p in zip(aa,props):
            av=Fraction(a) if a is not None else None
            support=sum(x[0] is not None and x[0]==av for x in props)
            consensus=sum(x is not None and x==av for x in corrected)
            rr.append([support,p[1],p[2],math.log1p(p[3]),int(p[0] is not None),consensus])
        rows.append(rr)
    return np.asarray(rows,np.float32).reshape(len(bank),4,len(NAMES))


def verify_synthetic():
    assert proposal('Three days: 3, 5, 7 snails.',r'3 + 5 + 7 = 16\n#### 16','16')[0]==15
    text=r'\frac{0.6}{9} = \frac{2}{15}'+'\n'+r'\frac{2}{15} * 60 = 8'+'\n#### 8'
    assert proposal('Speeds24 and15, gap0.6; ask minutes.',text,'8')[0]==4
    assert proposal('A rate question','0.6 / 9 = 0.0667\n0.0667 * 60 = 4\n#### 4','4')[0] is None
    assert proposal('x','x + 2 = 9\n#### 9','9')[0] is None
    return {'cases':4,'status':'passed','scope':'direct and propagated correction, rounding tolerance, unsupported variable unknown'}
