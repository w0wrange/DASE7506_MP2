"""Bounded numerical equality evidence. No reference access or hard vetoes."""
import math
import re
import numpy as np
from features import safe_arithmetic

NAMES = ['eq_checked','eq_errors','eq_strict_ok','eq_rounded_ok','eq_unknown',
         'eq_coverage','eq_error_share','fraction_checked','chain_checked',
         'terminal_support','terminal_conflict','relative_error_share']
_SIDE = r'[0-9.()+*/\-\s]+'


def normalize(text):
    text = text[:30000].replace('−','-').replace('×','*').replace('÷','/')
    for old,new in [(r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),
                    (r'\right',''),(r'\,',''),(r'\!',''),(r'\%','%')]:
        text=text.replace(old,new)
    text=re.sub(r'(?<=\d),(?=\d)','',text)
    for _ in range(8):
        new=re.sub(r'\\(?:d?frac|tfrac)\s*\{([^{}\n]{1,80})\}\s*\{([^{}\n]{1,80})\}',r'((\1)/(\2))',text)
        if new==text:break
        text=new
    text=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',text)
    return text.replace('$','').replace('**','').replace(r'\approx','≈')


def equalities(text):
    """Only adjacent, fully evaluable numerical sides; unsupported sides unknown."""
    checked=[];unknown=0;norm=normalize(text)
    for line in norm.splitlines():
        parts=re.split(r'=|≈',line)
        for i in range(len(parts)-1):
            left=re.search('('+_SIDE+r')\s*$',parts[i])
            right=re.match(r'\s*('+_SIDE+')',parts[i+1])
            if not left or not right:unknown+=1;continue
            a,b=left.group(1).strip(),right.group(1).strip()
            # Identity assignments do not count as arithmetic evidence.
            if not re.search(r'[*/]|\d\s*[+\-]\s*\d',a+b):unknown+=1;continue
            try:
                av,bv=safe_arithmetic(a),safe_arithmetic(b)
            except (SyntaxError,ValueError,TypeError,ZeroDivisionError,OverflowError,RecursionError):
                unknown+=1;continue
            strict=math.isclose(av,bv,rel_tol=1e-8,abs_tol=1e-8)
            decimal=re.fullmatch(r'[-+]?\d*\.(\d+)',b)
            tolerance=.50001*10**(-len(decimal.group(1))) if decimal else 1e-8
            rounded=not strict and abs(av-bv)<=tolerance
            checked.append({'ok':strict or rounded,'strict':strict,'rounded':rounded,
                            'value':bv,'fraction':'/' in a+b,'chain':len(parts)>2})
    return checked,unknown


def candidate_features(text,answer):
    checks,unknown=equalities(text);n=len(checks);bad=sum(not c['ok'] for c in checks)
    try:value=float(answer) if '/' not in str(answer) else __import__('fractions').Fraction(answer)
    except (ValueError,TypeError,ZeroDivisionError):value=None
    last=checks[-1] if checks else None
    terminal=last is not None and value is not None and math.isclose(float(last['value']),float(value),rel_tol=1e-5,abs_tol=1e-5)
    return [math.log1p(n),math.log1p(bad),math.log1p(sum(c['strict'] for c in checks)),
            math.log1p(sum(c['rounded'] for c in checks)),math.log1p(unknown),n/max(n+unknown,1),
            bad/max(n,1),math.log1p(sum(c['fraction'] for c in checks)),
            math.log1p(sum(c['chain'] for c in checks)),int(terminal and last['ok']),
            int(terminal and not last['ok']),bad/max(n,1)]


def extract_arithmetic(bank,answers):
    out=np.asarray([[candidate_features(c['text'],answers[n][i]) for i,c in enumerate(row['candidates'])]
                    for n,row in enumerate(bank)],dtype=np.float32).reshape(len(bank),4,len(NAMES))
    if len(bank):out[...,-1]-=out[...,-1].mean(axis=1,keepdims=True)
    return out


def verify_synthetic():
    cases=[('3 + 5 + 7 = 15',1,0),('3 + 5 + 7 = 16',1,1),
           (r'\frac{0.6}{9} = \frac{2}{15}',1,1),
           (r'\frac{0.6}{9} = 0.0667',1,0),
           ('2 * 3 = 6 = 2 + 4',2,0),('20 * 35% = 7',1,0),
           ('x + 2 = 8',0,0),('1 / 0 = 2',0,0)]
    for text,n,bad in cases:
        rows,_=equalities(text)
        assert (len(rows),sum(not c['ok'] for c in rows))==(n,bad),(text,rows)
    return {'cases':len(cases),'status':'passed','scope':'Bounded arithmetic, fractions, chains, rounding, percent, unsupported/division-zero unknown.'}
