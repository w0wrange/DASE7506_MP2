"""Aggregate input-derived answer groups; no labels or reference files at inference."""
import numpy as np
from features import choose


RULES = [{"mode": "candidate_max"}]+[{"mode": "answer_power", "alpha": a} for a in (0., .25, .5, .75, 1.)]


def rule_name(rule):
    return "candidate_max" if rule["mode"] == "candidate_max" else f"answer_power_{rule['alpha']:g}"


def select_answers(p, valid, answers, rule):
    if rule["mode"] == "candidate_max":
        return choose(p, valid, answers)
    if rule["mode"] != "answer_power": raise ValueError("Unknown aggregation rule")
    alpha = float(rule["alpha"])
    if not 0 <= alpha <= 1: raise ValueError("alpha must be in [0, 1]")
    result = []
    for n, row in enumerate(answers):
        groups = {}
        for i, answer in enumerate(row):
            if valid[n, i] and answer is not None: groups.setdefault(answer, []).append(i)
        if not groups:
            result.append(0); continue
        if len(groups) == 1:
            result.append(next(iter(groups.values()))[0]); continue
        # Stable ties follow first occurrence of an answer; choose its strongest trace.
        indices = max(groups.values(), key=lambda ix: float(np.sum(p[n, ix], dtype=np.float64))/len(ix)**alpha)
        result.append(int(max(indices, key=lambda i: float(p[n, i]))))
    return result


def summarize(data, selected):
    selected = np.asarray(selected, dtype=int)
    hit = data["y"][np.arange(len(selected)), selected] > .5
    hard = data["hard"]
    return {"n": len(selected), "correct": int(hit.sum()), "accuracy": 100*float(hit.mean()),
            "hard_n": int(hard.sum()), "hard_correct": int(hit[hard].sum()),
            "hard_accuracy": 100*float(hit[hard].mean())}


def paired(data, reference, proposed):
    from math import comb
    y = data["y"]; rows = np.arange(len(y))
    a = y[rows, np.asarray(reference, int)] > .5
    b = y[rows, np.asarray(proposed, int)] > .5
    hard = data["hard"]
    gains = int((hard & ~a & b).sum()); losses = int((hard & a & ~b).sum())
    m = gains+losses
    p = min(1., 2*sum(comb(m, i) for i in range(min(gains, losses)+1))/2**m) if m else 1.
    return {"hard_gains": gains, "hard_losses": losses, "hard_net": gains-losses,
            "hard_delta_pp": 100*(gains-losses)/int(hard.sum()), "mcnemar_exact_two_sided": p,
            "interpretation": "Descriptive paired comparison; not adjusted for model selection or multiple comparisons."}
