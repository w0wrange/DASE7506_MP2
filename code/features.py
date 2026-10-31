"""Input-only features; no reference files, question IDs or learned constants."""
import ast
from collections import Counter
from fractions import Fraction
import math
import re
import numpy as np
from common import parse_answer

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORDS = re.compile(r"[a-z]+")
EQUATION = re.compile(r"(?<![\w.])([\d.()+*/ \-]{3,80})\s*=\s*([-+]?\d+(?:\.\d+)?)(?![\w.])")
KEYWORDS = ("total", "each", "remaining", "left", "half", "twice", "percent", "per", "more", "less", "however", "therefore", "check", "wait", "mistake", "assume")
FEATURE_NAMES = ["valid", "answer_votes", "vote_share", "distinct_answers", "valid_count", "max_votes", "vote_margin", "answer_abs_log", "answer_signed_log", "negative", "integer", "zero", "in_question", "answer_question_ratio", "answer_rank", "agree_similarity", "other_similarity", "mean_similarity", "embedding_norm", "norm_ratio", "char_log", "word_log", "line_log", "digit_share", "number_log", "unique_number_log", "question_number_coverage", "number_overlap", "question_word_coverage", "word_overlap", "answer_mentions", "answer_at_end", "boxed", "hash_answer", "dollar_log", "equation_log", "arithmetic_checked", "arithmetic_errors", "arithmetic_correct_share", "multiply_log", "divide_log", "question_marks", "bold", "bullets", "latex_fraction", "length_relative", "number_relative"] + ["kw_"+k for k in KEYWORDS]


def numeric_values(text):
    result = []
    for token in NUMBER.findall(text):
        try:
            value = Fraction(token.replace(",", ""))
            if value.numerator.bit_length() <= 256 and value.denominator.bit_length() <= 256:
                result.append(value)
        except (ValueError, ZeroDivisionError): pass
    return result


def safe_arithmetic(expression):
    """Bounded arithmetic AST only; never execute candidate code."""
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 40: raise ValueError("Too large")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a+b
            elif isinstance(node.op, ast.Sub): value = a-b
            elif isinstance(node.op, ast.Mult): value = a*b
            else: value = a/b
        else: raise ValueError("Unsupported expression")
        if not math.isfinite(value) or abs(value) > 1e15: raise ValueError("Unbounded")
        return value
    return visit(tree.body)


def extract(bank, embeddings):
    """Return [N,4,F] features, validity mask, parser-normalised answers."""
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=-1)
    unit = emb/np.maximum(norms[..., None], 1e-8)
    sims = np.einsum("nid,njd->nij", unit, unit)
    rows, answer_rows = [], []
    for n, row in enumerate(bank):
        q = row["question"].lower(); qnums = set(numeric_values(q)); qwords = set(WORDS.findall(q))
        texts = [c["text"] for c in row["candidates"]]
        answers = [parse_answer(t) for t in texts]; answer_rows.append(answers)
        votes = Counter(a for a in answers if a is not None)
        counts = sorted(votes.values(), reverse=True)+[0, 0]; nv = sum(votes.values())
        nums = [numeric_values(t) for t in texts]
        lengths = [len(t) for t in texts]; number_counts = [len(a) for a in nums]
        ranks = sorted({Fraction(a) for a in votes}); qmax = max([abs(float(v)) for v in qnums], default=1.)
        group = []
        for i, text in enumerate(texts):
            lower = text.lower(); words = WORDS.findall(lower); wordset = set(words); numset = set(nums[i])
            answer = answers[i]; value = Fraction(answer) if answer is not None else Fraction(0)
            af = float(np.clip(float(value), -1e15, 1e15))
            same = [j for j in range(4) if j != i and answer is not None and answers[j] == answer]
            other = [j for j in range(4) if j != i and j not in same]
            checked = errors = 0
            plain = text.replace("×", "*").replace("÷", "/").replace("\\times", "*").replace("\\div", "/")
            for match in EQUATION.finditer(plain):
                expr = match.group(1).strip()
                if not re.search(r"[+*/]|\d\s*-\s*\d", expr): continue
                try:
                    a, b = safe_arithmetic(expr), float(match.group(2)); checked += 1
                    errors += int(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5))
                except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError): pass
            f = [answer is not None, votes.get(answer, 0), votes.get(answer, 0)/max(nv, 1), len(votes), nv, counts[0], counts[0]-counts[1],
                 math.log1p(abs(af)), math.copysign(math.log1p(abs(af)), af), af < 0, value.denominator == 1, af == 0,
                 answer is not None and value in qnums, math.log1p(abs(af)/max(qmax, 1e-6)), ranks.index(value)/max(len(ranks)-1, 1) if answer is not None else -1,
                 float(sims[n, i, same].mean()) if same else 0, float(sims[n, i, other].mean()) if other else 0,
                 float((sims[n, i].sum()-1)/3), float(norms[n, i]), float(norms[n, i]/max(norms[n].mean(), 1e-6)),
                 math.log1p(len(text)), math.log1p(len(words)), math.log1p(text.count("\n")), sum(c.isdigit() for c in text)/max(len(text), 1),
                 math.log1p(len(nums[i])), math.log1p(len(numset)), len(qnums & numset)/max(len(qnums), 1), len(qnums & numset)/max(len(numset), 1),
                 len(qwords & wordset)/max(len(qwords), 1), len(qwords & wordset)/max(len(wordset), 1), math.log1p(sum(v == value for v in nums[i]) if answer is not None else 0),
                 int(answer is not None and parse_answer(text[-180:]) == answer), "\\boxed" in text, "####" in text,
                 math.log1p(text.count("$")), math.log1p(text.count("=")), math.log1p(checked), math.log1p(errors), (checked-errors)/max(checked, 1),
                 math.log1p(plain.count("*")), math.log1p(plain.count("/")), math.log1p(text.count("?")), math.log1p(text.count("**")),
                 math.log1p(len(re.findall(r"(?m)^\s*[-*\d]+[.) ]", text))), math.log1p(text.count("\\frac")), len(text)/max(np.mean(lengths), 1), len(nums[i])/max(np.mean(number_counts), 1),
                 ] + [math.log1p(len(re.findall(r"\b"+k+r"\b", lower))) for k in KEYWORDS]
            group.append(f)
        rows.append(group)
    features = np.asarray(rows, dtype=np.float32).reshape(len(bank), 4, len(FEATURE_NAMES))
    return features, features[..., 0].astype(bool), answer_rows


def choose(scores, valid, answers):
    scores = np.where(valid, scores, -1e20)
    result = []
    for n, row in enumerate(answers):
        if len({a for a in row if a is not None}) == 1: result.append(next(i for i, a in enumerate(row) if a is not None))
        else: result.append(int(np.argmax(scores[n])))
    return result


def probabilities(scores, valid):
    x = np.where(valid, np.asarray(scores, dtype=np.float64), -1e20)
    x -= x.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(x, -80, 0))*valid
    return (exp/np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def hashed_text(bank, answers, mode, analyzer="word"):
    from sklearn.feature_extraction.text import HashingVectorizer
    docs = [row["question"]+"\n"+(c["text"] if mode == "full" else "answer "+str(answers[n][i])) for n, row in enumerate(bank) for i, c in enumerate(row["candidates"])]
    return HashingVectorizer(n_features=2**18, analyzer=analyzer, ngram_range=(1, 2) if analyzer == "word" else (3, 5), norm="l2", alternate_sign=False, dtype=np.float32).transform(docs)


def tree_features(embeddings, features, relative=True, structural=True):
    parts = [embeddings]
    if relative: parts.append(embeddings-embeddings.mean(axis=1, keepdims=True))
    if structural: parts.append(features)
    return np.concatenate(parts, axis=-1).reshape(-1, sum(p.shape[-1] for p in parts)).astype(np.float32)


def structural_features(features, comparison_features=True):
    """Controlled ablation: remove every feature depending on other candidates."""
    if comparison_features: return features
    group_names = {"answer_votes", "vote_share", "distinct_answers", "valid_count", "max_votes", "vote_margin", "answer_rank", "agree_similarity", "other_similarity", "mean_similarity", "norm_ratio", "length_relative", "number_relative"}
    result = features.copy()
    result[..., [i for i, name in enumerate(FEATURE_NAMES) if name in group_names]] = 0
    return result
