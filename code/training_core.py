"""Training helpers. References are read only for train/validation, never test."""
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl
from features import choose, extract, hashed_text, probabilities, tree_features, structural_features


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n"); temporary.replace(path)


def dataset(split, cache_dir=None):
    if split not in ("train", "validation"): raise ValueError("Training cannot open test references")
    bank, arrays = load_inputs(split)
    e = arrays["embeddings"]
    cache = Path(cache_dir)/f"{split}-features.npz" if cache_dir else None
    # Cache is development-only, never part of inference assets.
    f, valid, answers = extract(bank, e)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, features=f)
    refs = read_jsonl(DATA/f"gsm8k_{split}.jsonl")
    assert all(r["id"] == b["id"] and r["question"] == b["question"] for r, b in zip(refs, bank))
    y = np.array([[a is not None and a == parse_answer(ref["answer"]) for a in row] for row, ref in zip(answers, refs)], dtype=np.float32)
    assert y.shape == (len(bank), 4) and (y.sum(axis=1) >= 1).all()
    hard = np.array([len({a for a in row if a is not None}) >= 2 for row in answers])
    return {"bank": bank, "e": e, "f": f, "valid": valid, "answers": answers, "y": y, "hard": hard}


def subset(data, indices):
    return {k: [v[int(i)] for i in indices] if k in ("bank", "answers") else v[indices] for k, v in data.items()}


def evaluate(data, scores):
    choices = choose(scores, data["valid"], data["answers"])
    hits = data["y"][np.arange(len(choices)), choices]
    p = probabilities(scores, data["valid"])
    mass = (p*data["y"]).sum(axis=1)
    hard = data["hard"]
    return {"n": len(choices), "correct": int(hits.sum()), "accuracy": float(100*hits.mean()),
            "hard_n": int(hard.sum()), "hard_correct": int(hits[hard].sum()), "hard_accuracy": float(100*hits[hard].mean()) if hard.any() else 100.,
            "correct_mass_nll": float(-np.log(np.maximum(mass, 1e-8)).mean())}


def asset_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def fit(config, train, dev, destination, seed=42, fixed_epochs=None, device="cuda"):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    config = dict(config); config["seed"] = seed
    start = time.perf_counter(); kind = config["kind"]
    weights = {}
    sample_weights = np.repeat(np.where(train["hard"], 2., .5), 4)
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression
        x = train["e"].reshape(-1, 1024)
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
        model = LogisticRegression(C=config.get("C", .1), max_iter=400, solver="lbfgs", random_state=seed)
        model.fit((x-mean)/scale, train["y"].ravel(), sample_weight=sample_weights)
        weights = {"mean": mean, "scale": scale, "coef": model.coef_[0].astype(np.float32), "bias": model.intercept_.astype(np.float32)}
    elif kind == "sparse":
        from sklearn.linear_model import LogisticRegression
        x = hashed_text(train["bank"], train["answers"], config["text_mode"], config["analyzer"])
        model = LogisticRegression(C=config["C"], solver="liblinear", max_iter=150, random_state=seed)
        model.fit(x, train["y"].ravel(), sample_weight=sample_weights)
        weights = {"coef": model.coef_[0].astype(np.float32), "bias": model.intercept_.astype(np.float32)}
    elif kind == "catboost":
        from catboost import CatBoostClassifier
        x = tree_features(train["e"], train["f"], config.get("relative", True), config.get("structural", True))
        params = {"iterations": int(fixed_epochs or 1200), "depth": config["depth"], "learning_rate": config["learning_rate"], "l2_leaf_reg": config["l2_leaf_reg"],
                  "loss_function": "Logloss", "random_seed": seed, "thread_count": 4, "verbose": False, "allow_writing_files": False,
                  "task_type": "GPU" if device == "cuda" else "CPU"}
        model = CatBoostClassifier(**params)
        options = {}
        if dev is not None and fixed_epochs is None:
            xd = tree_features(dev["e"], dev["f"], config.get("relative", True), config.get("structural", True))
            options = {"eval_set": (xd, dev["y"].ravel()), "early_stopping_rounds": 100, "use_best_model": True}
        model.fit(x, train["y"].ravel(), sample_weight=sample_weights, **options)
        config["epochs"] = int(model.tree_count_)
        model.save_model(str(destination/"weights.cbm"))
    elif kind == "mlp":
        weights, epochs, parity = fit_mlp(config, train, dev, seed, fixed_epochs, device)
        config["epochs"] = epochs; config["export_max_abs_error"] = parity
    else: raise ValueError(kind)
    if weights: np.savez(destination/"weights.npz", **weights)
    write_json(destination/"model.json", config)
    from submission import model_scores
    if dev is None: metrics, scores = {}, None
    else:
        scores = model_scores(dev["bank"], dev["e"], dev["f"], dev["answers"], destination)
        metrics = evaluate(dev, scores)
    record = {"config": config, "metrics": metrics, "seconds": time.perf_counter()-start, "asset_bytes": asset_bytes(destination)}
    write_json(destination/"training.json", record)
    return record, scores


def fit_mlp(config, train, dev, seed, fixed_epochs, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    import torch.nn.functional as F
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Requested CUDA unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    width = config["width"]; dropout = config["dropout"]
    relative, structural = config.get("relative", True), config.get("structural", True)
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(1024, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 64), nn.ReLU())
            size = 64*(3 if relative else 1)+(train["f"].shape[-1] if structural else 0)
            self.head = nn.Sequential(nn.Linear(size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        def forward(self, e, f):
            h = self.encoder(e); parts = [h]
            if relative:
                avg = h.mean(dim=1, keepdim=True).expand_as(h); parts += [avg, h-avg]
            if structural: parts.append(f)
            return self.head(torch.cat(parts, dim=-1)).squeeze(-1)
    mean = train["e"].mean(axis=(0, 1)); scale = np.maximum(train["e"].std(axis=(0, 1)), .05)
    train_features = structural_features(train["f"], config.get("comparison_features", True))
    fm = train_features.mean(axis=(0, 1)); fs = np.maximum(train_features.std(axis=(0, 1)), .05)
    def tensors(data):
        return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (
            np.clip((data["e"]-mean)/scale, -8, 8), np.clip((structural_features(data["f"], config.get("comparison_features", True))-fm)/fs, -8, 8), data["y"], data["valid"], np.where(data["hard"], 2., .5))]
    te, tf, ty, tv, tw = tensors(train)
    dv = tensors(dev) if dev is not None else None
    model = Ranker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_key = (-1., -float("inf")); best_state = None; best_epoch = 0; stale = 0
    max_epochs = int(fixed_epochs or 150)
    for epoch in range(1, max_epochs+1):
        model.train()
        order = torch.randperm(len(te), device=device)
        for indices in order.split(256):
            logits = model(te[indices], tf[indices]); y = ty[indices]; valid = tv[indices].bool()
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=1)
            masked = logits.masked_fill(~valid, -1e4)
            mass = torch.logsumexp(masked, dim=1)-torch.logsumexp(masked.masked_fill(y < .5, -1e4), dim=1)
            loss_name = config["loss"]
            losses = bce if loss_name == "bce" else (mass if loss_name == "listwise" else bce+.5*mass)
            loss = (losses*tw[indices]).sum()/tw[indices].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        if fixed_epochs is not None or dev is None: continue
        model.eval()
        with torch.no_grad(): scores = model(dv[0], dv[1]).cpu().numpy()
        metrics = evaluate(dev, scores); key = (metrics["hard_accuracy"], -metrics["correct_mass_nll"])
        if key > best_key:
            best_key = key; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= 18: break
    if best_state is not None: model.load_state_dict(best_state)
    else: best_epoch = max_epochs
    model.eval()
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    weights.update(mean=mean, scale=scale, fmean=fm, fscale=fs)
    # Independent NumPy parity check on a small slice before accepting export.
    check = dev if dev is not None else train
    ce = np.clip((check["e"][:16]-mean)/scale, -8, 8); cf = np.clip((structural_features(check["f"][:16], config.get("comparison_features", True))-fm)/fs, -8, 8)
    with torch.no_grad(): expected = model(torch.tensor(ce, device=device), torch.tensor(cf, device=device)).cpu().numpy()
    h = np.maximum(ce @ weights["encoder.0.weight"].T+weights["encoder.0.bias"], 0)
    h = np.maximum(h @ weights["encoder.3.weight"].T+weights["encoder.3.bias"], 0); parts = [h]
    if relative:
        avg = np.broadcast_to(h.mean(axis=1, keepdims=True), h.shape); parts += [avg, h-avg]
    if structural: parts.append(cf)
    h = np.maximum(np.concatenate(parts, axis=-1) @ weights["head.0.weight"].T+weights["head.0.bias"], 0)
    actual = (h @ weights["head.3.weight"].T+weights["head.3.bias"])[..., 0]
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-3: raise ValueError(f"NumPy export differs: {error}")
    return weights, best_epoch, error
