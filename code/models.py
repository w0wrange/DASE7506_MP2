"""Numerically unchanged member scoring; release uses linear and MLP only."""
import json
from pathlib import Path
import numpy as np
from features import tree_features,hashed_text,structural_features

def model_scores(bank, embeddings, features, answers, directory):
    directory = Path(directory)
    config = json.loads((directory/"model.json").read_text())
    kind = config["kind"]
    if kind == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(); model.load_model(str(directory/"weights.cbm"))
        x = tree_features(embeddings, features, config.get("relative", True), config.get("structural", True))
        return model.predict(x, prediction_type="RawFormulaVal", thread_count=4).reshape(-1, 4)
    with np.load(directory/"weights.npz", allow_pickle=False) as z: w = {k: z[k] for k in z.files}
    if kind == "linear":
        return (((embeddings-w["mean"])/w["scale"]) @ w["coef"]+w["bias"]).reshape(-1, 4)
    if kind == "sparse":
        x = hashed_text(bank, answers, config["text_mode"], config["analyzer"])
        return (x @ w["coef"]+w["bias"]).reshape(-1, 4)
    if kind == "mlp":
        x = np.clip((embeddings-w["mean"])/w["scale"], -8, 8)
        h = np.maximum(x @ w["encoder.0.weight"].T+w["encoder.0.bias"], 0)
        h = np.maximum(h @ w["encoder.3.weight"].T+w["encoder.3.bias"], 0)
        parts = [h]
        if config.get("relative", True):
            avg = np.broadcast_to(h.mean(axis=1, keepdims=True), h.shape)
            parts += [avg, h-avg]
        if config.get("structural", True): parts.append(np.clip((structural_features(features, config.get("comparison_features", True))-w["fmean"])/w["fscale"], -8, 8))
        v = np.concatenate(parts, axis=-1)
        v = np.maximum(v @ w["head.0.weight"].T+w["head.0.bias"], 0)
        return (v @ w["head.3.weight"].T+w["head.3.bias"])[..., 0]
    raise ValueError(f"Unknown model kind: {kind}")
