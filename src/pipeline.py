"""Synthetic RIR research prototype. No human recordings are used."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["velocity", "velocity_loss", "tremor_amplitude", "tremor_band_power"]


def generate(seed=17, participants=24):
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(participants):
        person_speed = rng.normal(0, .045)
        person_tremor = rng.normal(0, .018)
        for exercise in ("squat", "bench press"):
            for set_no in range(4):
                total = int(rng.integers(8, 15))
                load = float(rng.uniform(.55, .85))
                base = (0.84 if exercise == "squat" else 0.69) - .24 * (load - .55) + person_speed
                fastest = 0.0
                for rep in range(1, total + 1):
                    progress = rep / total
                    velocity = max(.12, base - .34 * progress ** 1.8 + rng.normal(0, .018))
                    fastest = max(fastest, velocity)
                    loss = 1 - velocity / fastest
                    tremor = max(0.005, .028 + .17 * progress ** 2 + person_tremor + rng.normal(0, .012))
                    band_power = max(0.0001, tremor ** 2 * (1 + rng.normal(0, .12)))
                    rows.append(dict(participant=f"P{person:02d}", set_id=f"P{person:02d}-{exercise}-{set_no}",
                                     exercise=exercise, load_fraction=load, rep=rep, rir=total-rep,
                                     velocity=velocity, velocity_loss=loss, tremor_amplitude=tremor,
                                     tremor_band_power=band_power, source="SYNTHETIC"))
    return pd.DataFrame(rows)


def split_participants(df):
    ids = sorted(df.participant.unique())
    return {"train": ids[:16], "validation": ids[16:20], "test": ids[20:]}


def make_examples(df, combined=True):
    # Each row sees only its current and earlier repetitions; zero padding has a mask.
    channels = FEATURES if combined else FEATURES[:2]
    X, y, groups, details = [], [], [], []
    for _, group in df.groupby("set_id", sort=False):
        group = group.sort_values("rep")
        values = group[channels].to_numpy(float)
        for i, (_, row) in enumerate(group.iterrows()):
            history = np.zeros((4, len(channels)))
            mask = np.zeros(4)
            segment = values[max(0, i-3):i+1]
            history[-len(segment):] = segment
            mask[-len(segment):] = 1
            extras = [float(row.exercise == "squat"), float(row.load_fraction), float(row.rep)]
            X.append(np.r_[history.ravel(), mask, extras])
            y.append(float(row.rir))
            groups.append(row.participant)
            details.append((row.set_id, int(row.rep)))
    return np.array(X), np.array(y), np.array(groups), details


def scale_fit(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1
    return mean, std


def init_model(n_features, seed):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, np.sqrt(2/n_features), (n_features, 32)), "b1": np.zeros(32),
            "W2": rng.normal(0, np.sqrt(2/32), (32, 16)), "b2": np.zeros(16),
            "W3": rng.normal(0, np.sqrt(2/16), (16, 1)), "b3": np.zeros(1)}


def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(z2, 0)
    return (a2 @ p["W3"] + p["b3"]).ravel(), (X, z1, a1, z2, a2)


def fit(X, y, Xval, yval, seed=11, epochs=450, lr=.003):
    p = init_model(X.shape[1], seed)
    first = {k: np.zeros_like(v) for k,v in p.items()}
    second = {k: np.zeros_like(v) for k,v in p.items()}
    best, best_loss, patience, step = None, float("inf"), 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        for ix in np.array_split(rng.permutation(len(X)), max(1, int(np.ceil(len(X)/128)))):
            step += 1
            pred, (xb,z1,a1,z2,a2) = forward(X[ix], p)
            delta = (2 / len(ix) * (pred-y[ix]))[:, None]
            grads = {"W3": a2.T @ delta, "b3": delta.sum(axis=0)}
            dz2 = (delta @ p["W3"].T) * (z2 > 0)
            grads.update(W2=a1.T @ dz2, b2=dz2.sum(axis=0))
            dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
            grads.update(W1=xb.T @ dz1, b1=dz1.sum(axis=0))
            for k in p:
                first[k] = .9*first[k] + .1*grads[k]
                second[k] = .999*second[k] + .001*grads[k]**2
                p[k] -= lr * (first[k]/(1-.9**step)) / (np.sqrt(second[k]/(1-.999**step)) + 1e-8)
        val = np.mean((forward(Xval,p)[0]-yval)**2)
        if val < best_loss - 1e-4:
            best_loss, best, patience = val, {k:v.copy() for k,v in p.items()}, 0
        else:
            patience += 1
            if patience >= 45: break
    return best, epoch+1


def metrics(y, pred):
    pred = np.maximum(pred, 0)
    actual_warning = y <= 2
    warned = pred <= 2
    tp = int(np.sum(actual_warning & warned))
    fp = int(np.sum(~actual_warning & warned))
    fn = int(np.sum(actual_warning & ~warned))
    tn = int(np.sum(~actual_warning & ~warned))
    return {"MAE": round(float(np.mean(np.abs(y-pred))),3),
            "RMSE": round(float(np.sqrt(np.mean((y-pred)**2))),3),
            "warning_precision": round(tp/(tp+fp),3) if tp+fp else 0,
            "warning_recall": round(tp/(tp+fn),3) if tp+fn else 0,
            "warning_F1": round(2*tp/(2*tp+fp+fn),3) if 2*tp+fp+fn else 0,
            "false_positive_rate": round(fp/(fp+tn),3) if fp+tn else 0,
            "missed_warnings": fn, "n": len(y)}


def train_branch(df, combined, splits):
    X,y,groups,_ = make_examples(df, combined)
    masks = {name: np.isin(groups, ids) for name,ids in splits.items()}
    mean,std = scale_fit(X[masks["train"]])
    Z = (X-mean)/std
    model, epochs = fit(Z[masks["train"]], y[masks["train"]], Z[masks["validation"]], y[masks["validation"]])
    pred = forward(Z[masks["test"]], model)[0]
    return {"model":model,"mean":mean,"std":std,"epochs":epochs,"metrics":metrics(y[masks["test"]],pred),
            "test_y":y[masks["test"]],"test_pred":np.maximum(pred,0),"test_groups":groups[masks["test"]]}


def save_model(path, result, combined):
    payload = {"combined":combined,"channels":FEATURES if combined else FEATURES[:2],
               "architecture":"4-rep causal history -> dense 32 ReLU -> dense 16 ReLU -> linear RIR; Adam + early stopping",
               "mean":result["mean"].tolist(),"std":result["std"].tolist(),
               "weights":{k:v.tolist() for k,v in result["model"].items()}}
    Path(path).write_text(json.dumps(payload))


def predict_history(records, artifact):
    channels = artifact["channels"]
    history = np.zeros((4,len(channels)))
    mask = np.zeros(4)
    segment = records[-4:]
    for j,r in enumerate(segment,4-len(segment)):
        history[j] = [float(r[c]) for c in channels]
        mask[j] = 1
    latest = records[-1]
    x = np.r_[history.ravel(), mask, float(latest["exercise"] == "squat"),float(latest["load_fraction"]),float(latest["rep"])]
    z = (x-np.array(artifact["mean"]))/np.array(artifact["std"])
    p = {k:np.array(v) for k,v in artifact["weights"].items()}
    return round(float(max(0,forward(z[None,:],p)[0][0])),2)
