"""
Treniravimas: Baseline (taisyklės, logistinė regresija) ir Proposed (LSTM).
Slenkstis kiekvienam modeliui parenkamas pagal geriausią F1 validacijos aibėje.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from config import DATA_PROC, FEATURES, MODELS, RESULTS, SEED, SEQ_LEN
from models import LSTMFailurePredictor, lstm_predict_proba, rule_dashboard, rule_expert
from preprocess import make_sequences

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def train_lstm(Xtr, ytr, Xva, yva, epochs=30, patience=6, lr=1e-3, batch=256):
    model = LSTMFailurePredictor(Xtr.shape[2])
    pos_weight = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    history, best, best_state, bad = [], -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        pva = lstm_predict_proba(model, Xva)
        with torch.no_grad():
            val_loss = loss_fn(torch.logit(torch.from_numpy(pva).clamp(1e-6, 1 - 1e-6)), torch.from_numpy(yva)).item()
        ap = average_precision_score(yva, pva)
        history.append(dict(epoch=ep, train_loss=tot / len(Xt), val_loss=val_loss, val_pr_auc=ap))
        print(f"epocha {ep:2d}  train_loss {tot/len(Xt):.4f}  val_loss {val_loss:.4f}  val PR-AUC {ap:.3f}")
        if ap > best:
            best, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("Ankstyvas sustabdymas (early stopping)")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def main():
    minutes = pd.read_csv(DATA_PROC / "minutes.csv.gz")
    tr = minutes[minutes.split == "train"]
    mean = tr[FEATURES].mean().to_numpy()
    std = tr[FEATURES].std().replace(0, 1).to_numpy()
    json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES, "seq_len": SEQ_LEN},
              open(MODELS / "scaler.json", "w"), indent=2)

    seqs = {}
    for s in ("train", "val", "test"):
        X, y, idx = make_sequences(minutes[minutes.split == s], mean, std)
        seqs[s] = (X, y, idx)
        print(f"{s:5s}: {X.shape} sekų, teigiamų {int(y.sum())} ({y.mean():.1%})")

    # Visi modeliai vertinami tose pačiose minutėse (ten, kur LSTM turi pilną 30 min. istoriją)
    thresholds, preds = {}, {}
    for s in ("val", "test"):
        sub = minutes.loc[seqs[s][2]].copy()
        sub["p_rule_dashboard"] = rule_dashboard(sub)
        sub["p_rule_expert"] = rule_expert(sub)
        preds[s] = sub

    # --- Baseline: logistinė regresija (tik dabartinė minutė, be istorijos) ---
    Xtr_flat = ((tr[FEATURES].to_numpy() - mean) / std)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(Xtr_flat, tr["label"])
    joblib.dump(logreg, MODELS / "logreg.joblib")
    for s in ("val", "test"):
        preds[s]["p_logreg"] = logreg.predict_proba((preds[s][FEATURES].to_numpy() - mean) / std)[:, 1]
    thresholds["logreg"], _ = best_threshold(preds["val"]["label"], preds["val"]["p_logreg"])

    # --- Proposed: LSTM ---
    t0 = time.time()
    model, hist = train_lstm(seqs["train"][0], seqs["train"][1], seqs["val"][0], seqs["val"][1])
    train_time = time.time() - t0
    torch.save(model.state_dict(), MODELS / "lstm.pt")
    hist.to_csv(RESULTS / "lstm_history.csv", index=False)
    for s in ("val", "test"):
        preds[s]["p_lstm"] = lstm_predict_proba(model, seqs[s][0])
    thresholds["lstm"], _ = best_threshold(preds["val"]["label"], preds["val"]["p_lstm"])
    thresholds["rule_dashboard"] = 0.5
    thresholds["rule_expert"] = 0.5

    # inferencijos greitis vienai sekai
    x1 = torch.from_numpy(seqs["test"][0][:1])
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        t = time.perf_counter()
        for _ in range(200):
            model(x1)
        infer_ms = (time.perf_counter() - t) / 200 * 1000

    json.dump({"thresholds": thresholds, "lstm_train_seconds": round(train_time, 1),
               "lstm_inference_ms": round(infer_ms, 3),
               "lstm_params": sum(p.numel() for p in model.parameters())},
              open(MODELS / "thresholds.json", "w"), indent=2)
    for s in ("val", "test"):
        preds[s].to_csv(RESULTS / f"{s}_predictions.csv.gz", index=False)
    print("Slenksčiai:", thresholds, f"| LSTM treniravimas {train_time:.0f} s, inferencija {infer_ms:.2f} ms")


if __name__ == "__main__":
    main()
