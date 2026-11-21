"""
Vertinimas ir grafikai: Baseline vs Proposed palyginimas testinėje aibėje
(automobiliai, kurių modelis niekada nematė).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from config import DATA_PROC, DATA_RAW, FEATURES, FIGURES, HORIZON_KM, MODELS, RESULTS
from models import LSTMFailurePredictor, alarms_from_scores, lstm_predict_proba
from preprocess import make_sequences

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C_HEALTHY, C_FAULT, C_LSTM, C_LR, C_RULE = "#4C78A8", "#E45756", "#2E8B57", "#F58518", "#9D755D"

METHODS = {
    "rule_dashboard": "Taisyklė: skydelio lemputė (≥105 °C)",
    "rule_expert": "Taisyklės: diagnosto žinios",
    "logreg": "Logistinė regresija (be istorijos)",
    "lstm": "LSTM (30 min. istorija)",
}
FAULT_LT = {"healthy": "Sveikas", "cooling_degradation": "Aušinimo sistemos degradacija",
            "fan_failure": "Ventiliatoriaus gedimas", "thermostat_stuck_open": "Užstrigęs atviras termostatas",
            "misfire": "Uždegimo praleidimai"}


def minute_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    r = dict(Accuracy=accuracy_score(y, yhat), Precision=precision_score(y, yhat, zero_division=0),
             Recall=recall_score(y, yhat), F1=f1_score(y, yhat))
    probabilistic = len(np.unique(p)) > 2
    r["ROC_AUC"] = roc_auc_score(y, p) if probabilistic else np.nan
    r["PR_AUC"] = average_precision_score(y, p) if probabilistic else np.nan
    return r


def vehicle_level(df, col, thr):
    """Aliarmas = 3 minutės iš eilės virš slenksčio. Skaičiuojama kiekvienam automobiliui."""
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        alarm = alarms_from_scores((g[col].to_numpy() >= thr), consecutive=3)
        faulty = g["fault_type"].iloc[0] != "healthy"
        km = g["odometer_km"].to_numpy()
        ktf = g["km_to_failure"].to_numpy()
        in_h = faulty & (ktf <= HORIZON_KM) if faulty else np.zeros(len(g), bool)
        # klaidingi aliarmai: aliarmo pradžios (epizodai) ne gedimo horizonte
        starts = alarm & ~np.r_[False, alarm[:-1]]
        false_eps = int((starts & ~in_h).sum())
        normal_km = float(np.diff(km, prepend=km[0])[~in_h].sum())
        detected = bool((alarm & in_h).any()) if faulty else None
        lead = float(ktf[alarm & in_h].max()) if detected else None
        rows.append(dict(vehicle_id=vid, fault_type=g["fault_type"].iloc[0], detected=detected,
                         lead_km=lead, false_alarm_episodes=false_eps, normal_km=normal_km))
    return pd.DataFrame(rows)


def main():
    th = json.load(open(MODELS / "thresholds.json"))
    thr = th["thresholds"]
    test = pd.read_csv(RESULTS / "test_predictions.csv.gz")
    y = test["label"].to_numpy()

    # ---------- 1. Minučių lygio metrikos ----------
    table = {"Dažniausia klasė (visada 'sveika')": minute_metrics(y, np.zeros_like(y, float), 0.5)}
    table["Dažniausia klasė (visada 'sveika')"]["ROC_AUC"] = 0.5
    for m, name in METHODS.items():
        table[name] = minute_metrics(y, test[f"p_{m}"].to_numpy(), thr[m])
    metrics = pd.DataFrame(table).T.round(3)
    metrics.to_csv(RESULTS / "metrics_minute_level.csv")
    print(metrics.to_string())

    # ---------- 2. Automobilio lygio metrikos ----------
    veh_rows = []
    per_vehicle = {}
    for m, name in METHODS.items():
        vl = vehicle_level(test, f"p_{m}", thr[m])
        per_vehicle[m] = vl
        f = vl[vl.fault_type != "healthy"]
        veh_rows.append({"Metodas": name,
                         "Aptikta gedimų": f"{int(f.detected.sum())}/{len(f)}",
                         "Aptikimo dalis": round(f.detected.mean(), 3),
                         "Vid. perspėjimas, km iki gedimo": round(f.lead_km.dropna().mean(), 1) if f.detected.any() else None,
                         "Klaidingi aliarmai / 1000 km": round(vl.false_alarm_episodes.sum() / vl.normal_km.sum() * 1000, 2)})
    vtable = pd.DataFrame(veh_rows)
    vtable.to_csv(RESULTS / "metrics_vehicle_level.csv", index=False)
    print(vtable.to_string())
    per_vehicle["lstm"].to_csv(RESULTS / "lstm_per_vehicle.csv", index=False)

    # ---------- 3. Recall pagal gedimo tipą ----------
    pos = test[test.label == 1]
    per_fault = pd.DataFrame({METHODS[m]: pos.groupby("fault_type").apply(lambda g: (g[f"p_{m}"] >= thr[m]).mean(), include_groups=False)
                              for m in METHODS}).round(3)
    per_fault.index = [FAULT_LT[i] for i in per_fault.index]
    per_fault.to_csv(RESULTS / "recall_per_fault.csv")
    print(per_fault.to_string())

    # ---------- 4. XAI: permutacinis požymių svarbumas (LSTM) ----------
    sc = json.load(open(MODELS / "scaler.json"))
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    minutes = pd.read_csv(DATA_PROC / "minutes.csv.gz")
    Xte, yte, _ = make_sequences(minutes[minutes.split == "test"], mean, std)
    model = LSTMFailurePredictor(len(FEATURES))
    model.load_state_dict(torch.load(MODELS / "lstm.pt"))
    base_ap = average_precision_score(yte, lstm_predict_proba(model, Xte))
    rng = np.random.default_rng(0)
    imp = {}
    for j, f in enumerate(FEATURES):
        drops = []
        for _ in range(3):
            Xp = Xte.copy()
            Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
            drops.append(base_ap - average_precision_score(yte, lstm_predict_proba(model, Xp)))
        imp[f] = float(np.mean(drops))
    imp = pd.Series(imp).sort_values()
    imp.to_csv(RESULTS / "feature_importance.csv", header=["pr_auc_drop"])

    json.dump({"minute_level": metrics.to_dict(orient="index"), "vehicle_level": veh_rows,
               "recall_per_fault": per_fault.to_dict(), "feature_importance": imp.to_dict(),
               "lstm_test_pr_auc": base_ap, **th}, open(RESULTS / "metrics.json", "w"), indent=2, ensure_ascii=False)

    # ================= GRAFIKAI =================
    # (a) LSTM mokymosi kreivės
    h = pd.read_csv(RESULTS / "lstm_history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, "-o", ms=3, label="Mokymo aibė")
    ax[0].plot(h.epoch, h.val_loss, "-o", ms=3, label="Validacijos aibė")
    ax[0].set(title="Nuostolių funkcija (BCE)", xlabel="Epocha"); ax[0].legend(frameon=False)
    best = h.val_pr_auc.idxmax()
    ax[1].plot(h.epoch, h.val_pr_auc, "-o", ms=3, color=C_LSTM)
    ax[1].axvline(h.epoch[best], ls="--", c="grey", lw=0.8)
    ax[1].annotate(f"išsaugotas modelis\n(epocha {h.epoch[best]})", (h.epoch[best], h.val_pr_auc[best]),
                   xytext=(8, -30), textcoords="offset points", fontsize=8)
    ax[1].set(title="Validacijos PR-AUC", xlabel="Epocha")
    fig.tight_layout(); fig.savefig(FIGURES / "06_lstm_training.png"); plt.close(fig)

    # (b) Konfuzijos matricos
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, m in zip(axes, ["rule_expert", "logreg", "lstm"]):
        cm = confusion_matrix(y, (test[f"p_{m}"] >= thr[m]).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sveika", "Gedimas ≤100 km"],
               yticklabels=["Sveika", "Gedimas ≤100 km"], xlabel="Prognozė", ylabel="Tikrovė")
        ax.set_title(METHODS[m], fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIGURES / "07_confusion_matrices.png"); plt.close(fig)

    # (c) Precision-Recall kreivės
    fig, ax = plt.subplots(figsize=(5, 4))
    for m, c in [("logreg", C_LR), ("lstm", C_LSTM)]:
        p, r, _ = precision_recall_curve(y, test[f"p_{m}"])
        ax.plot(r, p, c=c, label=f"{METHODS[m]} (PR-AUC {average_precision_score(y, test[f'p_{m}']):.2f})")
    for m, mk in [("rule_dashboard", "s"), ("rule_expert", "D")]:
        yhat = test[f"p_{m}"] >= 0.5
        ax.plot(recall_score(y, yhat), precision_score(y, yhat, zero_division=0), mk, c=C_RULE, label=METHODS[m])
    ax.axhline(y.mean(), ls=":", c="grey", lw=0.8); ax.text(0.6, y.mean() + 0.02, "atsitiktinis spėjimas", fontsize=7, c="grey")
    ax.set(xlabel="Recall (aptikta gedimų dalis)", ylabel="Precision (teisingų aliarmų dalis)",
           title="Precision–Recall kreivės (testinė aibė)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIGURES / "08_pr_curves.png"); plt.close(fig)

    # (d) Recall pagal gedimo tipą
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cols = [METHODS["rule_expert"], METHODS["logreg"], METHODS["lstm"]]
    x = np.arange(len(per_fault))
    for k, (cname, c) in enumerate(zip(cols, [C_RULE, C_LR, C_LSTM])):
        ax.bar(x + (k - 1) * 0.27, per_fault[cname], 0.27, color=c, label=cname)
    ax.set_xticks(x, per_fault.index, fontsize=8)
    ax.set(ylabel="Recall", ylim=(0, 1.05), title="Kokia dalis minučių prieš gedimą (≤100 km) atpažinta")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(FIGURES / "09_recall_per_fault.png"); plt.close(fig)

    # (e) Požymių svarbumas
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp.index, imp.values, color=C_LSTM)
    ax.set(xlabel="PR-AUC sumažėjimas sumaišius požymį", title="Kurie CAN požymiai LSTM svarbiausi (permutacinis svarbumas)")
    fig.tight_layout(); fig.savefig(FIGURES / "10_feature_importance.png"); plt.close(fig)

    # (f) Rizikos laiko juosta keliems testiniams automobiliams
    pv = per_vehicle["lstm"]
    picks = []
    for ft in ["healthy", "cooling_degradation", "fan_failure", "thermostat_stuck_open", "misfire"]:
        cand = pv[pv.fault_type == ft]
        if ft == "healthy":
            picks.append(int(cand.sort_values("false_alarm_episodes").vehicle_id.iloc[0]))
        else:
            picks.append(int(cand.sort_values("detected", ascending=False).vehicle_id.iloc[0]))
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.0 * len(picks)), sharex=False)
    for ax, vid in zip(axes, picks):
        g = test[test.vehicle_id == vid]
        ft = g.fault_type.iloc[0]
        x_km = g.odometer_km - (g.odometer_km.max() if ft != "healthy" else 0)
        ax.plot(x_km, g.p_lstm, c=C_LSTM, lw=1, label="LSTM rizika")
        ax.axhline(thr["lstm"], ls="--", c="grey", lw=0.8)
        if ft != "healthy":
            ax.axvspan(-HORIZON_KM, 0, color=C_FAULT, alpha=0.12, label="paskutiniai 100 km")
        alarm = alarms_from_scores(g.p_lstm.to_numpy() >= thr["lstm"])
        ax.scatter(x_km[alarm], np.full(alarm.sum(), 1.04), s=4, c=C_FAULT, marker="|", label="aliarmas")
        ax.set(ylim=(0, 1.1), ylabel="P(gedimas)")
        ax.set_title(f"Automobilis #{vid}: {FAULT_LT[ft]}", fontsize=8.5, loc="left")
    handles, labels_ = axes[1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="--", c="grey", lw=0.8)); labels_.append("aliarmo slenkstis")
    fig.legend(handles, labels_, loc="upper center", ncol=4, frameon=False, fontsize=8)
    axes[-1].set_xlabel("km (gedimų atveju: 0 = gedimo vieta)")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIGURES / "11_risk_timelines.png"); plt.close(fig)
    print("Grafikai išsaugoti:", FIGURES)


if __name__ == "__main__":
    main()
