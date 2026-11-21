"""
Apmokyto LSTM naudojimas naujiems duomenims (pvz., realiam OBD log'ui).
Naudoja app.py (Streamlit demo).
"""
import json

import numpy as np
import pandas as pd
import torch

from config import FEATURES, MODELS, SEQ_LEN
from models import LSTMFailurePredictor, alarms_from_scores, lstm_predict_proba
from preprocess import aggregate_minutes

# Dažni stulpelių pavadinimai OBD programėlėse (Car Scanner, Torque ir pan.)
COLUMN_HINTS = {
    "speed_kmh": ["speed_kmh", "speed", "greitis", "vehicle speed"],
    "rpm": ["rpm", "engine rpm", "apsukos"],
    "coolant_c": ["coolant_c", "coolant", "engine coolant temperature", "aušinimo"],
    "engine_load_pct": ["engine_load_pct", "engine load", "calculated engine load", "load", "apkrova"],
    "time_s": ["time_s", "time", "seconds", "laikas"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bando atpažinti stulpelius pagal pavadinimą ir pervadina į sistemos formatą."""
    out = df.copy()
    lower = {c: c.lower().strip() for c in df.columns}
    for target, hints in COLUMN_HINTS.items():
        if target in out.columns:
            continue
        for col, lc in lower.items():
            if any(lc.startswith(h) or h in lc for h in hints):
                out = out.rename(columns={col: target})
                break
    missing = [c for c in ["speed_kmh", "rpm", "coolant_c", "engine_load_pct"] if c not in out.columns]
    if missing:
        raise ValueError(f"Trūksta stulpelių: {missing}. Reikalingi: speed_kmh, rpm, coolant_c, engine_load_pct")
    if "time_s" not in out.columns:
        out["time_s"] = np.arange(len(out))          # laikome, kad įrašas kas 1 s
    if not np.issubdtype(out["time_s"].dtype, np.number):
        t = pd.to_datetime(out["time_s"])
        out["time_s"] = (t - t.iloc[0]).dt.total_seconds()
    if "odometer_km" not in out.columns:
        dt = out["time_s"].diff().fillna(1).clip(0, 5)
        out["odometer_km"] = (out["speed_kmh"] * dt / 3600).cumsum()
    return out


class FailurePredictor:
    def __init__(self, models_dir=MODELS):
        sc = json.load(open(models_dir / "scaler.json"))
        self.mean, self.std = np.array(sc["mean"]), np.array(sc["std"])
        self.threshold = json.load(open(models_dir / "thresholds.json"))["thresholds"]["lstm"]
        self.model = LSTMFailurePredictor(len(FEATURES))
        self.model.load_state_dict(torch.load(models_dir / "lstm.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """1 Hz CAN duomenys -> kiekvienos minutės gedimo tikimybė ir aliarmas."""
        raw = normalize_columns(raw)
        minutes = aggregate_minutes(raw)
        F = ((minutes[FEATURES].to_numpy() - self.mean) / self.std).astype(np.float32)
        p = np.full(len(minutes), np.nan)
        if len(minutes) >= SEQ_LEN:
            win = np.lib.stride_tricks.sliding_window_view(F, (SEQ_LEN, F.shape[1]))[:, 0]
            p[SEQ_LEN - 1:] = lstm_predict_proba(self.model, np.ascontiguousarray(win))
        minutes["risk"] = p
        minutes["alarm"] = alarms_from_scores(np.nan_to_num(p) >= self.threshold)
        return minutes
