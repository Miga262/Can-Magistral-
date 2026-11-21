"""Modelių apibrėžimai: bazinės taisyklės, logistinė regresija ir LSTM."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------- Baseline 1: prietaisų skydelio taisyklė ----------
def rule_dashboard(minutes: pd.DataFrame) -> np.ndarray:
    """Kaip temperatūros lemputė prietaisų skydelyje: aliarmas, kai aušinimo skystis >= 105 °C."""
    return (minutes["coolant_max"] >= 105).astype(float).to_numpy()


# ---------- Baseline 2: mechaniko (eksperto) taisyklės ----------
def rule_expert(minutes: pd.DataFrame) -> np.ndarray:
    """Rankomis parašytos taisyklės, kurias naudotų patyręs diagnostas."""
    hot = minutes["coolant_max"] >= 98
    rough_idle = (minutes["rpm_idle_std"] >= 60) & (minutes["idle_frac"] > 0.3)
    cold_engine = (minutes["trip_minutes"] >= 20) & (minutes["speed_mean"] > 60) & (minutes["coolant_mean"] < 80)
    return (hot | rough_idle | cold_engine).astype(float).to_numpy()


# ---------- Proposed: LSTM ----------
class LSTMFailurePredictor(nn.Module):
    """
    Įėjimas: seka (B, SEQ_LEN, F) - paskutinių 30 minučių CAN požymiai.
    Išėjimas: logit'as, kad per artimiausius 100 km įvyks gedimas.
    """
    def __init__(self, n_features, hidden=48, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@torch.no_grad()
def lstm_predict_proba(model, X, batch=2048):
    model.eval()
    probs = []
    for i in range(0, len(X), batch):
        probs.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))).numpy())
    return np.concatenate(probs) if probs else np.array([])


def alarms_from_scores(pred_binary: np.ndarray, consecutive=3) -> np.ndarray:
    """Aliarmas įjungiamas tik jei prognozė teigiama 'consecutive' minučių iš eilės (mažina klaidingus)."""
    s = pd.Series(pred_binary.astype(int))
    return (s.rolling(consecutive, min_periods=consecutive).sum() >= consecutive).to_numpy()
