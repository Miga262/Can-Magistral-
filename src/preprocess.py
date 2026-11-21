"""
Duomenų paruošimas:
 1) 1 Hz CAN srautas agreguojamas į 1 minutės požymius (vidurkiai, sklaida, maksimumai);
 2) kiekvienai minutei priskiriama žymė: 1, jei iki gedimo liko <= 100 km;
 3) automobiliai padalijami į train / val / test (pagal automobilį, kad nebūtų duomenų nutekėjimo);
 4) sudaromos LSTM sekos (paskutinės SEQ_LEN minučių).
"""
import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_PROC, FEATURES, HORIZON_KM, SEQ_LEN, SEED


def aggregate_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """1 Hz CAN duomenys -> 1 minutės požymiai. Veikia ir su realiu OBD log'u (tie patys stulpeliai)."""
    df = raw.copy()
    if "trip_id" not in df:
        # jei kelionių nėra - nauja kelionė prasideda po >5 min pertraukos
        df["trip_id"] = (df["time_s"].diff().fillna(0) > 300).cumsum()
    trip_start = df.groupby("trip_id")["time_s"].transform("min")
    df["minute"] = ((df["time_s"] - trip_start) // 60).astype(int)
    df["idle"] = df["speed_kmh"] < 3
    df["rpm_idle"] = df["rpm"].where(df["idle"])
    g = df.groupby(["trip_id", "minute"], sort=True)
    out = pd.DataFrame({
        "n": g.size(),
        "speed_mean": g["speed_kmh"].mean(),
        "speed_std": g["speed_kmh"].std(),
        "rpm_mean": g["rpm"].mean(),
        "rpm_std": g["rpm"].std(),
        "rpm_idle_std": g["rpm_idle"].std(),
        "coolant_mean": g["coolant_c"].mean(),
        "coolant_max": g["coolant_c"].max(),
        "coolant_delta": g["coolant_c"].last() - g["coolant_c"].first(),
        "load_mean": g["engine_load_pct"].mean(),
        "idle_frac": g["idle"].mean(),
        "odometer_km": g["odometer_km"].last(),
    }).reset_index()
    out = out[out["n"] >= 30].copy()                 # atmetamos nepilnos minutės (kelionės pabaiga)
    out["trip_minutes"] = out["minute"].clip(upper=60)
    out[FEATURES] = out[FEATURES].fillna(0.0)
    return out.drop(columns=["n"]).reset_index(drop=True)


def split_vehicles(meta: pd.DataFrame) -> dict:
    """Stratifikuotas skaidymas pagal automobilį: 60 / 20 / 20 %."""
    rng = np.random.default_rng(SEED)
    split = {}
    for _, grp in meta.groupby("fault_type"):
        ids = grp["vehicle_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_val, n_test = int(round(n * 0.2)), int(round(n * 0.2))
        for i, vid in enumerate(ids):
            split[int(vid)] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return split


def make_sequences(minutes: pd.DataFrame, scaler_mean, scaler_std, seq_len=SEQ_LEN):
    """Kiekvienai minutei t: seka [t-seq_len+1 .. t] (to paties automobilio). Grąžina X, y, indeksus."""
    Xs, ys, idx = [], [], []
    for _, grp in minutes.groupby("vehicle_id", sort=False):
        F = ((grp[FEATURES].to_numpy() - scaler_mean) / scaler_std).astype(np.float32)
        y = grp["label"].to_numpy() if "label" in grp else np.zeros(len(grp))
        if len(grp) < seq_len:
            continue
        win = np.lib.stride_tricks.sliding_window_view(F, (seq_len, F.shape[1]))[:, 0]
        Xs.append(win)
        ys.append(y[seq_len - 1:])
        idx.append(grp.index.to_numpy()[seq_len - 1:])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(idx)


def main():
    meta = pd.read_csv(DATA_PROC / "vehicles.csv")
    split = split_vehicles(meta)
    parts = []
    for _, row in meta.iterrows():
        vid = int(row["vehicle_id"])
        raw = pd.read_csv(DATA_RAW / f"vehicle_{vid:03d}.csv.gz")
        m = aggregate_minutes(raw)
        m.insert(0, "vehicle_id", vid)
        m["fault_type"] = row["fault_type"]
        if row["fault_type"] == "healthy":
            m["km_to_failure"] = np.nan
            m["label"] = 0
        else:
            m["km_to_failure"] = row["failure_km"] - m["odometer_km"]
            m["label"] = (m["km_to_failure"] <= HORIZON_KM).astype(int)
        m["split"] = split[vid]
        parts.append(m)
    minutes = pd.concat(parts, ignore_index=True)
    minutes.to_csv(DATA_PROC / "minutes.csv.gz", index=False)
    print("Minučių įrašų:", len(minutes))
    print(minutes.groupby("split").agg(vehicles=("vehicle_id", "nunique"), minutes=("label", "size"),
                                       positives=("label", "sum")))


if __name__ == "__main__":
    main()
