"""Bendri projekto nustatymai (keliai, parametrai)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_DEMO = ROOT / "data" / "demo"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SEED = 42

# --- Simuliacija ---
N_HEALTHY = 60          # sveikų automobilių skaičius
N_PER_FAULT = 20        # automobilių skaičius kiekvienam gedimo tipui
FAULT_TYPES = ["cooling_degradation", "fan_failure", "thermostat_stuck_open", "misfire"]

# --- Uždavinys ---
HORIZON_KM = 100.0      # perspėti likus <= 100 km iki gedimo
AGG_SECONDS = 60        # CAN srautas (1 Hz) agreguojamas į 1 min. žingsnius
SEQ_LEN = 30            # LSTM mato paskutines 30 min. (~20-40 km)

# CAN signalai (1 Hz), kuriuos naudoja sistema
RAW_COLS = ["speed_kmh", "rpm", "coolant_c", "engine_load_pct"]

# Kiekvienos minutės požymiai (žr. preprocess.py)
FEATURES = [
    "speed_mean", "speed_std", "rpm_mean", "rpm_std", "rpm_idle_std",
    "coolant_mean", "coolant_max", "coolant_delta", "load_mean",
    "idle_frac", "trip_minutes",
]

for p in (DATA_RAW, DATA_DEMO, DATA_PROC, MODELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
