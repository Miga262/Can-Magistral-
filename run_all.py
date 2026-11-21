"""Visas procesas vienu paleidimu: duomenys -> paruošimas -> EDA -> treniravimas -> vertinimas -> demo failai."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent / "src"
for step in ["data_gen.py", "preprocess.py", "eda.py", "train.py", "evaluate.py"]:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)

# Demonstraciniai CSV failai Streamlit programėlei (testiniai automobiliai, kurių modelis nematė)
sys.path.insert(0, str(SRC))
from config import DATA_DEMO, DATA_PROC, DATA_RAW  # noqa: E402

meta = pd.read_csv(DATA_PROC / "vehicles.csv")
minutes = pd.read_csv(DATA_PROC / "minutes.csv.gz", usecols=["vehicle_id", "split"]).drop_duplicates()
test_ids = set(minutes[minutes.split == "test"].vehicle_id)
names = {"healthy": "sveikas", "cooling_degradation": "ausinimo_degradacija", "fan_failure": "ventiliatorius",
         "thermostat_stuck_open": "termostatas", "misfire": "uzdegimo_praleidimai"}
for f in DATA_DEMO.glob("*.csv"):
    f.unlink()
for ft, lt in names.items():
    vid = int(meta[(meta.fault_type == ft) & meta.vehicle_id.isin(test_ids)].vehicle_id.iloc[0])
    pd.read_csv(DATA_RAW / f"vehicle_{vid:03d}.csv.gz").to_csv(DATA_DEMO / f"{lt}_auto{vid}.csv", index=False)
print("\nDemo failai:", sorted(p.name for p in DATA_DEMO.glob("*.csv")))
