"""
Streamlit demonstracija: įkelk CAN / OBD-II log'ą (CSV) ir pamatyk gedimo riziką.
Paleidimas:  streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import DATA_DEMO, HORIZON_KM  # noqa: E402
from predict import FailurePredictor  # noqa: E402

st.set_page_config(page_title="Automobilio sveikatos prognozė", page_icon="🚗", layout="wide")
st.title("🚗 Automobilio sveikatos prognozavimas iš CAN duomenų")
st.caption("LSTM modelis analizuoja paskutines 30 min. važiavimo ir įvertina tikimybę, "
           f"kad per artimiausius {HORIZON_KM:.0f} km įvyks gedimas. Sistema tik informuoja – nieko nevaldo.")


@st.cache_resource
def load_predictor():
    return FailurePredictor()


predictor = load_predictor()

with st.sidebar:
    st.header("Duomenys")
    demos = sorted(DATA_DEMO.glob("*.csv"))
    choice = st.selectbox("Demonstracinis įrašas", ["—"] + [d.name for d in demos])
    up = st.file_uploader("…arba įkelk savo CSV (1 Hz)", type=["csv"])
    st.markdown("Reikalingi stulpeliai: `speed_kmh`, `rpm`, `coolant_c`, `engine_load_pct` "
                "(nebūtini: `time_s`, `odometer_km`, `trip_id`). Dažni OBD programėlių pavadinimai atpažįstami automatiškai.")
    st.markdown(f"Aliarmo slenkstis: **{predictor.threshold:.2f}**, 3 min. iš eilės")

raw = None
if up is not None:
    raw = pd.read_csv(up)
elif choice != "—":
    raw = pd.read_csv(DATA_DEMO / choice)

if raw is None:
    st.info("Pasirink demonstracinį įrašą kairėje arba įkelk savo CSV.")
    st.stop()

try:
    res = predictor.predict(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

valid = res.dropna(subset=["risk"])
if valid.empty:
    st.warning("Per mažai duomenų: reikia bent 30 min. važiavimo.")
    st.stop()

last = valid.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nuvažiuota (įraše)", f"{res.odometer_km.max() - res.odometer_km.min():.0f} km")
c2.metric("Dabartinė rizika", f"{last.risk:.0%}")
c3.metric("Aliarmo minučių", int(res.alarm.sum()))
if res.alarm.any():
    first = res[res.alarm].iloc[0]
    c4.metric("Pirmas aliarmas", f"{first.odometer_km:.0f} km")
    st.error(f"⚠️ Aptikti gedimo požymiai (nuo {first.odometer_km:.0f} km). Rekomenduojama patikrinti automobilį servise.")
else:
    c4.metric("Pirmas aliarmas", "—")
    st.success("✅ Gedimo požymių neaptikta.")

fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
km = res.odometer_km
ax[0].plot(km, res.risk, c="#2E8B57", lw=1.2)
ax[0].axhline(predictor.threshold, ls="--", c="grey", lw=0.8)
ax[0].fill_between(km, 0, 1, where=res.alarm, color="#E45756", alpha=0.2, step="mid")
ax[0].set(ylabel="Gedimo tikimybė", ylim=(0, 1.02))
ax[1].plot(km, res.coolant_mean, c="#E45756", lw=0.8, label="vidurkis")
ax[1].plot(km, res.coolant_max, c="#E45756", lw=0.5, alpha=0.4, label="maksimumas")
ax[1].set_ylabel("Aušinimo skystis, °C"); ax[1].legend(frameon=False, fontsize=8)
ax[2].plot(km, res.rpm_idle_std, c="#F58518", lw=0.8)
ax[2].set_ylabel("Apsukų sklaida\ntuščiąja eiga")
ax[3].plot(km, res.speed_mean, c="#4C78A8", lw=0.8)
ax[3].set(ylabel="Greitis, km/h", xlabel="Odometras, km")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)

with st.expander("Minučių požymiai ir prognozės"):
    st.dataframe(res.round(3))
    st.download_button("Atsisiųsti rezultatus (CSV)", res.to_csv(index=False), "prognozes.csv")
