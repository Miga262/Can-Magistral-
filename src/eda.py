"""Duomenų analizė (EDA) ir sistemos schemos grafikai ataskaitai."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from config import DATA_PROC, DATA_RAW, FEATURES, FIGURES

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FAULT_LT = {"healthy": "Sveikas", "cooling_degradation": "Aušinimo degradacija", "fan_failure": "Ventiliatorius",
            "thermostat_stuck_open": "Termostatas", "misfire": "Uždegimo praleidimai"}


def main():
    meta = pd.read_csv(DATA_PROC / "vehicles.csv")
    minutes = pd.read_csv(DATA_PROC / "minutes.csv.gz")

    # 1) Neapdoroto CAN srauto pavyzdys (viena kelionė)
    raw = pd.read_csv(DATA_RAW / f"vehicle_{int(meta[meta.fault_type == 'healthy'].vehicle_id.iloc[0]):03d}.csv.gz")
    trip = raw[raw.trip_id == raw.groupby("trip_id").size().idxmax()]
    t = (trip.time_s - trip.time_s.min()) / 60
    fig, ax = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
    for a, col, lab, c in zip(ax, ["speed_kmh", "rpm", "coolant_c", "engine_load_pct"],
                              ["Greitis, km/h", "Apsukos, rpm", "Aušinimo skystis, °C", "Apkrova, %"],
                              ["#4C78A8", "#F58518", "#E45756", "#54A24B"]):
        a.plot(t, trip[col], lw=0.6, c=c); a.set_ylabel(lab, fontsize=8)
    ax[-1].set_xlabel("Laikas nuo kelionės pradžios, min")
    ax[0].set_title("CAN magistralės srautas (1 Hz), viena sveiko automobilio kelionė", loc="left")
    fig.tight_layout(); fig.savefig(FIGURES / "01_can_stream_example.png"); plt.close(fig)

    # 2) Gedimų "parašai": kaip keičiasi požymiai artėjant gedimui
    warm = minutes[minutes.trip_minutes >= 10].merge(meta[["vehicle_id", "thermostat_c"]], on="vehicle_id")
    warm["coolant_dev"] = warm.coolant_mean - warm.thermostat_c
    bins = [-1, 25, 50, 100, 150, 250, 1000]
    labels = ["0–25", "25–50", "50–100", "100–150", "150–250", ">250"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    specs = [("coolant_dev", "Temp. virš termostato, °C", ["cooling_degradation", "fan_failure", "thermostat_stuck_open"]),
             ("coolant_max", "Maks. temp. per minutę, °C", ["cooling_degradation", "fan_failure"]),
             ("rpm_idle_std", "Apsukų sklaida tuščiąja eiga, rpm", ["misfire"])]
    healthy = warm[warm.fault_type == "healthy"]
    for ax, (col, lab, faults) in zip(axes, specs):
        for f in faults:
            g = warm[warm.fault_type == f].copy()
            g["bin"] = pd.cut(g.km_to_failure, bins, labels=labels)
            stat = g.groupby("bin", observed=False)[col].quantile(0.9 if col != "coolant_dev" else 0.5)
            ax.plot(labels[::-1], stat.values[::-1], "-o", ms=3, label=FAULT_LT[f])
        hval = healthy[col].quantile(0.9 if col != "coolant_dev" else 0.5)
        ax.axhline(hval, ls="--", c="grey", lw=0.8, label="Sveiki automobiliai")
        ax.set(xlabel="km iki gedimo", ylabel=lab)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Gedimų simptomai stiprėja artėjant gedimui (mediana / 90-as procentilis)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIGURES / "02_fault_signatures.png"); plt.close(fig)

    # 3) Požymių histogramos: sveika vs paskutiniai 100 km
    show = ["coolant_mean", "coolant_max", "coolant_delta", "rpm_idle_std", "load_mean", "speed_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for ax, f in zip(axes.ravel(), show):
        d = minutes[minutes.idle_frac > 0.3] if f == "rpm_idle_std" else minutes   # apsukų sklaida - tik stovint
        lo, hi = d[f].quantile([0.005, 0.995])
        b = np.linspace(lo, hi, 40)
        ax.hist(d[d.label == 0][f], bins=b, density=True, alpha=0.55, color="#4C78A8", label="Normali būsena")
        ax.hist(d[d.label == 1][f], bins=b, density=True, alpha=0.55, color="#E45756", label="≤100 km iki gedimo")
        ax.set_title(f + (" (minutės su >30 % tuščiosios eigos)" if f == "rpm_idle_std" else ""), fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Minučių požymių pasiskirstymas", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGURES / "03_feature_histograms.png"); plt.close(fig)

    # 4) Koreliacijų matrica
    corr = minutes[FEATURES + ["label"]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Požymių koreliacijos (Pearson)")
    fig.tight_layout(); fig.savefig(FIGURES / "04_correlation.png"); plt.close(fig)

    # 5) Sistemos blokinė schema
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
    boxes = [("CAN magistralė /\nOBD-II adapteris\n(1 Hz)", "#DCE9F5"),
             ("Agregavimas\n1 min. požymiai\n(11 požymių)", "#E8F3E1"),
             ("Normalizavimas\n+ 30 min. seka", "#E8F3E1"),
             ("LSTM\n2 sl. × 48\n→ P(gedimas ≤100 km)", "#FCE8D5"),
             ("Aliarmo logika\n3 min. iš eilės\n> slenkstis", "#FCE8D5"),
             ("Vairuotojo /\nmechaniko\nperspėjimas", "#F5DCDC")]
    w, gap = 1.45, 0.22
    for i, (txt, c) in enumerate(boxes):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 1.4, boxstyle="round,pad=0.03", fc=c, ec="#555"))
        ax.text(x + w / 2, 1.2, txt, ha="center", va="center", fontsize=7.8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 1.2), xytext=(x + w + 0.02, 1.2),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(5, 0.15, "Sistema tik informuoja – automobilio valdymui neturi jokios įtakos (informacinis kritiškumas)",
            ha="center", fontsize=7.5, style="italic", color="#444")
    fig.tight_layout(); fig.savefig(FIGURES / "05_architecture.png"); plt.close(fig)

    # Duomenų suvestinė ataskaitai
    summary = meta.groupby("fault_type").agg(automobiliai=("vehicle_id", "size"), km=("total_km", "sum"),
                                             eilutes_1hz=("rows_1hz", "sum")).round(0)
    summary.to_csv(FIGURES.parent / "data_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
