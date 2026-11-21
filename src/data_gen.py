"""
CAN magistralės duomenų simuliatorius (1 Hz).

Kiekvienas automobilis nuvažiuoja 350-650 km kelionėmis (miestas / užmiestis / magistralė).
Signalai: greitis, variklio apsukos, aušinimo skysčio temperatūra, variklio apkrova.

Dalis automobilių turi besivystantį gedimą, kuris prasideda 120-220 km iki galutinio gedimo
ir stiprėja (severity 0 -> 1). Gedimų tipai:
  cooling_degradation   - užsikimšęs radiatorius / sumažėjęs aušinimo efektyvumas
  fan_failure           - gendantis radiatoriaus ventiliatorius (matosi tik stovint / mieste)
  thermostat_stuck_open - užstrigęs atviras termostatas (variklis per šaltas magistralėje)
  misfire               - uždegimo / purkštuko problemos (nestabilios apsukos tuščiąja eiga)

Fizikos modeliai supaprastinti, bet atkartoja realius ryšius:
apkrova ~ pagreitis + greitis^2, apsukos ~ greitis * pavaros santykis,
temperatūra = šilumos gamyba (apkrova) - aušinimas (termostatas, oro srautas, ventiliatorius).
"""
import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_DEMO, DATA_PROC, SEED, N_HEALTHY, N_PER_FAULT, FAULT_TYPES

# Apsukos vienam km/h kiekvienoje pavaroje (dyzelinio variklio tipo santykiai)
GEAR_RPM_PER_KMH = np.array([110, 62, 42, 32, 25, 20])
GEAR_UP_KMH = np.array([0, 15, 30, 48, 68, 88])


def gen_trip_speed(rng, trip_type, dist_km):
    """Sugeneruoja vienos kelionės greičio profilį (km/h, 1 Hz)."""
    v, km, out = 0.0, 0.0, []
    while km < dist_km:
        # pasirenkamas naujas tikslinis greitis ir jo trukmė
        if trip_type == "city" or (trip_type != "highway" and rng.random() < 0.5) or len(out) < 120:
            if rng.random() < 0.35:
                target, hold = 0.0, rng.integers(10, 90)          # sustojimas (šviesoforas)
                if rng.random() < 0.15:
                    hold = rng.integers(120, 400)                 # ilgesnis stovėjimas (kamštis, laukimas)
            else:
                target, hold = rng.uniform(20, 60), rng.integers(20, 150)
        elif trip_type == "highway":
            if rng.random() < 0.06:
                target, hold = rng.uniform(40, 70), rng.integers(30, 120)  # lėtėjimas eismo
            else:
                target, hold = rng.uniform(85, 130), rng.integers(60, 400)
        else:  # mixed - užmiesčio kelias
            target, hold = rng.uniform(60, 90), rng.integers(40, 250)
        a_up = rng.uniform(1.0, 2.8) * 3.6   # km/h per sekundę
        a_dn = 2.5 * 3.6
        for _ in range(int(hold)):
            dv = np.clip((target - v) * 0.3, -a_dn, a_up)
            v = max(0.0, v + dv + (rng.normal(0, 0.6) if v > 5 else 0.0))
            out.append(v)
            km += v / 3600.0
            if km >= dist_km:
                break
    # kelionės pabaigoje sustojame
    for _ in range(20):
        v = max(0.0, v - 9.0)
        out.append(v)
    return np.array(out)


def severity(km, onset_km, failure_km):
    """Gedimo stiprumas 0..1 (0 - dar neprasidėjo, 1 - gedimas)."""
    if failure_km is None:
        return np.zeros_like(km)
    s = np.clip((km - onset_km) / (failure_km - onset_km), 0, 1)
    return s ** 0.7


def simulate_vehicle(vid, fault, rng):
    """Simuliuoja vieną automobilį. Grąžina 1 Hz DataFrame ir metaduomenis."""
    total_km = rng.uniform(350, 650)
    ambient = rng.uniform(-5, 30)
    setpoint = rng.uniform(86, 92)          # termostato atsidarymo temperatūra
    idle_rpm = rng.uniform(700, 820)
    idle_noise = rng.uniform(5, 25)          # kiekvienas variklis dirba šiek tiek kitaip
    load_bias = rng.choice([0, 0, 0, 0, rng.uniform(8, 18)])  # ~20% mašinų - sunki apkrova (priekaba)
    k_rad = rng.uniform(0.12, 0.15)
    failure_km = total_km if fault != "healthy" else None
    onset_km = total_km - rng.uniform(120, 220) if fault != "healthy" else None

    frames, odo, t0 = [], 0.0, 0
    temp = ambient
    fan_on = False
    trip_id = 0
    while odo < total_km:
        trip_type = rng.choice(["city", "mixed", "highway"], p=[0.45, 0.30, 0.25])
        dist = {"city": rng.uniform(4, 18), "mixed": rng.uniform(12, 40), "highway": rng.uniform(30, 90)}[trip_type]
        dist = min(dist, total_km - odo + 0.05)
        v = gen_trip_speed(rng, trip_type, dist)
        n = len(v)
        km = odo + np.cumsum(v) / 3600.0
        s = severity(km, onset_km, failure_km)
        acc = np.diff(v, prepend=0.0) / 3.6                        # m/s^2

        # --- variklio apkrova (%) ---
        load = 16 + 0.0035 * v ** 2 + 14 * np.clip(acc, 0, None) + load_bias * (v > 50)
        load = np.where(acc < -0.8, rng.uniform(0, 6, n), load)    # stabdant - kuro padavimas nutrūksta
        load = np.where(v < 3, 15 + rng.normal(0, 1.5, n), load)
        if fault == "misfire":
            load = load + 6 * s                                     # ECU kompensuoja prarastą galią
        load = np.clip(load + rng.normal(0, 2, n), 0, 100)

        # --- apsukos ---
        gear = np.searchsorted(GEAR_UP_KMH, v, side="right") - 1
        rpm = v * GEAR_RPM_PER_KMH[gear] * (1 + 0.12 * np.clip(acc, 0, 2))
        rpm = np.maximum(rpm, idle_rpm)
        idle = v < 3
        noise = np.where(idle, idle_noise, 15.0)
        if fault == "misfire":
            noise = noise + np.where(idle, 130 * s, 45 * s)
            dips = rng.random(n) < 0.12 * s                          # pavieniai "kritimai" uždegimo praleidimo metu
            rpm = rpm - dips * rng.uniform(60, 220, n)
        rpm = rpm + rng.normal(0, 1, n) * noise

        # --- aušinimo skysčio temperatūra (šiluminis modelis) ---
        tc = np.empty(n)
        extra = 0.0   # papildomas įkaitimas dėl gedimo (°C)
        for i in range(n):
            heat = 0.16 * (0.35 + load[i] / 100)
            sp = setpoint
            opening = np.clip((temp - sp) / 6, 0, 1)
            if fault == "thermostat_stuck_open":
                opening = max(opening, 0.85 * s[i])               # termostatas nebeužsidaro
            if temp > sp + 8:
                fan_on = True
            elif temp < sp + 4:
                fan_on = False
            airflow = 0.12 + v[i] / 100 + (1.0 if fan_on else 0)
            cooling = opening * k_rad * airflow * (temp - ambient) / 60 + 0.0003 * (temp - ambient)
            temp = temp + heat - cooling
            # gedimų simptomai, kurių termostatas nebegali kompensuoti:
            if fault == "cooling_degradation":
                # radiatorius aušina prasčiau -> perkaitimas proporcingas apkrovai
                target = 16 * s[i] * (0.15 + load[i] / 100)
                extra += (target - extra) / 90
            elif fault == "fan_failure":
                # ventiliatorius nebeįsijungia -> stovint / lėtai važiuojant temperatūra kyla
                if v[i] < 10 and temp > sp + 2:
                    extra += 0.05 * s[i]
                else:
                    extra *= 0.985
            tc[i] = temp + extra
        tc = tc + rng.normal(0, 0.25, n)

        frames.append(pd.DataFrame({
            "time_s": t0 + np.arange(n),
            "trip_id": trip_id,
            "speed_kmh": v.round(1),
            "rpm": rpm.round(0),
            "coolant_c": tc.round(1),
            "engine_load_pct": load.round(1),
            "odometer_km": km.round(3),
        }))
        odo = km[-1]
        trip_id += 1
        # stovėjimas tarp kelionių: variklis vėsta
        park_s = rng.uniform(600, 12 * 3600)
        temp = ambient + (temp - ambient) * np.exp(-park_s / 3600)
        fan_on = False
        t0 += n + int(park_s)

    df = pd.concat(frames, ignore_index=True)
    meta = dict(vehicle_id=vid, fault_type=fault, total_km=round(odo, 2),
                failure_km=round(odo, 2) if failure_km else None,
                onset_km=round(onset_km, 2) if onset_km else None,
                ambient_c=round(ambient, 1), thermostat_c=round(setpoint, 1),
                heavy_load=bool(load_bias > 0))
    return df, meta


def main():
    rng = np.random.default_rng(SEED)
    plan = ["healthy"] * N_HEALTHY + [f for f in FAULT_TYPES for _ in range(N_PER_FAULT)]
    rng.shuffle(plan)
    metas, total_rows = [], 0
    for vid, fault in enumerate(plan):
        df, meta = simulate_vehicle(vid, fault, rng)
        df.to_csv(DATA_RAW / f"vehicle_{vid:03d}.csv.gz", index=False)
        meta["rows_1hz"] = len(df)
        metas.append(meta)
        total_rows += len(df)
        print(f"vehicle {vid:03d} {fault:22s} {meta['total_km']:6.1f} km  {len(df):7d} eil.")
    meta_df = pd.DataFrame(metas)
    meta_df.to_csv(DATA_PROC / "vehicles.csv", index=False)
    print(f"Iš viso: {len(metas)} automobilių, {total_rows} CAN eilučių (1 Hz), "
          f"{meta_df.total_km.sum():.0f} km")


if __name__ == "__main__":
    main()
