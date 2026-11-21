# Transporto priemonės sveikatos prognozavimas iš CAN magistralės duomenų (LSTM)

LSTM modelis analizuoja CAN magistralės srautą: greitį, variklio apsukas, aušinimo skysčio temperatūrą ir apkrovą. Jis perspėja, kad **per artimiausius 100 km gali įvykti gedimas**.

Aptinkami 4 gedimų tipai: aušinimo sistemos degradacija, ventiliatoriaus gedimas, užstrigęs atviras termostatas ir uždegimo praleidimai.

## Rezultatai (testinė aibė: 28 automobiliai, kurių modelis nematė)

| Metodas | Precision | Recall | F1 | Aptikta gedimų | Klaidingi aliarmai / 1000 km |
|---|---|---|---|---|---|
| Skydelio lemputė (≥105 °C) | 0.82 | 0.01 | 0.02 | 3/16 | 0.0 |
| Diagnosto taisyklės | 0.46 | 0.20 | 0.28 | 12/16 | 4.1 |
| Logistinė regresija (Baseline) | 0.17 | 0.24 | 0.20 | 8/16 | 13.2 |
| **LSTM (Proposed)** | **0.84** | **0.61** | **0.70** | **15/16** | **1.3** |

Vidutiniškai LSTM perspėja likus **81 km** iki gedimo. Vienos prognozės skaičiavimas užtrunka apie 0,4 ms (CPU).

## Paleidimas

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                # duomenys -> modelis -> metrikos -> grafikai (~3 min.)
streamlit run app.py             # demonstracija naršyklėje (http://localhost:8501)
```

Su Docker:

```bash
docker build -t can-health .
docker run -p 8501:8501 can-health
```

## Struktūra

```
src/
  config.py      nustatymai (horizontas 100 km, sekos ilgis 30 min., požymiai)
  data_gen.py    CAN srauto simuliatorius (1 Hz) su besivystančiais gedimais
  preprocess.py  1 min. agregavimas, žymės, train/val/test skaidymas, LSTM sekos
  eda.py         duomenų analizės grafikai ir sistemos schema
  models.py      taisyklės, LSTM architektūra, aliarmo logika
  train.py       Baseline (taisyklės, logistinė regresija) ir LSTM treniravimas
  evaluate.py    metrikos, konfuzijos matricos, PR kreivės, XAI (permutacinis svarbumas)
  predict.py     apmokyto modelio naudojimas naujiems / realiems duomenims
app.py           Streamlit demonstracija
run_all.py       visas procesas vienu paleidimu
data/demo/       5 demonstraciniai CSV (testiniai automobiliai)
models/          apmokytas LSTM, slenksčiai, normalizavimo parametrai
results/         metrikos (CSV/JSON) ir grafikai
```

## Savo OBD log'o tikrinimas

Užtenka ELM327 adapterio ir programėlės, kuri eksportuoja CSV (pvz., Car Scanner ar Torque). Signalus reikia įrašinėti 1 Hz dažniu.

CSV turi turėti greitį, apsukas, aušinimo skysčio temperatūrą ir apkrovą. Dažni stulpelių pavadinimai atpažįstami automatiškai (žr. `src/predict.py`).

Kad modelis galėtų skaičiuoti, įraše reikia bent 30 min. važiavimo. Įkelk failą į `app.py`.

## Ribotumai

Modelis apmokytas su **sintetiniais** duomenimis. Prieš naudojant realiame automobilyje, jį reikėtų apmokyti iš naujo arba bent sukalibruoti su realiais duomenimis. Daugiau apie tai rašoma ataskaitos VIII skyriuje.
