# IoT-CARES — An IoT-Based Healthcare Monitoring and Surveillance Framework for People with Special Needs

Code, data and results that accompany the manuscript
**"An IoT-Based Healthcare Monitoring and Surveillance Framework for People with Special Needs"**.

IoT-CARES (Calibrated, Adaptive, Resource-aware, Explainable Surveillance) is evaluated on five monitoring services:

| Service | Target group | Open dataset |
|---|---|---|
| S1 fall detection + calibrated alarms | older / frail people living alone | UCI Simulated Falls and Daily Living Activities |
| S2 activity & mobility profile | adults aged 70–95 | UCI HAR70+ |
| S3 freezing-of-gait detection | people with Parkinson's disease | UCI Daphnet Freezing of Gait |
| S4 arrhythmia / ectopy burden | people with cardiac conditions | PhysioNet MIT-BIH Arrhythmia Database |
| S5 cognitive-health surveillance | people at risk of / living with dementia | CASAS cognitive assessment (Zenodo 15713579) |

Everything is plain Python (NumPy / SciPy / pandas / scikit-learn / Matplotlib). **No GPU and no deep-learning
framework is needed** — the neural network is written in NumPy with explicit back-propagation (checked by
`tests/test_gradients.py`), so the gateway part runs on any small computer.

---------------------------------------------------------------------------------------------------------
## 1. Where to put the dataset path and the results path (the ONLY place to edit)

Open **`config.py`** and edit the three lines at the top:

```python
DATA_PATH      = ".../IoTCARES/data/raw"        # <- folder with the raw downloaded datasets
PROCESSED_PATH = ".../IoTCARES/data/processed"  # <- windows / features are written here
RESULTS_PATH   = ".../IoTCARES/results"         # <- tables (CSV), figures (PNG), logs, predictions
```

(or set the environment variables `CARES_DATA`, `CARES_PROCESSED`, `CARES_RESULTS`).
By default all three point inside this folder, so the package works out-of-the-box after unzipping.

`DATA_PATH` must contain:

| file | dataset | size |
|---|---|---|
| `daphnet_fog.zip` | UCI Daphnet Freezing of Gait | 20.5 MB |
| `har70plus.zip` | UCI HAR70+ | 42.2 MB |
| `falls_selected/` (17 files) **or** `falls_selected.npz` | accelerometer + gyroscope of the UCI falls dataset (extracted by script 04 from the 1.2 GB archive) | 174 MB |
| `mitbih.zip` | PhysioNet MIT-BIH Arrhythmia Database v1.0.0 | 77 MB |
| `casas_cognitive_assessment.zip` | CASAS cognitive assessment (Zenodo 15713579) | 12.6 MB |

All of them are included in `data/raw/` of this package. `01_download_data.py` downloads them again from the
official sources if needed (the 1.2 GB falls archive only if `falls_selected` is missing).

---------------------------------------------------------------------------------------------------------
## 2. Installation

```bash
python -m pip install -r requirements.txt
python tests/test_gradients.py        # optional: checks the back-propagation code
# figures 1-4 (diagrams) also need Graphviz (the `dot` command): https://graphviz.org/download/
# the Word manuscript is rebuilt with Node.js + `npm install docx` (optional)
```

---------------------------------------------------------------------------------------------------------
## 3. Run order (files are numbered — run them in this order)

| # | script | what it does | approx. time (2-core CPU) |
|---|---|---|---|
| 01 | `01_download_data.py` | downloads the 5 datasets into `DATA_PATH` (skip if `data/raw` is filled) | 2–30 min |
| 02 | `02_prepare_fog_parkinson.py` | Daphnet: 4-s windows, freezing episodes | 10 s |
| 03 | `03_prepare_har70_older_adults.py` | HAR70+: 5-s windows, 6 activity classes | 10 s |
| 04 | `04_prepare_falls.py` | falls: trial records, peak-centred windows (needs the 1.2 GB archive only if `falls_selected` is missing) | 30 s |
| 05 | `05_prepare_ecg_mitbih.py` | MIT-BIH: beats, AAMI classes, RR + morphology, DS1/DS2 | 10 s |
| 06 | `06_prepare_cognitive_smarthome.py` | CASAS: per-task ambient-sensor features, diagnoses | 10 s |
| 07 | `07_extract_edge_features.py` | edge descriptors for the wearable services | 1 min |
| 08 | `08_benchmark_models.py falls / fog / har70` | leave-one-person-out benchmark of 7 detectors | 10 / 60 / 90 min |
| 08b | `08b_benchmark_ecg.py` | ECG inter-patient, patient-adaptive and federated | 30 min |
| 09 | `09_federated_personalisation.py fog / har70 / falls` | FedAvg/FedProx, partial participation, onboarding personalisation | 20–40 min each |
| 10 | `10_calibrated_alarms.py fog / falls` | conformal false-alarm budget, streaming fall alarms | 1 / 40 min |
| 11 | `11_cognitive_surveillance.py` | dementia / MCI screening from ambient sensors | 5 min |
| 12 | `12_explainability_robustness.py falls / fog / har70` | sensor placement, dropout, sampling rate, noise, alert explanations | 10–60 min each |
| 13 | `13_edge_iot_simulation.py` | gateway latency, data leaving the home, alert-delivery simulation | 3 min |
| 14 | `14_make_figures.py` | result figures (PNG, 300 dpi) | 1 min |
| 15 | `15_make_diagrams.py` | architecture / flow / study-design diagrams (Graphviz) | 10 s |
| 16 | `16_collect_results.py` | gathers every number of the paper into `results/paper_numbers.json` | 5 s |
| 17 | `17_verify_manuscript_numbers.py` | (optional) checks that the Word manuscript quotes the generated numbers | 10 s |

One-click alternatives: `run_all.sh` (Linux/macOS) or `run_all.bat` (Windows).

---------------------------------------------------------------------------------------------------------
## 4. Outputs (`RESULTS_PATH`)

* `tables/` — every table of the manuscript and supplement as CSV (T1* cohorts, T3 benchmark, T4 ECG, T5 federated,
  T6 alarm budget, T7 cognition, T8 robustness, T9 explanations, T10–T12 edge/IoT, S* supplementary)
* `figures/` — Fig1–Fig16 (PNG 300 dpi) + Graphviz sources (`.dot`)
* `predictions/` — person-level predictions of every model (`bench_*.npz`, `stream_falls.npz`, ...)
* `logs/` — run logs
* `paper_numbers.json` — machine-readable copy of all numbers reported in the manuscript

---------------------------------------------------------------------------------------------------------
## 5. Data statement

All five datasets are public and de-identified; no new data were collected and no ethics approval was required for
this secondary analysis.

1. Özdemir A.T., Barshan B. Simulated Falls and Daily Living Activities Data Set. UCI ML Repository,
   doi:10.24432/C52028 (CC BY 4.0). Paper: Sensors 2014, 14, 10691–10708.
2. Ustad A. et al. HAR70+. UCI ML Repository, https://archive.ics.uci.edu/dataset/780/har70 (CC BY 4.0).
   Paper: Sensors 2023, 23, 2368.
3. Bächlin M. et al. Daphnet Freezing of Gait. UCI ML Repository,
   https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait (CC BY 4.0). Paper: IEEE TITB 2010, 14, 436–446.
4. Moody G.B., Mark R.G. MIT-BIH Arrhythmia Database v1.0.0. PhysioNet, https://physionet.org/content/mitdb/1.0.0/
   (ODC-By 1.0). Goldberger A.L. et al. Circulation 2000, 101, e215–e220.
5. Cook D.J. et al. CASAS smart home dataset — scripted complex activities, activity scores and cognitive diagnosis.
   Zenodo, doi:10.5281/zenodo.15713579 (CC BY 4.0). Paper: Dawadi et al., IEEE TSMC Systems 2013, 43, 1302–1313.

SHA-256 checksums of the raw files: `data/raw/SHA256SUMS.txt`. The falls subset contains only accelerometer and
gyroscope channels plus age/sex/height/weight of the volunteers (names and contact data of the original
information files are not copied).

---------------------------------------------------------------------------------------------------------
## 6. Reproducibility notes

* Fixed seeds: 2026 (data splits, MiniRocket-lite), 11 / 23 / 47 (federated runs).
* The held-out person is never used for training, thresholds, calibration or model selection.
* All hyper-parameters are in `config.py` and Appendix A of the paper; none was tuned on test people.
* Network parameters of the alert-delivery simulation are assumptions (`results/tables/S12_simulation_assumptions.csv`).

## 7. Manuscript

`00_PAPER/` holds the Word manuscript and a PDF copy. It is generated by `node manuscript/build_manuscript.js`
from `results/paper_numbers.json` and `results/figures/`, so every number in the paper comes from the scripts above.
