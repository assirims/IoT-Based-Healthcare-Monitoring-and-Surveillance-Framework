
import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from config import PROCESSED_PATH, TAB_PATH, PRED_PATH, SEED, FL
from lib.features import extract
from lib.zoo import load_service, make_rocket, CARES

SVC = {"fog": dict(hop_s=1.0, ch=9, fs=64), "har70": dict(hop_s=2.5, ch=6, fs=50), "falls": dict(hop_s=0.4, ch=12, fs=25)}
SENS = {"fog": {"ankle": [0, 1, 2], "thigh": [3, 4, 5], "trunk": [6, 7, 8]}, "har70": {"back": [0, 1, 2], "thigh": [3, 4, 5]},
        "falls": {"waist_acc": [0, 1, 2], "waist_gyr": [3, 4, 5], "wrist_acc": [6, 7, 8], "wrist_gyr": [9, 10, 11]}}

NET = dict(ble_interval_ms=(7.5, 30.0), ble_loss=0.01, wan_median_ms=60.0, wan_sigma=0.5, rto_s=1.0, rto_max_s=8.0,
           broker_ms=5.0, push_median_s=0.5, push_sigma=0.6, push_fail=0.01, sms_after_s=15.0, sms_delay_s=(2.0, 10.0),
           alert_bytes=256, summary_bytes=512, n_alerts=20000)


def gateway_latency(service, reps=300):
    d = load_service(service)
    w = np.load(os.path.join(PROCESSED_PATH, f"{service}_windows.npz"))
    X = w["X"]
    if service == "falls":
        X = X[:, :, :12]
    idx = np.random.default_rng(SEED).choice(len(d["y"]), size=min(6000, len(d["y"])), replace=False)
    mr = make_rocket(service, d["R"][idx], seed=SEED)
    m = CARES(SEED).fit(d["F"][idx], mr.transform(d["R"][idx]), d["y"][idx], d["K"])
    one, r1 = X[:1], d["R"][:1]
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        F1, _ = extract(one, d["fs"], SENS[service])
        M1 = mr.transform(r1)
        m.predict_proba(F1, M1)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    size = len(pickle.dumps(m, protocol=4)) + len(pickle.dumps(mr, protocol=4))
    return dict(service=service, median_ms_per_window=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                windows_per_s_needed=1.0 / SVC[service]["hop_s"],
                persons_per_core=float(1000.0 / np.median(t) * SVC[service]["hop_s"]), model_kB=size / 1024,
                duty_cycle_pct=float(100 * np.median(t) / (1000 * SVC[service]["hop_s"])))


def mlp_param_bytes(d_in, K):
    h = FL["hidden"]
    sizes = [d_in] + list(h) + [K]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1)) * 4


def bandwidth(alert_rates):
    rows = []
    feat_dim = {"fog": 276, "har70": 184, "falls": 368, "ecg": 56}
    K = {"fog": 2, "har70": 6, "falls": 2, "ecg": 5}
    spec = dict(SVC, ecg=dict(hop_s=1 / 1.2, ch=1, fs=360))
    for s, c in spec.items():
        raw = c["ch"] * c["fs"] * 2 * 86400                         # 16-bit samples
        feats = feat_dim[s] * 4 * 86400 / c["hop_s"]
        alerts = alert_rates.get(s, 1.0) * 24 * NET["alert_bytes"]
        summ = 24 * NET["summary_bytes"]
        fl = mlp_param_bytes(feat_dim[s], K[s])                      # one model update per day (float32)
        edge = alerts + summ + fl
        rows.append(dict(service=s, raw_MB_per_day=raw / 1e6, features_MB_per_day=feats / 1e6, cares_MB_per_day=edge / 1e6,
                         reduction_vs_raw=raw / edge, fl_update_kB=fl / 1024, alerts_per_day=alert_rates.get(s, 1.0) * 24))
    return pd.DataFrame(rows)


def simulate(det_latency_s, loss, outage=False, n=NET["n_alerts"], seed=SEED):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(n):
        t = rng.choice(det_latency_s)                                       # detection latency (measured)
        # BLE wearable -> gateway (link-layer retransmission every connection interval)
        ci = rng.uniform(*NET["ble_interval_ms"]) / 1000
        t += ci
        while rng.random() < NET["ble_loss"]:
            t += ci
        # gateway -> broker, MQTT QoS 1 with retransmission timeout doubling up to a cap
        rto = NET["rto_s"]
        t_wan = 0.0
        if outage:
            t_wan += rng.uniform(0, 60)                                   # cloud unreachable for up to 1 min
        while True:
            if rng.random() >= loss and rng.random() >= loss:              # PUBLISH and PUBACK both delivered
                t_wan += rng.lognormal(np.log(NET["wan_median_ms"] / 1000), NET["wan_sigma"])
                break
            t_wan += rto
            rto = min(2 * rto, NET["rto_max_s"])
        t += t_wan + NET["broker_ms"] / 1000
        # broker -> caregiver phone: push notification, SMS fallback after 15 s
        if rng.random() >= NET["push_fail"]:
            push = rng.lognormal(np.log(NET["push_median_s"]), NET["push_sigma"])
        else:
            push = np.inf
        sms = NET["sms_after_s"] + rng.uniform(*NET["sms_delay_s"])
        out[i] = t + min(push, sms)
    return out


def main():
    lat = pd.DataFrame([gateway_latency(s) for s in ["fog", "har70", "falls"]])
    lat.to_csv(os.path.join(TAB_PATH, "T10_gateway_latency.csv"), index=False)
    print(lat.round(3).to_string(index=False))
    # alert rates per hour actually produced by the calibrated alarm manager (script 10, conformal policy)
    rates = {}
    for s, B in [("fog", 4.0), ("falls", 0.0)]:
        p = os.path.join(TAB_PATH, f"T6_alarm_budget_{s}.csv")
        if os.path.exists(p):
            a = pd.read_csv(p)
            r = a[(a.policy == "conformal (proposed)") & (a.budget_per_h == B)]
            rates[s] = float(r.pooled_fa_per_h.iloc[0]) + (1.0 if s == "fog" else 0.0)
    bw = bandwidth(rates)
    bw.to_csv(os.path.join(TAB_PATH, "T11_bandwidth.csv"), index=False)
    print(bw.round(3).to_string(index=False))
    # detection latency distribution of the fall service (script 10)
    fl = os.path.join(TAB_PATH, "S9b_falls_trials_conformal.csv")
    det = np.array([1.0, 1.5, 2.0, 2.5])
    if os.path.exists(fl):
        a = pd.read_csv(fl)
        v = a[a.fall & a.detected].latency_s.dropna().to_numpy()
        if len(v):
            det = np.clip(v, 0, None)
    rows, dist = [], {}
    for loss in [0.0, 0.01, 0.05, 0.10, 0.20]:
        e2e = simulate(det, loss)
        dist[f"loss {int(loss * 100)}%"] = e2e
        rows.append(dict(scenario=f"packet loss {int(loss * 100)}%", median_s=np.median(e2e), p95_s=np.percentile(e2e, 95),
                         p99_s=np.percentile(e2e, 99), within_10s=np.mean(e2e <= 10), within_30s=np.mean(e2e <= 30),
                         within_60s=np.mean(e2e <= 60)))
    e2e = simulate(det, 0.01, outage=True)
    dist["cloud outage (<=60 s)"] = e2e
    rows.append(dict(scenario="cloud outage up to 60 s, 1% loss", median_s=np.median(e2e), p95_s=np.percentile(e2e, 95),
                     p99_s=np.percentile(e2e, 99), within_10s=np.mean(e2e <= 10), within_30s=np.mean(e2e <= 30),
                     within_60s=np.mean(e2e <= 60)))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(TAB_PATH, "T12_alert_delivery_simulation.csv"), index=False)
    np.savez_compressed(os.path.join(PRED_PATH, "alert_delivery_sim.npz"), **{k: v.astype(np.float32) for k, v in dist.items()})
    pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in NET.items()]).to_csv(os.path.join(TAB_PATH, "S12_simulation_assumptions.csv"), index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
