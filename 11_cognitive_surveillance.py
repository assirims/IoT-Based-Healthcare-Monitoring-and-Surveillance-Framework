
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_PATH, TAB_PATH, PRED_PATH, SEED
from lib.metrics import youden_threshold

IMPAIRED = ["dementia", "MCI"]
HEALTHY = ["middle-aged", "young-old", "old-old"]


def models():
    return {
        "LR": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)),
        "RF": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                            RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=1)),
        "HGB": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0,
                                              class_weight="balanced", random_state=SEED),
    }


def cv_predict(X, y, reps=10):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=SEED)
    P = {m: np.zeros((reps, len(y))) for m in list(models()) + ["CARES-Cog (ensemble)"]}
    for k, (tr, te) in enumerate(cv.split(X, y)):
        rep = k // 5
        ps = []
        for nm, m in models().items():
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            P[nm][rep, te] = p
            ps.append(p)
        P["CARES-Cog (ensemble)"][rep, te] = np.mean(ps, 0)
    return P


def summarise(task, y, P, rows):
    for nm, Pr in P.items():
        aucs = [roc_auc_score(y, Pr[r]) for r in range(len(Pr))]
        aps = [average_precision_score(y, Pr[r]) for r in range(len(Pr))]
        pm = Pr.mean(0)
        th = youden_threshold(y, pm)          # descriptive operating point on the averaged CV scores
        yh = (pm >= th).astype(int)
        rows.append(dict(task=task, model=nm, n=len(y), n_impaired=int(y.sum()), AUROC=np.mean(aucs),
                         AUROC_lo=np.percentile(aucs, 2.5), AUROC_hi=np.percentile(aucs, 97.5), AUROC_sd=np.std(aucs),
                         AUPRC=np.mean(aps), Sens=yh[y == 1].mean(), Spec=1 - yh[y == 0].mean(),
                         BalAcc=balanced_accuracy_score(y, yh)))


def main():
    D = pd.read_csv(os.path.join(PROCESSED_PATH, "cog_features.csv"))
    feat = [c for c in D.columns if c.startswith(("t0", "t1", "t2", "uncued_", "cued_", "dayout_")) and c not in
            ("dayout_accuracy", "dayout_sequencing")] + ["n_tasks"]
    rows = []
    out_pred = {}
    for task, impaired, healthy in [("A: impaired vs healthy (45+)", IMPAIRED, HEALTHY),
                                    ("B: impaired vs healthy (60+)", IMPAIRED, ["young-old", "old-old"]),
                                    ("C: dementia vs healthy (60+)", ["dementia"], ["young-old", "old-old"]),
                                    ("D: MCI vs healthy (60+)", ["MCI"], ["young-old", "old-old"])]:
        S = D[D.group.isin(impaired + healthy)].reset_index(drop=True)
        X = S[feat].to_numpy(float); y = S.group.isin(impaired).astype(int).to_numpy()
        P = cv_predict(X, y)
        summarise(task, y, P, rows)
        out_pred[task] = (S.id.to_numpy(), y, P["CARES-Cog (ensemble)"].mean(0), S.group.to_numpy())
        if task.startswith("A"):
            # dementia vs MCI vs healthy: mean CV risk score per group (monotonic ordering check)
            grp = pd.DataFrame(dict(group=S.group, risk=P["CARES-Cog (ensemble)"].mean(0)))
            g = grp.groupby("group").risk.describe()[["count", "mean", "25%", "50%", "75%"]]
            g.to_csv(os.path.join(TAB_PATH, "S10_cognitive_risk_by_group.csv"))
            kw = stats.kruskal(*[grp.risk[grp.group == x] for x in ["dementia", "MCI", "young-old", "old-old", "middle-aged"]])
            rho = stats.spearmanr(grp.group.map({"dementia": 2, "MCI": 1}).fillna(0), grp.risk).correlation
            print(g.round(3)); print("Kruskal p=%.2e, Spearman(severity, risk)=%.3f" % (kw.pvalue, rho))
            pd.DataFrame([dict(kruskal_p=kw.pvalue, spearman_severity_risk=rho)]).to_csv(
                os.path.join(TAB_PATH, "S10b_cognitive_risk_ordering.csv"), index=False)
            # permutation importance aggregated by task (HGB refitted on all data; descriptive)
            m = models()["HGB"].fit(X, y)
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(m, X, y, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=1)
            imp = pd.DataFrame(dict(feature=feat, importance=pi.importances_mean))
            imp["task"] = imp.feature.str.extract(r"^(t\d\d|uncued|cued|dayout|n_tasks)")[0]
            imp["kind"] = imp.feature.str.replace(r"^(t\d\d_|uncued_|cued_|dayout_)", "", regex=True)
            imp.groupby("task").importance.sum().sort_values(ascending=False).to_csv(os.path.join(TAB_PATH, "S11_cognitive_importance_by_task.csv"))
            imp.groupby("kind").importance.sum().sort_values(ascending=False).to_csv(os.path.join(TAB_PATH, "S11b_cognitive_importance_by_feature.csv"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB_PATH, "T7_cognitive_surveillance.csv"), index=False)
    print(T.round(3).to_string(index=False))
    # ---------------- Task E: task-quality score ----------------
    S = D[D.score_sum_1_8.notna() & (D.score_sum_1_8 > 0) & D.group.isin(IMPAIRED + HEALTHY + ["at risk", "other medical"])].reset_index(drop=True)
    X = S[feat].to_numpy(float); yq = S.score_sum_1_8.to_numpy(float)
    pred = np.zeros((10, len(yq)))
    for rep in range(10):
        for tr, te in KFold(5, shuffle=True, random_state=SEED + rep).split(X):
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0,
                                              random_state=SEED).fit(X[tr], yq[tr])
            pred[rep, te] = m.predict(X[te])
    rhos = [stats.spearmanr(yq, pred[r]).correlation for r in range(10)]
    pm = pred.mean(0)
    C = pd.DataFrame([dict(n=len(yq), spearman_rho=np.mean(rhos), rho_lo=np.percentile(rhos, 2.5), rho_hi=np.percentile(rhos, 97.5),
                           MAE=np.mean(np.abs(pm - yq)), MAE_mean_baseline=np.mean(np.abs(yq - yq.mean())))])
    C.to_csv(os.path.join(TAB_PATH, "T7b_task_quality_regression.csv"), index=False)
    print(C.round(3).to_string(index=False))
    ida, ya, pa, ga = out_pred["A: impaired vs healthy (45+)"]
    np.savez_compressed(os.path.join(PRED_PATH, "cog_predictions.npz"), id=ida, y=ya, p=pa, group=ga.astype(str),
                        yq=yq, pq=pm)


if __name__ == "__main__":
    main()
