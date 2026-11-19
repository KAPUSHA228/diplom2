import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

runs = Path("../../experiments_output/runs")
data = {}
raw_shap = {}

for d in runs.iterdir():
    c, s = d / "config.json", d / "shap_values.npy"
    if not (c.exists() and s.exists()):
        continue
    cfg = json.load(open(c, encoding="utf-8"))
    shap_3d = np.load(s)
    raw_shap[(cfg["dataset"], cfg["model"], cfg["balancer"])] = shap_3d
    data[(cfg["dataset"], cfg["model"], cfg["balancer"])] = np.abs(np.load(s)).mean(axis=1)  # (15, n_features)

rows = []
for (ds, m, b), imp in data.items():
    base_imp = data.get((ds, m, "baseline"))
    base_raw = raw_shap.get((ds, m, "baseline"))
    if base_imp is None or base_raw is None:
        continue

    synth = ds.startswith("synth_")
    for i in range(imp.shape[0]):
        v = imp[i]
        base_v = base_imp[i]

        target_raw = raw_shap[(ds, m, b)][i]
        base_raw_i = base_raw[i]

        top5 = set(np.argsort(-v)[:5])
        btop5 = set(np.argsort(-base_v)[:5])

        sign_agreement = np.mean(np.sign(base_raw_i) == np.sign(target_raw))

        rows.append(
            dict(
                dataset=ds,
                model=m,
                balancer=b,
                split=i,
                spearman=spearmanr(base_v, v)[0],
                overlap5=len(top5 & btop5) / 5,
                sign_agreement=sign_agreement,
                info5=len(top5 & set(range(5))) / 5 if synth else np.nan,
                signal5=len(top5 & set(range(7))) / 5 if synth else np.nan,
                mass7=v[:7].sum() / v.sum() if synth else np.nan,
            )
        )
pd.DataFrame(rows).to_csv("../../experiments_output/summary/per_split_metrics.csv", index=False)
print("Сохранено в per_split_metrics.csv")
