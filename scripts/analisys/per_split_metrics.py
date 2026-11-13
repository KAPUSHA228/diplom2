import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

runs = Path("../../experiments_output/runs")
data = {}
for d in runs.iterdir():
    c, s = d / "config.json", d / "shap_values.npy"
    if not (c.exists() and s.exists()):
        continue
    cfg = json.load(open(c, encoding="utf-8"))
    data[(cfg["dataset"], cfg["model"], cfg["balancer"])] = np.abs(np.load(s)).mean(axis=1)  # (15, n_features)

rows = []
for (ds, m, b), imp in data.items():
    base = data.get((ds, m, "baseline"))
    if base is None:
        continue
    synth = ds.startswith("synth_")
    for i in range(imp.shape[0]):
        v = imp[i]
        top5 = set(np.argsort(-v)[:5])
        btop5 = set(np.argsort(-base[i])[:5])
        rows.append(
            dict(
                dataset=ds,
                model=m,
                balancer=b,
                split=i,
                spearman=spearmanr(base[i], v)[0],
                overlap5=len(top5 & btop5) / 5,
                info5=len(top5 & set(range(5))) / 5 if synth else np.nan,
                signal5=len(top5 & set(range(7))) / 5 if synth else np.nan,
                mass7=v[:7].sum() / v.sum() if synth else np.nan,
            )
        )
pd.DataFrame(rows).to_csv("../../experiments_output/summary/per_split_metrics.csv", index=False)
print("Сохранено в per_split_metrics.csv")
