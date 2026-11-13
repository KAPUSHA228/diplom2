import pandas as pd
from sklearn.model_selection import StratifiedKFold
from balancing.registry import BALANCERS
from experiments.runner import _load_dataset, _fit_model
from ml_core.models.registry import get_model
from xai.explainers import get_shap_values
from xai.stability import rank_correlation


def shap_for(X_tr, y_tr, X_te, bal, seed):
    cls = BALANCERS[bal]
    b = cls(random_state=seed) if bal == "smote" else cls()
    Xr, yr = b.fit_resample(X_tr, y_tr)
    sw = b.get_sample_weight(yr) if bal == "class_weight" else None
    m = get_model("RF")
    m.set_params(random_state=seed)
    return get_shap_values(_fit_model(m, Xr, yr, sw), X_te)


rows = []
for ds in ["synth_05_r1", "synth_10_r1", "synth_10_r2", "synth_10_r3", "synth_20_r1"]:
    X, y = _load_dataset(ds)
    for rep in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, y):
            Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
            S = {(b, s): shap_for(Xtr, ytr, Xte, b, s) for b in ["baseline", "smote", "class_weight"] for s in (42, 43)}
            rows.append(
                dict(
                    dataset=ds,
                    noise_baseline=rank_correlation(S["baseline", 42], S["baseline", 43]),
                    noise_smote=rank_correlation(S["smote", 42], S["smote", 43]),
                    noise_cw=rank_correlation(S["class_weight", 42], S["class_weight", 43]),
                    effect_smote=rank_correlation(S["baseline", 42], S["smote", 42]),
                    effect_cw=rank_correlation(S["baseline", 42], S["class_weight", 42]),
                )
            )
            print(ds, rep, len(rows), flush=True)
df = pd.DataFrame(rows)
print(df.groupby("dataset").mean().round(3))
df.to_csv("experiments_output/summary/noise_control.csv", index=False)
