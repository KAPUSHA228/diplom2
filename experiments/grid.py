"""Сетка экспериментов: что перебираем."""

EXPERIMENT_GRID = {
    "balancer": ["baseline", "smote", "borderline_smote", "adasyn", "smote_tomek", "smote_enn", "class_weight"],
    "model": ["LR", "RF", "XGB"],
    "dataset": ["student_autumn", "synth_01", "synth_05", "synth_10", "synth_20", "synth_30"],
    "cv_folds": 5,
    "n_repeats": 3,
    "primary_metric": "pr_auc",
    "shap_top_k": 5,
    "bootstrap_n": 30,
}
