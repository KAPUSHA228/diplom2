"""Сетка экспериментов: что перебираем."""

synth_datasets = [f"synth_{r:02d}_r{i}" for r in [1, 5, 10, 20, 30] for i in range(1, 4)]

EXPERIMENT_GRID = {
    "balancer": ["baseline", "smote", "borderline_smote", "adasyn", "smote_tomek", "smote_enn", "class_weight"],
    "model": ["LR", "RF", "XGB"],
    "dataset": synth_datasets + ["uci_student_dropout"] + ["synth_lr_test"],
    "cv_folds": 5,
    "n_repeats": 3,
    "primary_metric": "pr_auc",
    "shap_top_k": 5,
    "bootstrap_n": 30,
}
