from experiments.runner import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run(n_jobs=-1)
    # runner.run(
    #     datasets=["synth_lr_test"],
    #     models=["LR"],
    #     balancers=["baseline", "smote", "class_weight"],
    #     n_jobs=-1
    # )
