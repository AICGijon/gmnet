import pandas as pd
from dlquantification.utils.lossfunc import MAE, MRAE, NMD
import torch

print("Computing errors...")

methods = ['CC','PCC','ACC','PACC','DMy','EMQ','EMQ-Platt']
datasets = ['cifar10']

n_test_bags = 1000

for dataset in datasets:
    sample_size = 500
    n_classes = 10
    test_prevalences = pd.read_csv(f"{dataset}_testbags/prevalences.csv", index_col=0).values
    for method in methods:
        path = f"results_traditional/{method}_{dataset}/task_{dataset}.csv"
        results = pd.read_csv(path, sep=",", index_col=0)
        results_errors = pd.DataFrame(columns=("AE", "RAE"), index=range(n_test_bags), dtype="float")
        loss_mrae = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)
        for i in range(n_test_bags):
            p_hat = torch.from_numpy(results.iloc[i].to_numpy())
            p = torch.from_numpy(test_prevalences[i])
            ae = torch.nn.functional.l1_loss(p_hat, p).item()
            rae = loss_mrae(p, p_hat).item()
            results_errors.loc[i] = [ae, rae]
        results_errors.to_csv(f"results/{dataset}_{method}_mrae_errors.txt", index_label="id")
