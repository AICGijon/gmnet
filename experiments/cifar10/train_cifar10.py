import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from dlquantification.utils.utils import APPBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.histnet import HistNet
from dlquantification.gmnet import GMNet
from dlquantification.deepsets import DeepSets
from dlquantification.utils.lossfunc import MAE, MRAE, NMD
import os
from tqdm import tqdm
import json


def load_feature_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    features = torch.tensor(df.drop(columns="label").values, dtype=torch.float32)
    return features, labels


def train_model(train_name, dataset_name, network, network_parameters_path, feature_extraction,loss_function, cuda_device):
    print("Loading training data...")
    csv_train_path = f"{dataset_name}_train.csv"
    x_all, y_all = load_feature_dataset(csv_train_path)
    seed = 42

    # Split 10,000 examples for validation
    val_size = 10000
    x_val = x_all[:val_size]
    y_val = y_all[:val_size]
    x_train = x_all[val_size:]
    y_train = y_all[val_size:]

    # Calcular media y std con x_train
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std[std == 0] = 1  # evitar división por cero

    # Guardar para uso posterior en test
    torch.save({'mean': mean, 'std': std}, f"./meanstd/mean_std_{train_name}.pth")

    # Estandarizar
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    n_classes = len(torch.unique(y_train))

    train_bag_generator = APPBagGenerator(device='cpu', seed=seed)
    val_bag_generator = APPBagGenerator(device='cpu', seed=seed)

    if dataset_name == "cifar10":
        common_param_path = os.path.join("../parameters/common_parameters_cifar10.json")
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")


    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())

    with open(network_parameters_path, "r") as f:
        network_parameters = json.load(f)

    if feature_extraction == "rff":
        fe = FCFeatureExtractionModule(
            input_size=256,
            output_size=network_parameters.pop("fe_output_size"),
            hidden_sizes=network_parameters.pop("fe_hidden_sizes"),
            dropout=network_parameters.pop("dropout_fe"),
        )
    elif feature_extraction == "nofe":
        fe = NoFeatureExtractionModule(input_size=256)

    parameters = {**common_parameters, **network_parameters}
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    if loss_function == "mae":
        parameters["quant_loss"] = MAE()
    elif loss_function == "mrae":
        parameters["quant_loss"] = MRAE(eps=1.0 / (2 * parameters["bag_size"]), n_classes=n_classes)
    parameters["save_model_path"] = "savedmodels/" + train_name + ".pkl"
    parameters["wandb_experiment_name"] = train_name
    parameters["use_multiple_devices"] = False
    parameters["num_workers"] = 8
    print("Network parameteres: ", parameters)

    print("Initializing model...")
    if network == "histnet":
        model = HistNet(**parameters)
    elif network == "deepsets":
        model = DeepSets(**parameters)
    elif network == "gmnet":
        model = GMNet(**parameters)
    else:
        raise ValueError("Invalid network name")

    print("Training model...")
    #model.fit(dataset=train_dataset, val_dataset=val_dataset)
    return model, x_train.shape[1], n_classes


def test_model(model, dataset_name, train_name, n_classes, cuda_device):
    print("Loading mean and std...")
    meanstd = torch.load(f"./meanstd/mean_std_{train_name}.pth")
    mean = meanstd['mean']
    std = meanstd['std']

    bag_dir = f"{dataset_name}_testbags"
    prevalences_path = os.path.join(bag_dir, "prevalences.csv")
    print(f"Loading prevalences from {prevalences_path}...")
    prevalences_df = pd.read_csv(prevalences_path, index_col="id")
    test_prevalences = torch.tensor(prevalences_df.values, dtype=torch.float32)

    n_test_bags = len(test_prevalences)
    print(f"Loading and standardizing {n_test_bags} bags...")

    # Load all bags into one tensor
    first_bag = np.loadtxt(os.path.join(bag_dir, "0.txt"), delimiter=",", dtype=np.float32)
    bag_size, input_size = first_bag.shape
    test_tensor = torch.empty((n_test_bags, bag_size, input_size), dtype=torch.float32)

    for i in tqdm(range(n_test_bags)):
        bag_path = os.path.join(bag_dir, f"{i}.txt")
        bag = np.loadtxt(bag_path, delimiter=",", dtype=np.float32)
        bag = torch.from_numpy(bag)
        bag = (bag - mean) / std
        test_tensor[i] = bag

    test_dataset = TensorDataset(test_tensor)

    print("Predicting in batches...")
    p_hat = model.predict(test_dataset, process_in_batches=500)  # shape: (n_test_bags, n_classes)

    print("Evaluating...")
    results = pd.DataFrame(p_hat.numpy())
    results_errors = pd.DataFrame(columns=("AE", "RAE", "NMD"), index=range(n_test_bags), dtype="float")

    sample_size = bag_size
    loss_mrae = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)
    loss_nmd = NMD()

    for i in range(n_test_bags):
        ae = torch.nn.functional.l1_loss(p_hat[i], test_prevalences[i]).item()
        rae = loss_mrae(test_prevalences[i], p_hat[i]).item()
        nmd = loss_nmd(p_hat[i].unsqueeze(0), test_prevalences[i].unsqueeze(0)).item()
        results_errors.loc[i] = [ae, rae, nmd]

    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    results.to_csv(f"results/{train_name}.txt", index_label="id")
    results_errors.to_csv(f"results/{train_name}_errors.txt", index_label="id")
    print(results_errors.describe())



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2032)
    np.random.seed(2032)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_name", required=True)
    parser.add_argument("--dataset", required=True, help="Name of the dataset: fashion-mnist, cifar10, cifar100")
    parser.add_argument("--network", choices=["histnet", "settransformers", "deepsets", "gmnet"], required=True)
    parser.add_argument("--network_parameters", required=True)
    parser.add_argument("--feature_extraction", help="nofe, rff")
    parser.add_argument("--loss_function", help="mae,mrae")
    parser.add_argument("--cuda_device", default="cuda:0")
    args = parser.parse_args()

    args.cuda_device = torch.device(args.cuda_device)

    model, input_size, n_classes = train_model(
        train_name=args.train_name,
        dataset_name=args.dataset,
        network=args.network,
        network_parameters_path=args.network_parameters,
        feature_extraction=args.feature_extraction,
        loss_function=args.loss_function,
        cuda_device=args.cuda_device
    )

    #Measure time
    import time
    start_time = time.time()
    print("Testing model...")
    test_model(
        model=model,
        dataset_name=args.dataset,
        train_name=args.train_name,
        n_classes=n_classes,
        cuda_device=args.cuda_device
    )
    end_time = time.time()
    print(f"Testing completed in {end_time - start_time:.2f} seconds.")
