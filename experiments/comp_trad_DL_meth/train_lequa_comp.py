import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dlquantification.transnet import TransNet
from dlquantification.histnet import HistNet
from dlquantification.gmnet import GMNet
from dlquantification.deepsets import DeepSets
from dlquantification.utils.utils import UnlabeledBagGenerator, UnlabeledMixerBagGenerator,APPBagGenerator
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
import os
from dlquantification.utils.lossfunc import MRAE
from dlquantification.utils.lossfunc import NMD
import json
import argparse
from tqdm import tqdm
import wandb


def get_n_classes(dataset):
    if dataset == "T2" or dataset == "T1B":
        return 28
    elif dataset == "T3":
        return 5
    else:
        raise ValueError("Dataset is not correct")

def standarize_tensors(tensors_standarize,ref_tensor,filename):
    print("Standarizing data:")
    if ref_tensor!=None:
        mean = ref_tensor.mean(dim=0)
        std = ref_tensor.std(dim=0)
        torch.save({'mean': mean, 'std': std}, filename)
    else:
        meanstd = torch.load(filename)
        mean = meanstd['mean']
        std = meanstd['std']
    
    for i in range(len(tensors_standarize)):
        tensors_standarize[i] = (tensors_standarize[i] - mean) / std
    return tensors_standarize


def load_dataset_T2_T1B(path, train_name, n_samples,n_labeled_examples, n_train_samples, n_val_samples, sample_size, n_features, include_labeled, standarize):
    if include_labeled:
        print("Loading training data...")
        training_labeled = pd.read_csv(os.path.join(path, "validation_training_data.csv"))

        x_train = training_labeled.iloc[0:n_labeled_examples, 1:].to_numpy()
        y_train = training_labeled.iloc[0:n_labeled_examples, 0].to_numpy()
        x_train = torch.FloatTensor(x_train)
        y_train = torch.LongTensor(y_train)
    print("Loading dev samples...")
    x_unlabeled_train = np.zeros((n_train_samples * sample_size, n_features)).astype(np.float32)
    x_unlabeled_val = np.zeros((n_val_samples * sample_size, n_features)).astype(np.float32)
    prevalences = pd.read_csv(os.path.join(path, "dev_prevalences.txt"))
    train_prevalences = torch.from_numpy(prevalences.iloc[0:n_train_samples, 1:].to_numpy().astype(np.float32))
    val_prevalences = torch.from_numpy(
        prevalences.iloc[n_train_samples : n_train_samples + n_val_samples, 1:].to_numpy().astype(np.float32)
    )

    for i in range(n_samples):
        sample = pd.read_csv(os.path.join(path, "dev_samples/{}.txt".format(i)))
        if i < n_train_samples:
            x_unlabeled_train[
                i * sample_size : (i + 1) * sample_size,
            ] = sample.to_numpy()
        else:
            j = i - n_train_samples
            x_unlabeled_val[
                j * sample_size : (j + 1) * sample_size,
            ] = sample.to_numpy()

    x_unlabeled_train = torch.from_numpy(x_unlabeled_train)
    x_unlabeled_val = torch.from_numpy(x_unlabeled_val)
    
    if include_labeled:
        if standarize:
            x_train,x_unlabeled_train,x_unlabeled_val = standarize_tensors(tensors_standarize=[x_train,x_unlabeled_train,x_unlabeled_val],ref_tensor=torch.cat((x_train, x_unlabeled_train)),filename="mean_std_{}.pth".format(train_name))
        dataset_train = TensorDataset(
            torch.cat((x_train, x_unlabeled_train)),
            torch.cat((y_train, torch.empty((x_unlabeled_train.shape[0]), dtype=torch.int64))),
        )
    else:
        if standarize:
            x_unlabeled_train,x_unlabeled_val=standarize_tensors(tensors_standarize=[x_unlabeled_train,x_unlabeled_val],ref_tensor=x_unlabeled_train,filename="mean_std_{}.pth".format(train_name))
        dataset_train = TensorDataset(x_unlabeled_train)
    
    dataset_val = TensorDataset(x_unlabeled_val)
    print("Done.")
    return dataset_train, train_prevalences, dataset_val, val_prevalences


def load_dataset_T3(path, train_name, n_samples, n_train_samples, n_val_samples, sample_size, n_features, include_labeled, standarize):
    """In this dataset we also have samples in the training"""
    x_train = np.zeros((n_train_samples * sample_size, n_features)).astype(np.float32)
    x_val = np.zeros((n_val_samples * sample_size, n_features)).astype(np.float32)
    x_train_labeled = np.zeros((100*sample_size, n_features)).astype(np.float32)
    y_train_labeled = np.zeros((100*sample_size)).astype(np.int32)
    train_prevalences = np.zeros((n_train_samples, 5))
    val_prevalences = np.zeros((n_val_samples, 5))

    # We have 100 training bags and 1000 dev bags
    print("Loading training bags...")
    prevalences = pd.concat(
        [
            pd.read_csv(os.path.join(path, "training_prevalences.txt")),
            pd.read_csv(os.path.join(path, "dev_prevalences.txt")),
        ]
    )

    for i in range(100):
        sample = pd.read_csv(os.path.join(path, "training_samples/{}.txt".format(i)))
        x_train_labeled[i*sample_size : (i + 1) * sample_size, ] = sample.iloc[:, 1:].to_numpy()
        y_train_labeled[i*sample_size : (i + 1) * sample_size, ] = sample.iloc[:, 0].to_numpy()
        x_train[
            i * sample_size : (i + 1) * sample_size,
        ] = sample.iloc[:, 1:].to_numpy()

    print("Loading dev bags...")

    for i in range(1000):
        sample = pd.read_csv(os.path.join(path, "dev_samples/{}.txt".format(i)))
        if i < n_train_samples - 100:
            x_train[
                (i+100) * sample_size : (i + 100 + 1) * sample_size,
            ] = sample.to_numpy()
        else:
            j = i - n_train_samples + 100
            x_val[
                j * sample_size : (j + 1) * sample_size,
            ] = sample.to_numpy()

    train_prevalences = torch.from_numpy(prevalences.iloc[0:n_train_samples, 1:].to_numpy().astype(np.float32))
    val_prevalences = torch.from_numpy(
        prevalences.iloc[n_train_samples : n_train_samples + n_val_samples, 1:].to_numpy().astype(np.float32)
    )

    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    x_train_labeled = torch.from_numpy(x_train_labeled)
    y_train_labeled = torch.from_numpy(y_train_labeled)
    
    if include_labeled: 
        if standarize:
            x_train_labeled,x_train,x_val=standarize_tensors(tensors_standarize=[x_train_labeled,x_train,x_val],ref_tensor=torch.cat((x_train_labeled, x_train)),filename="mean_std_{}.pth".format(train_name))       
        dataset_train = TensorDataset(
            torch.cat((x_train_labeled, x_train)),
            torch.cat((y_train_labeled, torch.empty((x_train.shape[0]), dtype=torch.int64))),
        )
    else:
        if standarize:
            x_train,x_val=standarize_tensors(tensors_standarize=[x_train,x_val],ref_tensor=x_train,filename="mean_std_{}.pth".format(train_name))
        dataset_train = TensorDataset(x_train)

    dataset_val = TensorDataset(x_val)
    print("Done.")
    return dataset_train, train_prevalences, dataset_val, val_prevalences


def train_lequa(
    data_path, 
    parameters_path,
    train_name,
    network,
    network_parameters,
    dataset,
    standarize,
    n_labeled_examples,
    app_bags_proportion=1,
    feature_extraction="rff",
    bag_generator="UnlabeledMixerBagGenerator",
    cuda_device="cuda:0",
):
    app_bags_proportion=float(app_bags_proportion)
    n_labeled_examples = int(n_labeled_examples)
    n_train_samples = n_labeled_examples//1000
    n_classes = get_n_classes(dataset)
    if dataset == "T2":
        path = os.path.join(data_path, "lequa2024/T2/public")
        common_param_path = os.path.join(parameters_path, "common_parameters_T2.json")
        n_val_samples = 1000-n_train_samples
        n_samples = 1000
        sample_size = 1000
        n_features = 256
        real_bags_proportion = 0.5 
        loss = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)
    elif dataset == "T1B":
        path = os.path.join(data_path, "leQua2022/T1B/public")
        common_param_path = os.path.join(parameters_path, "common_parameters_T1B.json")
        n_train_samples = 0
        n_val_samples = 1000
        n_samples = 1000
        sample_size = 1000
        n_features = 300
        real_bags_proportion = 0.5
        loss = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)
    elif dataset == "T3":
        path = os.path.join(data_path, "lequa2024/T3/public")
        common_param_path = os.path.join(parameters_path, "common_parameters_T3.json")
        n_train_samples = 770 
        n_val_samples = 330 
        n_samples = 1100
        sample_size = 200
        n_features = 256
        real_bags_proportion = 0.5 
        loss = loss_val = NMD()

    seed = 2032
    if dataset == "T2" or dataset == "T1B":
        include_labeled = bag_generator == "LeQuaBagGenerator"
        dataset_train, train_prevalences, dataset_val, val_prevalences = load_dataset_T2_T1B(
            path, train_name, n_samples,n_labeled_examples, n_train_samples, n_val_samples, sample_size, n_features, include_labeled, standarize
        )
    elif dataset == "T3":
        include_labeled = bag_generator == "LeQuaBagGenerator"
        dataset_train, train_prevalences, dataset_val, val_prevalences = load_dataset_T3(
            path, train_name, n_samples, n_train_samples, n_val_samples, sample_size, n_features, include_labeled, standarize
        )

    # Bag generators
    if bag_generator == "UnlabeledMixerBagGenerator":
        train_bag_generator = UnlabeledMixerBagGenerator(
            torch.device("cpu"),
            prevalences=train_prevalences,
            sample_size=sample_size,
            real_bags_proportion=real_bags_proportion,
            seed=seed,
        )
    elif bag_generator == "LeQuaBagGenerator":
        train_bag_generator = LeQuaBagGenerator(
            device=torch.device("cpu"),
            seed=seed,
            prevalences=train_prevalences,
            sample_size=sample_size,
            app_bags_proportion=app_bags_proportion, 
            mixed_bags_proportion=1 - real_bags_proportion,
            labeled_unlabeled_split=(
                range(0, n_labeled_examples),
                range(n_labeled_examples, n_labeled_examples + n_train_samples * sample_size),
            ),
        )
    else:
        raise ValueError("BagGenerator is not correct")

    val_bag_generator = UnlabeledBagGenerator(
        torch.device("cpu"), val_prevalences, sample_size, pick_all=True, seed=seed
    )

    torch.manual_seed(seed)

    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())
    with open(network_parameters, "r") as f:
        network_parameters = json.loads(f.read())

    if feature_extraction == "rff":
        fe = FCFeatureExtractionModule(
            input_size=n_features,
            output_size=network_parameters.pop("fe_output_size"),
            hidden_sizes=network_parameters.pop("fe_hidden_sizes"),
            dropout=network_parameters.pop("dropout_fe"),
        )
    elif feature_extraction == "nofe":
        fe = NoFeatureExtractionModule(input_size=n_features)

    

    parameters = {**common_parameters, **network_parameters}
    parameters["n_classes"] = n_classes
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    parameters["quant_loss"] = loss
    if dataset=='T3':
        parameters["quant_loss_val"] = loss_val
    parameters["dataset_name"] = dataset
    parameters["save_model_path"] = "savedmodels/" + train_name + ".pkl"
    parameters["wandb_experiment_name"] = train_name
    parameters["use_wandb"] = True
    parameters["use_multiple_devices"] = False
    parameters["num_workers"] = 8
    parameters["n_bags"] = [5000, n_val_samples, 1]
    print("Network parameteres: ", parameters)

    if network == "histnet":
        model = HistNet(**parameters)
    elif network == "settransformers":
        model = TransNet(**parameters)
    elif network == "deepsets":
        model = DeepSets(**parameters)
    elif network == "gmnet":
        model = GMNet(**parameters)
    else:
        raise ValueError("network has not a proper value")

    model.fit(dataset=dataset_train, val_dataset=dataset_val)
    wandb.log({"n_labeled_examples":n_labeled_examples})
    return model

def test_lequa(model, data_path, train_name, dataset, standarize):
    print("Testing the model...")
    n_test_bags = 5000
    if dataset=='T2':
        bag_size = 1000
        path = os.path.join(data_path, "lequa2024/")
        input_size=256
    elif dataset=='T1B':
        path = os.path.join(data_path, "leQua2022/")
        input_size=300
        bag_size = 1000
    elif dataset=='T3':
        bag_size = 200
        path = os.path.join(data_path, "lequa2024/")
        input_size=256
    
    n_classes = get_n_classes(dataset)

    if standarize:
        meanstd = torch.load('mean_std_{}.pth'.format(train_name))
        mean = meanstd['mean']
        std = meanstd['std']

    samples_to_predict_path = path + dataset + "/public/test_samples/"
    prevalences = pd.read_csv(os.path.join(path + dataset + "/public/test_prevalences.txt"))
    results = pd.DataFrame(columns=np.arange(n_classes), index=range(n_test_bags), dtype="float")
    results_errors = pd.DataFrame(columns=("AE", "RAE","NMD"), index=range(n_test_bags), dtype="float")
    if dataset=='T2' or dataset == 'T1B':
        loss_mrae = MRAE(eps=1.0 / (2 * bag_size), n_classes=28)
    loss_nmd = NMD()
    test_tensor = torch.empty(n_test_bags,bag_size,input_size)
    print("Loading test data...")
    for i in tqdm(range(n_test_bags)):
        test_bag = pd.read_csv(os.path.join(samples_to_predict_path, "{}.txt".format(i)))
        test_bag = torch.from_numpy(test_bag.to_numpy().astype(np.float32))
        if standarize:
            test_bag = (test_bag - mean) / std
        test_tensor[i,:] = test_bag
    test_dataset = TensorDataset(test_tensor)
    print("Evaluating test data...")
    p_hat = model.predict(test_dataset,process_in_batches=500)
    results = pd.DataFrame(p_hat.numpy())
    print("Computing errors...")
    for i in range(len(results)):
        if dataset == 'T2' or dataset == 'T1B':
            results_errors.iloc[i]["AE"] = torch.nn.functional.l1_loss(
                p_hat[i,:], torch.FloatTensor(prevalences.iloc[i, 1:])
            ).numpy()
            results_errors.iloc[i]["RAE"] = loss_mrae(
                torch.FloatTensor(prevalences.iloc[i, 1:]), p_hat[i,:]
            ).numpy()
        elif dataset == 'T3':
            results_errors.iloc[i]["NMD"] = loss_nmd(
                p_hat[i,:].unsqueeze(0), torch.FloatTensor(prevalences.iloc[i, 1:]).unsqueeze(0)
            ).numpy()


    results.to_csv(os.path.join("results/", train_name + ".txt"), index_label="id")
    results_errors.to_csv(os.path.join("results/", train_name + "_errors.txt"), index_label="id")
    print(results_errors.describe())
    rae_mean = results_errors.describe().loc['mean', 'RAE']
    wandb.log({"test_error":rae_mean})


if __name__ == "__main__":
    # Parametrice the script with argparse
    parser = argparse.ArgumentParser(description="LEQUA2024 training script")
    parser.add_argument("-t", "--train_name", help="Name for this training", required=True)
    parser.add_argument(
        "-n", "--network", help="network to use: histnet, settransformers, deepsets, gmnet", required=True
    )
    parser.add_argument("-p", "--network_parameters", help="File with the specific network parameters")
    parser.add_argument("-f", "--feature_extraction", help="nofe, rff")
    parser.add_argument("-b", "--bag_generator", help="Bag generator to use")
    parser.add_argument("-s", "--standarize", help="Standarize input data", action='store_true')
    parser.add_argument("-d", "--dataset", help="Dataset to use: T1B, T2, T3", required=True)
    parser.add_argument("-l", "--n_labeled_examples",required=False)
    parser.add_argument("-a", "--app_bags_proportion",required=False)
    parser.add_argument("-c", "--cuda_device", help="Device cuda:0 or cuda:1", required=True)
    print("Using following arguments:")
    args = vars(parser.parse_args())
    print(args)

    args["cuda_device"] = torch.device(args["cuda_device"])

    ## MODIFY THESE 2 PATHS
    data_path = "data/" #data directory
    parameters_path = "gmnet/experiments/parameters" #parameters directory
    
    model = train_lequa(data_path, parameters_path, **args)
    test_lequa(model,data_path,args["train_name"],args["dataset"],args["standarize"])
