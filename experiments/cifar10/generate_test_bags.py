import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from dlquantification.utils.utils import APPBagGenerator

def load_feature_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    features = torch.tensor(df.drop(columns="label").values, dtype=torch.float32)
    return features, labels

def generate_test_bags(dataset_name, n_bags=1000, bag_size=500, seed=42):
    print("Loading test data...")
    csv_test_path = f"{dataset_name}_test.csv"
    x_test, y_test = load_feature_dataset(csv_test_path)

    print("Creating bag generator...")
    bag_generator = APPBagGenerator(seed=seed,device='cpu')

    print(f"Generating {n_bags} bags of size {bag_size}...")
    indices_list, prevalences = bag_generator.compute_bags(n_bags=n_bags, bag_size=bag_size, y=y_test)

    output_dir = f"{dataset_name}_testbags"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving bags to '{output_dir}'...")
    for i, indices in tqdm(enumerate(indices_list), total=n_bags):
        bag_path = os.path.join(output_dir, f"{i}.txt")
        with open(bag_path, "w") as f:
            bag_data = x_test[indices].numpy()
            np.savetxt(bag_path, bag_data, fmt="%.5f", delimiter=",")

    prevalences_df = pd.DataFrame(prevalences.numpy())
    prevalences_df.to_csv(os.path.join(output_dir, "prevalences.csv"), index_label="id")

    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_bags.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    generate_test_bags(dataset_name, n_bags=1000, bag_size=500)
