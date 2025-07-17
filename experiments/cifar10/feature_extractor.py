import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd

class FeatureCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(FeatureCNN, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x16
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 8x8
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 4x4
            nn.Dropout(0.25),
        )

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),                  # 256 × 4 × 4 = 4096
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.feature_extractor(x)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.conv_block(x)
        x = self.feature_extractor(x)
        return x

def get_dataset(name, train, transform):
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Dataset '{name}' not supported.")
    return dataset, input_channels, num_classes

def train_model(model, device, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_train, correct_train = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        acc_train = 100. * correct_train / total_train

        model.eval()
        total_val, correct_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        acc_val = 100. * correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {acc_train:.2f}% - Val Acc: {acc_val:.2f}%")

def extract_and_save_features(model, device, loader, split, output_prefix):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats = model.extract_features(inputs).cpu().numpy()
            labels = labels.cpu().numpy().astype(int)
            features_list.append(feats)
            labels_list.append(labels)

    features_all = np.vstack(features_list)
    labels_all = np.concatenate(labels_list)

    # Round and format
    features_all = np.round(features_all, 5)

    # Create DataFrame
    feature_columns = [str(i) for i in range(256)]
    df_features = pd.DataFrame(features_all, columns=feature_columns)
    df_labels = pd.Series(labels_all, name="label", dtype="int")

    df = pd.concat([df_labels, df_features], axis=1)
    df.to_csv(f"{output_prefix}_{split}.csv", index=False, float_format="%.5f")



def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(2032)
    np.random.seed(2032)

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    full_train_set, input_channels, num_classes = get_dataset(args.dataset, train=True, transform=transform_train)
    test_set, _, _ = get_dataset(args.dataset, train=False, transform=transform_test)

    val_size = int(0.1 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = FeatureCNN(input_channels, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {args.dataset} in {args.device}")
    train_model(model, device, train_loader, val_loader, optimizer, criterion, args.epochs)

    print("Extracting features...")
    full_train_loader = DataLoader(full_train_set, batch_size=64, shuffle=False)
    extract_and_save_features(model, device, full_train_loader, 'train', args.dataset)
    extract_and_save_features(model, device, test_loader, 'test', args.dataset)
    print("Done.")

if __name__ == '__main__':
    main()
