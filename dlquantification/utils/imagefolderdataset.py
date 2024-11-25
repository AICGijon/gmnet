import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


class IFCBImageFolderDataset(torch.utils.data.Dataset):
    """
    In this datset, the images are organized in folders by sample. Note that the idea of this dataset is that
    we do not have labels for each of the images
    """

    def __init__(self, root_dir, transform=None, minimun_ds_size=300):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.sample_idxs = {}

        index = 0
        for _, sample_dirs, _ in os.walk(root_dir):
            for sample_dir in tqdm(sample_dirs):
                for _, _, sample_imgs in os.walk(os.path.join(root_dir, sample_dir)):
                    num_sample_images = 0
                    if len(sample_imgs) > minimun_ds_size:  # consider only samples with enough examples
                        for file_name in sample_imgs:
                            if file_name.endswith(".png"):
                                self.image_paths.append(os.path.join(root_dir, sample_dir, file_name))
                                num_sample_images += 1
                        # finished reading images from this sample, store the indexes in a list
                        self.sample_idxs[sample_dir] = np.arange(index, index + num_sample_images)
                        index += num_sample_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (image,)
