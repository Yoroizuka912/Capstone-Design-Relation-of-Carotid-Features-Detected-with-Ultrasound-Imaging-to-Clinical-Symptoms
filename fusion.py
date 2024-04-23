import os
import numpy as np
import torch
import random
from torchvision import models
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


seed = 3407
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18()
        # remove classification head
        dim_out_feature = 10
        resnet.fc = nn.Linear(resnet.fc.in_features, dim_out_feature)
        self.cnn = resnet
        # fusion classifier
        cluster_types = 4
        num_classes = 2
        self.linear = nn.Linear(dim_out_feature + cluster_types, num_classes)

    def forward(self, image, cluster):
        image_feature = self.cnn(image)
        fusion_feature = torch.cat((image_feature, cluster), dim=1)
        # fusion_feature = nn.ReLU()(fusion_feature)  # relu: more stable, but max test accuracy decreases
        return self.linear(fusion_feature)


class FusionDataset(Dataset):
    def __init__(self, image_root: str, cluster_root: str, image_transform=None):
        self.image_transform = image_transform
        self.image_paths = []
        self.cluster_paths = []
        self.labels = []
        label_names = os.listdir(image_root)

        for i, label_name in enumerate(label_names):
            image_names = [os.path.join(image_root, label_name, _) for _ in
                           os.listdir(os.path.join(image_root, label_name))]
            self.image_paths.extend(image_names)
            self.labels.extend([i] * len(image_names))

            cluster_names = [os.path.join(cluster_root, label_name, _) for _ in
                             os.listdir(os.path.join(cluster_root, label_name))]
            self.cluster_paths.extend(cluster_names)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path, mode=ImageReadMode.RGB).float()
        if self.image_transform:
            image = self.image_transform(image)
        cluster_path = self.cluster_paths[index]
        cluster = torch.tensor(np.load(cluster_path), dtype=torch.float)
        label = self.labels[index]
        return image_path, image, cluster, label

    def __len__(self):
        return len(self.labels)


