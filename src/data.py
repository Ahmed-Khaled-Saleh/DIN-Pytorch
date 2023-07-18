



from torch.utils.data import Dataset, DataLoader


import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class A9ADataset(Dataset):
    def __init__(self, file_path):
        self.features, self.labels = self._load_data(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index].squeeze()

    def _load_data(self, file_path):
        data = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                labels.append(int(line[0]))
                feature_vector = [0] * 123  # Assuming the data has 123 features
                for item in line[1:]:
                    if ':' in item:
                        feature_index, feature_value = item.split(':')
                        feature_vector[int(feature_index)-1] = float(feature_value)
                data.append(feature_vector)
        return torch.tensor(data), torch.tensor(labels)

