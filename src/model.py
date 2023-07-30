import torch
from torch import nn



class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes, bias= False)

    def forward(self, x):
        out = self.linear(x).squeeze()
        return out
