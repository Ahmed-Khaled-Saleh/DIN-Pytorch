import torch
from torch import nn



class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes, bias= False)

    def forward(self, x, loss_ = None):
        x = self.linear(x)
        x = torch.clamp(x, -10, 10)
        out = torch.squeeze(torch.sigmoid(x))
        return out
