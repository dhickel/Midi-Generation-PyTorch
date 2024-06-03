import torch
from torch import nn
import torch.nn.functional as F


class IncepBottleNeckModule(nn.Module):
    def __init__(self, in_channels, out_channels,bn_channels):
        super(IncepBottleNeckModule, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=4, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Conv1d(4 * out_channels, bn_channels, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        x2 = F.pad(x, (1,2))
        branch3 = self.branch3(x2)
        branch4 = self.branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        x = F.relu(x)
        return self.bottleneck(x)



class IncepModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IncepModule, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return F.leaky_relu(torch.cat([branch1, branch2, branch3, branch4], 1))

class IncepModule2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IncepModule2, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=4, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=8, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=1, padding=0),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(F.pad(x, (1,2), 'constant'))
        branch3 = self.branch3(F.pad(x, (3,4), 'constant'))
        branch4 = self.branch4(F.pad(x, (1,2), 'constant'))
        return F.leaky_relu(torch.cat([branch1, branch2, branch3, branch4], 1))

class IncepModuleMidi(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IncepModuleMidi, self).__init__()

        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=4, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=8, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(F.pad(x, (1,2), 'constant'))
        branch4 = self.branch4(F.pad(x, (3,4), 'constant'))
        return F.leaky_relu(torch.cat([branch1, branch2, branch3, branch4], 1))