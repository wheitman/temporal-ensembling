import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import config
from temporal_ensembling import train
from utils import GaussianNoise, savetime, save_exp
from time import time
from tqdm import trange


class CNN(nn.Module):

    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(CNN, self).__init__()

        self.fm1 = fm1
        self.fm2 = fm2
        self.std = std
        self.gaussian_noise = GaussianNoise(batch_size, std=self.std)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)

        self.model = nn.Sequential(
            nn.Conv2d(1, self.fm1, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.fm1, self.fm2, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.training:
            x = self.gaussian_noise(x)

        x = self.model(x)

        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


# metrics
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []


ts = savetime()
cfg = vars(config)

for i in trange(cfg["n_exp"], desc="Seed restart"):
    model = CNN(cfg["batch_size"], cfg["std"])
    seed = cfg["seeds"][i]
    acc, acc_best, l, sl, usl, indices = train(model, seed, **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)

print("saving experiment")

save_exp(ts, losses, sup_losses, unsup_losses, accs, accs_best, idxs, **cfg)
