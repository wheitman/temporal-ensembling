import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import config
from temporal_ensembling import train, sample_train
from utils import GaussianNoise, savetime, save_exp
from time import time
from tqdm import trange
from timeit import default_timer as timer
import numpy as np


import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import prepare_mnist


class CNN(L.LightningModule):

    def __init__(
        self,
        batch_size,
        std,
        max_epochs,
        p=0.5,
        fm1=16,
        fm2=32,
        max_val=100.0,
        ramp_up_mult=-5.0,
        k=100,
        n_samples=60000,
        n_classes=10,
        lr=0.002,
        alpha=0.6,
    ):
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
        self.max_epochs = max_epochs
        self.max_val = max_val
        self.ramp_up_mult = ramp_up_mult
        self.n_classes = num_epochs
        self.n_samples = n_samples
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha

        train_dataset, test_dataset = prepare_mnist()
        ntrain = len(train_dataset)

        self.Z = torch.zeros(ntrain, n_classes).float().cuda()  # intermediate values
        self.z = torch.zeros(ntrain, n_classes).float().cuda()  # temporal outputs
        self.outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

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

    def ramp_up(self, epoch, max_epochs, max_val, mult):
        if epoch == 0:
            return 0.0
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1.0 - float(epoch) / max_epochs) ** 2)

    def weight_schedule(self, epoch, max_epochs, max_val, mult, n_labeled, n_samples):
        max_val = max_val * (float(n_labeled) / n_samples)
        return self.ramp_up(epoch, max_epochs, max_val, mult)

    def on_train_epoch_start(self):
        w = self.weight_schedule(
            self.current_epoch,
            self.max_epochs,
            self.max_val,
            self.ramp_up_mult,
            self.n_classes,
            self.n_samples,
        )

        self.log("unsupervised_loss_weight", w)

        # turn it into a usable pytorch object
        self.w = torch.autograd.Variable(
            torch.tensor([w], device="cuda"), requires_grad=False
        )

    def on_train_epoch_end(self):
        self.Z = self.alpha * self.Z + (1.0 - self.alpha) * self.outputs
        self.z = self.Z * (1.0 / (1.0 - self.alpha ** (self.current_epoch + 1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        zcomp = torch.autograd.Variable(
            self.z[i * self.batch_size : (i + 1) * self.batch_size], requires_grad=False
        )
        loss, supervised_loss, unsupervised_loss, nbsup = self.temporal_loss(
            y_pred, zcomp, self.w, y
        )

        self.log("loss", loss)
        self.log("supervised_loss", nbsup * supervised_loss)
        self.log("unsupervised_loss", unsupervised_loss)

        # calculate acc
        labels_hat = torch.argmax(y_pred, dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": train_acc,
                "supervised_loss": nbsup * supervised_loss,
                "unsupervised_loss": unsupervised_loss,
            },
        )

        return loss

    def temporal_loss(self, out1, out2, w, labels):

        # MSE between current and temporal outputs
        def mse_loss(out1, out2):
            quad_diff = torch.sum(
                (F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2
            )
            return quad_diff / out1.data.nelement()

        def masked_crossentropy(out, labels):
            cond = labels >= 0
            nnz = torch.nonzero(cond)
            nbsup = len(nnz)
            # check if labeled samples in batch, return 0 if none
            if nbsup > 0:
                masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
                masked_labels = labels[cond]
                loss = F.cross_entropy(masked_outputs, masked_labels)
                return loss, nbsup
            return (
                torch.autograd.Variable(
                    torch.FloatTensor([0.0]).cuda(), requires_grad=False
                ),
                0,
            )

        sup_loss, nbsup = masked_crossentropy(out1, labels)
        unsup_loss = mse_loss(out1, out2)
        return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.99))

        # Setting gamma to 1.0 basically turns the lr_scheduler off
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=1.0
        )

        return [optimizer], [lr_scheduler]

    def calc_metrics(model, loader):
        correct = 0
        total = 0
        for i, (samples, labels) in enumerate(loader):
            with torch.no_grad:
                samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data.view_as(predicted)).sum()

        acc = 100 * float(correct) / total
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        zcomp = torch.autograd.Variable(
            self.z[i * self.batch_size : (i + 1) * self.batch_size], requires_grad=False
        )
        loss, supervised_loss, unsupervised_loss, nbsup = self.temporal_loss(
            y_pred, zcomp, self.w, y
        )

        # with torch.no_grad:
        #     samples = Variable(samples.cuda())
        # labels = Variable(labels.cuda())
        # outputs = model(x)
        # _, predicted = torch.max(outputs.data, 1)
        # total += y.size(0)
        # correct += (predicted == y.data.view_as(predicted)).sum()

        # acc = correct/total

        # self.log("test_accuracy", acc)

        # calculate acc
        labels_hat = torch.argmax(y_pred, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({"test_loss": loss, "test_acc": test_acc})


# metrics
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []


ts = savetime()
cfg = vars(config)

for i in trange(cfg["n_seed_restarts"], desc="Seed restart"):
    seed = cfg["seeds"][i]
    L.seed_everything(seed)
    num_epochs = cfg["num_epochs"]
    model = CNN(
        cfg["batch_size"],
        cfg["std"],
        num_epochs,
    )

    # make data loaders
    n_classes = 10
    k = 100
    batch_size = 100
    train_dataset, test_dataset = prepare_mnist()

    train_loader, test_loader, indices = sample_train(
        train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=True
    )

    trainer = L.Trainer(max_epochs=num_epochs)
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)

    # acc, acc_best, l, sl, usl, indices = train(model, seed, num_epochs, **cfg)
    # accs.append(acc)
    # accs_best.append(acc_best)
    # losses.append(l)
    # sup_losses.append(sl)
    # unsup_losses.append(usl)
    # idxs.append(indices)

# for i in trange(cfg["n_exp"], desc="Seed restart"):
#     seed = cfg["seeds"][i]
#     num_epochs = cfg["num_epochs"]
#     model = CNN(
#         cfg["batch_size"],
#         cfg["std"],
#         num_epochs,
#     )

#     acc, acc_best, l, sl, usl, indices = train(model, seed, num_epochs, **cfg)
#     accs.append(acc)
#     accs_best.append(acc_best)
#     losses.append(l)
#     sup_losses.append(sl)
#     unsup_losses.append(usl)
#     idxs.append(indices)

# print("saving experiment")

# save_exp(ts, losses, sup_losses, unsup_losses, accs, accs_best, idxs, **cfg)
