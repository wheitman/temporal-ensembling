import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import calc_metrics, prepare_mnist, weight_schedule
from tqdm import trange


def sample_train(
    train_dataset,
    test_dataset,
    batch_size,
    k,
    n_classes,
    seed,
    shuffle_train=True,
    return_idxs=True,
):

    n = len(train_dataset)
    rrng = np.random.RandomState(seed)

    cpt = 0
    indices = torch.zeros(k)
    other = torch.zeros(n - k)
    card = k // n_classes

    for i in range(n_classes):
        class_items = (train_dataset.targets == i).nonzero()

        # print(f"Class items:\n {class_items}")
        # print(f"Card: {card}")

        n_samples_in_class = len(class_items)
        rd = np.random.permutation(np.arange(n_samples_in_class))

        # print(f"Indices shape: {indices.shape}")
        # print(f"class_items shape: {class_items.shape}")
        # print(f"rd shape: {rd.shape}")

        # indices[0:10] = class_items[5923[:10]]
        indices[i * card : (i + 1) * card] = class_items[rd[:card]].squeeze()
        other[cpt : cpt + n_samples_in_class - card] = class_items[rd[card:]].squeeze()
        cpt += n_samples_in_class - card

    other = other.long()
    train_dataset.targets[other] = -1

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle_train,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False
    )

    if return_idxs:
        return train_loader, test_loader, indices
    return train_loader, test_loader


def temporal_loss(out1, out2, w, labels):

    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
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
        return Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=False), 0

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


def train(
    model,
    seed,
    k=100,
    alpha=0.6,
    lr=0.002,
    beta2=0.99,
    num_epochs=150,
    batch_size=100,
    drop=0.5,
    std=0.15,
    fm1=16,
    fm2=32,
    divide_by_bs=False,
    w_norm=False,
    data_norm="pixelwise",
    early_stop=None,
    c=300,
    n_classes=10,
    max_epochs=80,
    max_val=30.0,
    ramp_up_mult=-5.0,
    n_samples=60000,
    print_res=True,
    **kwargs,
):

    # retrieve data
    train_dataset, test_dataset = prepare_mnist()
    ntrain = len(train_dataset)

    # build model
    model.cuda()

    # make data loaders
    train_loader, test_loader, indices = sample_train(
        train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=False
    )

    # setup param optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # train
    model.train()
    losses = []
    sup_losses = []
    unsup_losses = []
    best_loss = 20.0

    Z = torch.zeros(ntrain, n_classes).float().cuda()  # intermediate values
    z = torch.zeros(ntrain, n_classes).float().cuda()  # temporal outputs
    outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

    for epoch in trange(num_epochs, desc="Epoch"):
        t = timer()

        # evaluate unsupervised cost weight
        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)

        if (epoch + 1) % 10 == 0:
            print(f"unsupervised loss weight : {w}")

        # turn it into a usable pytorch object
        w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

        l = []
        supl = []
        unsupl = []
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda(), requires_grad=False)

            # get output and calculate loss
            optimizer.zero_grad()
            out = model(images)
            zcomp = Variable(
                z[i * batch_size : (i + 1) * batch_size], requires_grad=False
            )
            loss, supervised_loss, unsupervised_loss, nbsup = temporal_loss(
                out, zcomp, w, labels
            )

            # save outputs and losses
            outputs[i * batch_size : (i + 1) * batch_size] = out.data.clone()
            l.append(loss.item())
            supl.append(nbsup * supervised_loss.item())
            unsupl.append(unsupervised_loss.item())

            # backprop
            loss.backward()
            optimizer.step()

            # print loss
            if (epoch + 1) % 10 == 0:
                if i + 1 == 2 * c:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i + 1}/{len(train_dataset) // batch_size}], Loss: {np.mean(l):.6f}, Time (this epoch): {timer() - t} s"
                    )
                elif (i + 1) % c == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i + 1}/{len(train_dataset) // batch_size}], Loss: {np.mean(l):.6f}"
                    )

        # update temporal ensemble
        Z = alpha * Z + (1.0 - alpha) * outputs
        z = Z * (1.0 / (1.0 - alpha ** (epoch + 1)))

        # handle metrics, losses, etc.
        eloss = np.mean(l)
        losses.append(eloss)
        sup_losses.append(
            (1.0 / k) * np.sum(supl)
        )  # division by 1/k to obtain the mean supervised loss
        unsup_losses.append(np.mean(unsupl))

        # saving model
        if eloss < best_loss:
            best_loss = eloss
            torch.save({"state_dict": model.state_dict()}, "model_best.pth.tar")

    # test
    model.eval()
    acc = calc_metrics(model, test_loader)
    if print_res:
        print(f"Accuracy of the network on the 10000 test images: {acc:.2f}")

    # test best model
    checkpoint = torch.load("model_best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    acc_best = calc_metrics(model, test_loader)
    if print_res:
        print(
            f"Accuracy of the network (best model) on the 10000 test images: {acc_best:.2f}"
        )

    return acc, acc_best, losses, sup_losses, unsup_losses, indices
