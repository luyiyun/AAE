import os
from datetime import datetime
import json
from argparse import ArgumentParser
from math import ceil

import torchvision.datasets as D
import torchvision.transforms as T
import torch.utils.data as data
import torch
from torchvision.utils import save_image
import numpy as np


from AAE.models import (
    AAE,
    SuperviseAAE,
    SemiSuperviseAAE
)
from AAE.train import (
    trainAAE,
    trainSuperviseAAE,
    trainSemiSuperviseAAE
)


def load_mnist(conf, test=False, semi=None):
    # dataset
    transfers = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    dat = D.MNIST(conf.data_path, train=True,
                  download=True, transform=transfers)
    loader = data.DataLoader(
        dat, batch_size=conf.bs, shuffle=True,
        pin_memory=True, num_workers=conf.nw
    )
    res = [loader]

    if semi:
        index = np.random.choice(len(dat), size=(semi,), replace=False)
        labeled = data.Subset(dat, index)
        lloader = data.DataLoader(
            labeled, batch_size=conf.bs,
            shuffle=True, pin_memory=True, num_workers=conf.nw
        )
        res.append(lloader)

    if test:
        te_dat = D.MNIST(
            conf.data_path, train=False,
            download=True, transform=transfers
        )
        te_loader = data.DataLoader(
            te_dat, batch_size=conf.bs, shuffle=False,
            pin_memory=True, num_workers=conf.nw
        )
        res.append(te_loader)


    return tuple(res)


def task_normal(conf):
    if isinstance(conf.lrs, float):
        conf.lrs = {k: conf.lrs for k in ["rec", "adv1", "adv2"]}

    # dataset
    loader = load_mnist(conf)

    # model
    net = AAE(
        (1, 28, 28), conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs,
        conf.act, conf.bn, conf.dropout
    )

    # train
    hist = trainAAE(net, loader, conf.epoch, conf.lrs, conf.device)

    # generator
    net.eval()
    samples = net(64, phase="sample")

    # save
    save_path = os.path.join("./results",
                             datetime.now().strftime("normal_%m-%d-%H-%M"))
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    save_image(
        samples, os.path.join(save_path, "sample.png"),
        nrow=8, normalize=True, value_range=(-1, 1)
    )
    with open(os.path.join(save_path, "conf.json"), "w") as f:
        json.dump(conf.__dict__, f)


def task_supervise(conf):
    if isinstance(conf.lrs, float):
        conf.lrs = {k: conf.lrs for k in ["rec", "adv1", "adv2"]}

    # dataset
    loader = load_mnist(conf)

    # model
    net = SuperviseAAE(
        (1, 28, 28), 10, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs,
        conf.act, conf.bn, conf.dropout
    )

    # train
    hist = trainSuperviseAAE(net, loader, conf.epoch, conf.lrs, conf.device)

    # generator
    net.eval()
    y = torch.arange(10).repeat_interleave(8)
    samples = net(y, phase="sample")

    # save
    save_path = os.path.join("./results",
                             datetime.now().strftime("supervise_%m-%d-%H-%M"))
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    save_image(
        samples, os.path.join(save_path, "sample.png"),
        nrow=8, normalize=True, value_range=(-1, 1)
    )
    with open(os.path.join(save_path, "conf.json"), "w") as f:
        json.dump(conf.__dict__, f)


def task_semisupervised(conf):
    phases = ["rec", "adv1", "adv2", "cat_adv1", "cat_adv2", "semi"]

    if isinstance(conf.lrs, float):
        conf.lrs = {k: conf.lrs for k in phases}

    # dataset
    tr_loader, sm_loader, te_loader = load_mnist(
        conf, test=True, semi=conf.Nlabel
    )

    # model
    net = SemiSuperviseAAE(
        (1, 28, 28), 10, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs, conf.cat_disc_hs,
        conf.act, conf.bn, conf.dropout, conf.tau
    )

    # train
    hist = trainSemiSuperviseAAE(net, tr_loader, sm_loader, conf.epoch,
                                 conf.lrs, conf.device, te_loader)

    # generator
    net.eval()
    y = torch.arange(10).repeat_interleave(8)
    samples = net(y, phase="sample")

    # save
    save_path = os.path.join(
        "./results",
        datetime.now().strftime("semi-supervise_%m-%d-%H-%M")
    )
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    save_image(
        samples, os.path.join(save_path, "sample.png"),
        nrow=8, normalize=True, value_range=(-1, 1)
    )
    with open(os.path.join(save_path, "conf.json"), "w") as f:
        json.dump(conf.__dict__, f)


def dict_parse(s, vtype=float):
    if ("," not in s) and ("=" not in s):
        return float(s)
    res = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        res[k] = vtype(v)
    return res


def main():

    # configuration
    parser = ArgumentParser()
    parser.add_argument("--task", default="normal")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--nw", default=4, type=int)
    parser.add_argument("--lrs", default=0.0005, type=dict_parse)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code_dim", default=100, type=int)
    parser.add_argument("--enc_hs", default=[1000, 500], type=int, nargs="*")
    parser.add_argument("--dec_hs", default=[500, 1000], type=int, nargs="*")
    parser.add_argument("--disc_hs", default=[1000, 500], type=int, nargs="*")
    parser.add_argument("--act", default="lrelu", choices=["relu", "lrelu"])
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--dropout", default=0.7, type=dict_parse)
    parser.add_argument("--data_path", default="~/Datasets/")
    parser.add_argument("--Nlabel", default=1000, type=int)
    parser.add_argument("--cat_disc_hs", default=[1000, 500], type=int,
                        nargs="*")
    parser.add_argument("--tau", default=0.01, type=float)
    args = parser.parse_args()

    if args.task == "normal":
        task_normal(args)
    elif args.task == "supervise":
        task_supervise(args)
    elif args.task == "semisupervise":
        task_semisupervised(args)


if __name__ == "__main__":
    main()
