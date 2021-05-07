import os
from datetime import datetime
import json
from argparse import ArgumentParser
from functools import partial

import torchvision.datasets as D
import torchvision.transforms as T
import torch.utils.data as data
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from AAE import (
    AAE,
    SuperviseAAE,
    SemiSuperviseAAE,
    UnSuperviseAAE,
    EmbedSuperviseAAE
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
    # dataset
    loader, = load_mnist(conf)

    # model
    net = AAE(
        (1, 28, 28), conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs,
        conf.act, conf.bn, conf.dropout
    )

    # train
    hist = AAE.fit(net, loader, conf.epoch, conf.lr, conf.beta1, conf.device)

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
    # dataset
    loader, = load_mnist(conf)

    # model
    net = SuperviseAAE(
        (1, 28, 28), 10, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs,
        conf.act, conf.bn, conf.dropout
    )

    # train
    hist = SuperviseAAE.fit(net, loader, conf.epoch, conf.lr, conf.beta1,
                            conf.device)

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


def task_semisupervise(conf):
    # dataset
    tr_loader, sm_loader, te_loader = load_mnist(
        conf, test=True, semi=conf.Nlabel
    )

    # model
    net = SemiSuperviseAAE(
        (1, 28, 28), 10, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs, conf.cdisc_hs,
        conf.act, conf.bn, conf.dropout, conf.tau
    )

    # train
    hist = SemiSuperviseAAE.fit(net, tr_loader, sm_loader, conf.epoch,
                                conf.lr, conf.beta1, conf.device, te_loader)

    # generator
    net.eval()
    y = torch.arange(10).repeat_interleave(8)
    samples = net(y, phase="sample")

    # plot
    plt.plot(hist["acc"])
    plt.title("Validation Accuracy")

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
    plt.savefig(os.path.join(save_path, "acc.png"))


def task_unsupervise(conf):
    # dataset
    tr_loader, te_loader = load_mnist(conf, test=True)

    # model
    net = UnSuperviseAAE(
        (1, 28, 28), conf.Ncluster, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs, conf.cdisc_hs,
        conf.act, conf.bn, conf.dropout, conf.tau
    )

    # train
    hist = UnSuperviseAAE.fit(net, tr_loader, conf.epoch, conf.lr, conf.beta1,
                              conf.device, te_loader)

    # generator
    net.eval()
    y = torch.arange(conf.Ncluster).repeat_interleave(10)
    samples = net(y, phase="sample")

    # plot
    plt.plot(hist["rand"])
    plt.title("Validation Adjusted Rand Score")

    # save
    save_path = os.path.join(
        "./results",
        datetime.now().strftime("unsupervise_%m-%d-%H-%M")
    )
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    save_image(
        samples, os.path.join(save_path, "sample.png"),
        nrow=10, normalize=True, value_range=(-1, 1)
    )
    with open(os.path.join(save_path, "conf.json"), "w") as f:
        json.dump(conf.__dict__, f)
    plt.savefig(os.path.join(save_path, "rand.png"))


def task_embed(conf):
    # dataset
    tr_loader, te_loader = load_mnist(conf, test=True)

    # model
    net = EmbedSuperviseAAE(
        (1, 28, 28), conf.Ncluster, conf.code_dim,
        conf.enc_hs, conf.dec_hs, conf.disc_hs, conf.cdisc_hs,
        conf.act, conf.bn, conf.dropout, conf.tau
    )

    # train
    hist = EmbedSuperviseAAE.fit(net, tr_loader, conf.epoch, conf.lr,
                                 conf.beta1, conf.device)

    # transform
    code, y = EmbedSuperviseAAE.transform(net, te_loader, conf.device)

    # plot
    df = pd.DataFrame({"Z1": code[:, 0], "Z2": code[:, 1], "class": y})
    df["class"] = df["class"].astype("category")
    fig = sns.relplot(data=df, x="Z1", y="Z2", hue="class")
    fig.set_titles("Validation Embed")

    # save
    save_path = os.path.join(
        "./results", datetime.now().strftime("embed_%m-%d-%H-%M")
    )
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    with open(os.path.join(save_path, "conf.json"), "w") as f:
        json.dump(conf.__dict__, f)
    fig.savefig(os.path.join(save_path, "embed.png"))


def dict_parse(s, vtype=float):
    if ("," not in s) and ("=" not in s):
        return vtype(s)
    res = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        res[k] = vtype(v)
    return res


def fixed_bool(s):
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError


def main():

    # configuration
    parser = ArgumentParser()
    parser.add_argument("--task", default="normal")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--nw", default=4, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code_dim", default=100, type=int)
    parser.add_argument("--enc_hs", default=[1000, 500], type=int, nargs="*")
    parser.add_argument("--dec_hs", default=[500, 1000], type=int, nargs="*")
    parser.add_argument("--disc_hs", default=[1000, 500], type=int, nargs="*")
    parser.add_argument("--data_path", default="~/Datasets/")

    # dict parsed arguments
    parser.add_argument("--lr", default=0.0005, type=dict_parse)
    parser.add_argument("--bn", default="false",
                        type=partial(dict_parse, vtype=fixed_bool))
    parser.add_argument("--act", default="lrelu",
                        type=partial(dict_parse, vtype=str))
    parser.add_argument("--dropout", default=0.5, type=dict_parse)
    parser.add_argument("--beta1", default=0.9, type=dict_parse)

    # semi-supervised learning
    parser.add_argument("--Nlabel", default=1000, type=int)
    parser.add_argument("--cdisc_hs", default=[1000, 500], type=int,
                        nargs="*")
    parser.add_argument("--tau", default=0.05, type=float)

    # unsupervised leanring
    parser.add_argument("--Ncluster", default=10, type=int)

    args = parser.parse_args()

    if args.task == "normal":
        task_normal(args)
    elif args.task == "supervise":
        task_supervise(args)
    elif args.task == "semisupervise":
        task_semisupervise(args)
    elif args.task == "unsupervise":
        task_unsupervise(args)
    elif args.task == "embed":
        task_embed(args)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
