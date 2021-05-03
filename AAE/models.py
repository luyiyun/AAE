from itertools import chain
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _act(name=None):
    if name is None:
        return nn.Identity()
    else:
        return {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh()
        }[name]


def _fc(inc, outc, bn=True, act=True, dropout=None):
    layers = [nn.Linear(inc, outc)]
    if bn:
        layers.append(nn.BatchNorm1d(outc))
    if act:
        layers.append(_act(act))
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MLP(nn.Module):

    def __init__(
        self, inc, out, hiddens, bn=True, act="relu",
        out_bn=False, out_act=None, dropout=None
    ):
        super().__init__()
        net = []
        for i, o in zip([inc] + hiddens[:-1], hiddens):
            net.append(_fc(i, o, bn=bn, act=act, dropout=dropout))
        net.append(nn.Linear(hiddens[-1], out))
        if out_bn:
            net.append(nn.BatchNorm1d(out))
        if out_act:
            net.append(_act(out_act))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class AAE(nn.Module):

    def __init__(
        self, inc, hidden, enc_hidden, dec_hidden, disc_hidden,
        act="lrelu", bn=False, dropout=None
    ):
        super().__init__()
        self._hidden = hidden
        self._inc = inc
        self._inc1 = prod(inc)
        self.enc = MLP(
            self._inc1, hidden, enc_hidden,
            act=act, bn=bn, dropout=dropout
        )
        self.dec = MLP(
            hidden, self._inc1, dec_hidden,
            act=act, out_act="tanh", bn=bn,
            dropout=dropout
        )
        self.disc = MLP(
            hidden, 1, disc_hidden,
            act=act, bn=bn, dropout=dropout
        )
        self.flatten = nn.Flatten()

    def forward(self, x, phase="rec"):
        if self.training:
            x = self.flatten(x)
            h = self.enc(x)
            if phase == "rec":
                rec = self.dec(h)
                return F.mse_loss(rec, x)
            elif phase == "adv1":
                h_true = torch.randn_like(h)
                p_fake = self.disc(h)
                p_true = self.disc(h_true)
                t_fake = torch.zeros_like(p_fake)
                t_true = torch.ones_like(p_true)
                p = torch.cat([p_fake, p_true])
                t = torch.cat([t_fake, t_true])
                return F.binary_cross_entropy_with_logits(p, t)
            elif phase == "adv2":
                p_fake = self.disc(h)
                t_fake = torch.ones_like(p_fake)
                return F.binary_cross_entropy_with_logits(p_fake, t_fake)
        else:
            if phase == "sample":
                device = next(iter((self.parameters()))).device
                # x is sample size
                x = torch.randn((x, self._hidden), device=device)
                x = self.dec(x)
                return x.reshape(-1, *self._inc)

    def optimizers(self, lrs):
        return {
            "rec": optim.Adam(
                chain(self.enc.parameters(), self.dec.parameters()),
                lrs["rec"]
            ),
            "adv1": optim.Adam(
                self.disc.parameters(), lrs["adv1"],
                # betas=(0.5, 0.999)
            ),
            "adv2": optim.Adam(
                self.enc.parameters(), lrs["adv2"],
                # betas=(0.5, 0.999)
            )
        }
