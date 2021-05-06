from itertools import chain
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import OneHotCategorical


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
        if dropout is None or isinstance(dropout, float):
            dropout = {k: dropout for k in ["enc", "dec", "disc"]}
        self.enc = MLP(
            self._inc1, hidden, enc_hidden,
            act=act, bn=bn, dropout=dropout["enc"]
        )
        self.dec = MLP(
            hidden, self._inc1, dec_hidden,
            act=act, out_act="tanh", bn=bn,
            dropout=dropout["dec"]
        )
        self.disc = MLP(
            hidden, 1, disc_hidden,
            act=act, bn=bn, dropout=dropout["disc"]
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


class SuperviseAAE(nn.Module):

    def __init__(
        self, inc, out, hidden, enc_hidden, dec_hidden, disc_hidden,
        act="lrelu", bn=False, dropout=None
    ):
        super().__init__()
        self._hidden = hidden
        self._inc = inc
        self._inc1 = prod(inc)
        if dropout is None or isinstance(dropout, float):
            dropout = {k: dropout for k in ["enc", "dec", "disc"]}
        self.enc = MLP(
            self._inc1, hidden, enc_hidden,
            act=act, bn=bn, dropout=dropout["enc"]
        )
        self.dec = MLP(
            hidden+out, self._inc1, dec_hidden,
            act=act, out_act="tanh", bn=bn,
            dropout=dropout["dec"]
        )
        self.disc = MLP(
            hidden, 1, disc_hidden,
            act=act, bn=bn, dropout=dropout["disc"]
        )
        self.flatten = nn.Flatten()

        self.iden = torch.eye(out, dtype=torch.float)

    def forward(self, x, y=None, phase="rec"):
        if self.training:
            x = self.flatten(x)
            h = self.enc(x)
            if phase == "rec":
                y_oh = self.iden.to(x)[y]
                rec = self.dec(torch.cat([h, y_oh], 1))
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
                # x is labels
                labels = self.iden[x].to(device)
                hiddens = torch.randn((x.size(0), self._hidden), device=device)
                x = self.dec(torch.cat([hiddens, labels], 1))
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


class SemiSuperviseAAE(nn.Module):

    def __init__(
        self, inc, out, hidden, enc_hidden, dec_hidden,
        disc_hidden, cat_disc_hidden, act="lrelu", bn=False, dropout=None,
        tau=0.01, prior=None
    ):
        super().__init__()
        self._hidden = hidden
        self._inc = inc
        self._tau = tau
        self._inc1 = prod(inc)
        if dropout is None or isinstance(dropout, float):
            dropout = {k: dropout for k in ["enc", "dec", "disc", "cat_disc"]}
        self.enc = MLP(
            self._inc1, hidden+out, enc_hidden,
            act=act, bn=bn, dropout=dropout["enc"]
        )
        self.dec = MLP(
            hidden+out, self._inc1, dec_hidden,
            act=act, out_act="tanh", bn=bn,
            dropout=dropout["dec"]
        )
        self.disc = MLP(
            hidden, 1, disc_hidden,
            act=act, bn=bn, dropout=dropout["disc"]
        )
        self.cat_disc = MLP(
            out, 1, cat_disc_hidden,
            act=act, bn=bn, dropout=dropout["cat_disc"]
        )
        self.flatten = nn.Flatten()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.iden = torch.eye(out, dtype=torch.float)
        prior = torch.ones(out) if prior is None else prior
        self.cat_dist = OneHotCategorical(prior)

    def forward(self, x, y=None, phase="rec"):
        if self.training:
            x = self.flatten(x)
            h = self.enc(x)
            hz = h[:, :self._hidden]
            hc = h[:, self._hidden:]
            if phase in ["rec", "cat_adv1", "cat_adv2"]:
                loghc = self.logsoftmax(hc)
                hc_sample = self._gumbel_softmax_sample(loghc)
            if phase == "rec":
                rec = self.dec(torch.cat([hz, hc_sample], 1))
                return F.mse_loss(rec, x)
            elif phase == "adv1":
                h_true = torch.randn_like(hz)
                p_fake = self.disc(hz)
                p_true = self.disc(h_true)
                t_fake = torch.zeros_like(p_fake)
                t_true = torch.ones_like(p_true)
                p = torch.cat([p_fake, p_true])
                t = torch.cat([t_fake, t_true])
                return F.binary_cross_entropy_with_logits(p, t)
            elif phase == "adv2":
                p_fake = self.disc(hz)
                t_fake = torch.ones_like(p_fake)
                return F.binary_cross_entropy_with_logits(p_fake, t_fake)
            elif phase == "cat_adv1":
                h_true = self.cat_dist.sample((hc_sample.size(0),)).to(x)
                p_fake = self.cat_disc(hc_sample)
                p_true = self.cat_disc(h_true)
                t_fake = torch.zeros_like(p_fake)
                t_true = torch.ones_like(p_true)
                p = torch.cat([p_fake, p_true])
                t = torch.cat([t_fake, t_true])
                return F.binary_cross_entropy_with_logits(p, t)
            elif phase == "cat_adv2":
                p_fake = self.cat_disc(hc_sample)
                t_fake = torch.ones_like(p_fake)
                return F.binary_cross_entropy_with_logits(p_fake, t_fake)
            elif phase == "semi":
                return F.cross_entropy(hc, y)
        else:
            if phase == "sample":
                device = next(iter((self.parameters()))).device
                # x is labels
                labels = self.iden[x].to(device)
                hiddens = torch.randn((x.size(0), self._hidden), device=device)
                x = self.dec(torch.cat([hiddens, labels], 1))
                return x.reshape(-1, *self._inc)
            elif phase == "pred":
                x = self.flatten(x)
                h = self.enc(x)
                return h[:, self._hidden:]

    def optimizers(self, lrs):
        return {
            "rec": optim.Adam(
                chain(self.enc.parameters(), self.dec.parameters()),
                lrs["rec"]
            ),
            "adv1": optim.Adam(self.disc.parameters(), lrs["adv1"]),
            "adv2": optim.Adam(self.enc.parameters(), lrs["adv2"]),
            "cat_adv1": optim.Adam(self.cat_disc.parameters(),
                                   lrs["cat_adv1"]),
            "cat_adv2": optim.Adam(self.enc.parameters(), lrs["cat_adv2"]),
            "semi": optim.Adam(self.enc.parameters(), lrs["semi"])
        }

    def _gumbel_softmax_sample(self, loghc):
        u = torch.rand_like(loghc)
        u = -(-u.log()).log()
        return F.softmax((loghc + u) / self._tau, dim=1)
