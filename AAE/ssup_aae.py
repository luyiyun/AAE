from itertools import chain
from math import prod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import OneHotCategorical

from .utils import Metric, Accuracy, set_dict_attr
from .base import MLP


class SemiSuperviseAAE(nn.Module):

    components = ("enc", "dec", "disc", "cdisc")
    phases = ("rec", "adv1", "adv2", "cadv1", "cadv2", "semi")

    def __init__(
        self, inc, out, hidden, enc_hidden, dec_hidden,
        disc_hidden, cdisc_hidden, act="lrelu", bn=False, dropout=None,
        tau=0.01, prior=None
    ):
        super().__init__()
        self._inc = inc
        self._inc1 = prod(inc)
        self._out = out
        self._hidden = hidden
        self._enc_h = enc_hidden
        self._dec_h = dec_hidden
        self._disc_h = disc_hidden
        self._cdisc_h = cdisc_hidden
        self._tau = tau
        self._prior = prior
        set_dict_attr(self, act, "_act", self.components, str)
        set_dict_attr(self, bn, "_bn", self.components, bool)
        set_dict_attr(self, dropout, "_dropout", self.components,
                      (float, type(None)))

        self._build_model()

    def _build_model(self):
        self.enc = MLP(
            self._inc1, self._hidden+self._out, self._enc_h,
            act=self._act["enc"],
            bn=self._bn["enc"],
            dropout=self._dropout["enc"]
        )
        self.dec = MLP(
            self._hidden+self._out, self._inc1, self._dec_h,
            act=self._act["dec"],
            bn=self._bn["dec"],
            dropout=self._dropout["dec"],
            out_act="tanh",
        )
        self.disc = MLP(
            self._hidden, 1, self._disc_h,
            act=self._act["disc"],
            bn=self._bn["disc"],
            dropout=self._dropout["disc"],
        )
        self.cdisc = MLP(
            self._out, 1, self._cdisc_h,
            act=self._act["cdisc"],
            bn=self._bn["cdisc"],
            dropout=self._dropout["cdisc"],
        )
        self.flatten = nn.Flatten()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.iden = torch.eye(self._out, dtype=torch.float)
        if self._prior is None:
            prior = torch.ones(self._out)
        else:
            prior = torch.tensor(self._prior)
        self.cdist = OneHotCategorical(prior)

    def forward(self, x, y=None, phase="rec"):
        if self.training:
            x = self.flatten(x)
            h = self.enc(x)
            hz = h[:, :self._hidden]
            hc = h[:, self._hidden:]
            if phase in ["rec", "cadv1", "cadv2"]:
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
            elif phase == "cadv1":
                h_true = self.cdist.sample((hc_sample.size(0),)).to(x)
                p_fake = self.cdisc(hc_sample)
                p_true = self.cdisc(h_true)
                t_fake = torch.zeros_like(p_fake)
                t_true = torch.ones_like(p_true)
                p = torch.cat([p_fake, p_true])
                t = torch.cat([t_fake, t_true])
                return F.binary_cross_entropy_with_logits(p, t)
            elif phase == "cadv2":
                p_fake = self.cdisc(hc_sample)
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

    def optimizers(self, lr, beta1):
        set_dict_attr(self, lr, "_lr", self.phases, float)
        set_dict_attr(self, beta1, "_beta1", self.phases, float)
        return {
            "rec": optim.Adam(
                chain(self.enc.parameters(), self.dec.parameters()),
                lr=self._lr["rec"],
                betas=(self._beta1["rec"], 0.999)
            ),
            "adv1": optim.Adam(
                self.disc.parameters(),
                lr=self._lr["adv1"],
                betas=(self._beta1["adv1"], 0.999)
            ),
            "adv2": optim.Adam(
                self.enc.parameters(),
                lr=self._lr["adv2"],
                betas=(self._beta1["adv2"], 0.999)
            ),
            "cadv1": optim.Adam(
                self.cdisc.parameters(),
                lr=self._lr["cadv1"],
                betas=(self._beta1["cadv1"], 0.999)
            ),
            "cadv2": optim.Adam(
                self.enc.parameters(),
                lr=self._lr["cadv2"],
                betas=(self._beta1["cadv2"], 0.999)
            ),
            "semi": optim.Adam(
                self.enc.parameters(),
                lr=self._lr["semi"],
                betas=(self._beta1["semi"], 0.999)
            )
        }

    def _gumbel_softmax_sample(self, loghc):
        u = torch.rand_like(loghc)
        u = -(-u.log()).log()
        return F.softmax((loghc + u) / self._tau, dim=1)

    @classmethod
    def fit(
        cls, net, tr_loader, sm_loader, epoch, lr, beta1, device,
        va_loader=None
    ):
        device = torch.device(device)
        net.to(device)
        optimizers = net.optimizers(lr, beta1)

        if va_loader is not None:
            best = {"epoch": -1,
                    "acc": -1000,
                    "model": deepcopy(net.state_dict())}

        hist = {}
        sm_iter = iter(sm_loader)
        for e in tqdm(range(epoch), "Epoch: "):
            # train
            metric_cache = Metric()
            net.train()
            for x, _ in tqdm(tr_loader, "Train: ", leave=False):
                # 保证labeled datasets源源不绝
                try:
                    xl, yl = next(sm_iter)
                except StopIteration:
                    sm_iter = iter(sm_loader)
                    xl, yl = next(sm_iter)
                x = x.to(device)
                xl = xl.to(device)
                yl = yl.to(device)
                losses = {}
                with torch.enable_grad():
                    for phase in cls.phases:
                        optimizers[phase].zero_grad()
                        if phase == "semi":
                            loss = net(xl, yl, phase)
                        else:
                            loss = net(x, None, phase)
                        loss.backward()
                        optimizers[phase].step()
                        losses[phase] = loss
                metric_cache.add(losses, x.size(0))
            metric_values = metric_cache.value()
            for k, v in metric_values.items():
                hist.setdefault(k, []).append(v)
            tqdm.write(
                "Train Epoch: %d, " % e + ", ".join([
                    "%s: %.4f" % (k, v) for k, v in metric_values.items()
                ])
            )

            # valid
            if va_loader is not None:
                acc_cache = Accuracy()
                net.eval()
                for x, y in tqdm(va_loader, "Valid: ", leave=False):
                    x = x.to(device)
                    y = y.to(device)
                    with torch.no_grad():
                        pred = net(x, None, "pred")
                        acc_cache.add(pred, y)
                acc = acc_cache.value()
                hist.setdefault("acc", []).append(acc)
                tqdm.write("Valid Epoch: %d, ACC: %.4f" % (e, acc))

                if best["acc"] < acc:
                    best["epoch"] = e
                    best["acc"] = acc
                    best["model"] = deepcopy(net.state_dict())

        if va_loader is not None:
            print("The best epoch: %d, best acc: %.4f" %
                  (best["epoch"], best["acc"]))
            net.load_state_dict(best["model"])

        return hist
