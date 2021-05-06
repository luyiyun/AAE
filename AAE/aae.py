from itertools import chain
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .utils import Metric, set_dict_attr
from .base import MLP


class AAE(nn.Module):

    components = ("enc", "dec", "disc")
    phases = ("rec", "adv1", "adv2")

    def __init__(
        self, inc, hidden, enc_hidden, dec_hidden, disc_hidden,
        act="lrelu", bn=False, dropout=None
    ):
        super().__init__()
        self._inc = inc
        self._inc1 = prod(inc)
        self._hidden = hidden
        self._enc_h = enc_hidden
        self._dec_h = dec_hidden
        self._disc_h = disc_hidden
        set_dict_attr(self, act, "_act", self.components, str)
        set_dict_attr(self, bn, "_bn", self.components, bool)
        set_dict_attr(self, dropout, "_dropout", self.components,
                      (float, type(None)))

        self._build_model()

    def _build_model(self):
        self.enc = MLP(
            self._inc1, self._hidden, self._enc_h,
            act=self._act["enc"],
            bn=self._bn["enc"],
            dropout=self._dropout["enc"]
        )
        self.dec = MLP(
            self._hidden, self._inc1, self._dec_h,
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
            )
        }

    @classmethod
    def fit(cls, net, tr_loader, epoch, lr, beta1, device):
        device = torch.device(device)
        net.to(device)
        optimizers = net.optimizers(lr, beta1)

        hist = {}
        for e in tqdm(range(epoch), "Epoch: "):
            metric_cache = Metric()
            net.train()
            for x, y in tqdm(tr_loader, "Train: ", leave=False):
                x = x.to(device)
                losses = {}
                with torch.enable_grad():
                    for phase in cls.phases:
                        optimizers[phase].zero_grad()
                        loss = net(x, phase)
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

        return hist
