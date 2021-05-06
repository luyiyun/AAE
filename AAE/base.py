import torch.nn as nn


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
