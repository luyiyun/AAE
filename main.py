import os
from datetime import datetime
import json

import torchvision.datasets as D
import torchvision.transforms as T
import torch.utils.data as data
import torch
from tqdm import tqdm
from torchvision.utils import save_image


from AAE.models import AAE


class Metric:

    def __init__(self):
        self._total = {}
        self._count = 0

    def add(self, losses, bs):
        self._count += bs
        for k, v in losses.items():
            self._total[k] = self._total.get(k, 0.) + v.item() * bs

    def value(self):
        if hasattr(self, "_values"):
            return self._values
        self._values = {k: v / self._count for k, v in self._total.items()}
        return self._values


def train(net, tr_loader, epoch, lrs, device):
    device = torch.device(device)
    net.to(device)
    optimizers = net.optimizers(lrs)

    hist = {}
    for e in tqdm(range(epoch), "Epoch: "):
        metric_cache = Metric()
        net.train()
        for x, y in tqdm(tr_loader, "Train: ", leave=False):
            x = x.to(device)
            losses = {}
            with torch.enable_grad():
                for phase in ["rec", "adv1", "adv2"]:
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


def main():
    # configuration
    epoch = 50
    bs = 256
    nw = 4
    lrs = {"rec": 0.0005, "adv1": 0.0005, "adv2": 0.0005}
    device = "cuda:0"

    code_dim = 100
    enc_hiddens = [1000, 500]
    dec_hiddens = [500, 1000]
    disc_hiddens = [1000, 500]
    act = 'lrelu'
    bn = False
    dropout = 0.7

    # dataset
    transfers = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    dat = D.MNIST("~/Datasets/", train=True,
                  download=True, transform=transfers)
    loader = data.DataLoader(
        dat, batch_size=bs, shuffle=True, pin_memory=True, num_workers=nw)

    # model
    net = AAE(
        (1, 28, 28), code_dim, enc_hiddens, dec_hiddens, disc_hiddens,
        act, bn, dropout
    )

    # train
    hist = train(net, loader, epoch, lrs, device)

    # generator
    net.eval()
    samples = net(64, phase="sample")

    # save
    save_path = os.path.join("./results",
                             datetime.now().strftime("%m-%d-%H-%M"))
    print("savint to %s ..." % save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hist.json"), "w") as f:
        json.dump(hist, f)
    torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
    save_image(
        samples, os.path.join(save_path, "sample.png"),
        nrow=8, normalize=True, value_range=(-1, 1)
    )


if __name__ == "__main__":
    main()
