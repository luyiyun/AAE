import torch
from tqdm import tqdm


from .utils import Metric


def trainAAE(net, tr_loader, epoch, lrs, device):
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
