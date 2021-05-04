from copy import deepcopy

import torch
from tqdm import tqdm

from .utils import Metric, Accuracy


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

    return hist


def trainSuperviseAAE(net, tr_loader, epoch, lrs, device):
    device = torch.device(device)
    net.to(device)
    optimizers = net.optimizers(lrs)

    hist = {}
    for e in tqdm(range(epoch), "Epoch: "):
        metric_cache = Metric()
        net.train()
        for x, y in tqdm(tr_loader, "Train: ", leave=False):
            x = x.to(device)
            y = y.to(device)
            losses = {}
            with torch.enable_grad():
                for phase in ["rec", "adv1", "adv2"]:
                    optimizers[phase].zero_grad()
                    loss = net(x, y, phase)  # 唯一改动
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


def trainSemiSuperviseAAE(
    net, tr_loader, sm_loader, epoch, lrs, device, va_loader=None
):
    device = torch.device(device)
    net.to(device)
    optimizers = net.optimizers(lrs)
    phases = ["semi", "rec", "adv1", "adv2", "cat_adv1", "cat_adv2"]

    if va_loader is not None:
        best = {"epoch": -1, "acc": -1000, "model": deepcopy(net.state_dict())}

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
                for phase in phases:
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
