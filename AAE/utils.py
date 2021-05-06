import torch
from sklearn.metrics import adjusted_rand_score


def set_dict_attr(obj, value, attrname, phases, type):
    if isinstance(value, dict):
        setattr(obj, attrname, value)
    elif isinstance(value, type):
        newvalue = {k: value for k in phases}
        setattr(obj, attrname, newvalue)
    else:
        raise ValueError


class Metric:

    """ losses """

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


class Accuracy:

    def __init__(self):
        self._correct = 0
        self._count = 0

    def add(self, pred, target):
        pred = pred.argmax(dim=1)
        correct = (target == pred).sum().item()
        self._correct += correct
        self._count += pred.size(0)

    def value(self):
        return self._correct / self._count


class AdjRand:

    def __init__(self):
        self._preds = []
        self._targets = []

    def add(self, pred, target):
        self._preds.append(pred.argmax(dim=1))
        self._targets.append(target)

    def value(self):
        if hasattr(self, "_value"):
            return self._value
        preds = torch.cat(self._preds).detach().cpu().numpy()
        targets = torch.cat(self._targets).detach().cpu().numpy()
        self._value = adjusted_rand_score(targets, preds)
        return self._value
