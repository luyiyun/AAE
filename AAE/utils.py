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
