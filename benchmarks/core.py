import importlib
import pandas as pd

N_TRIALS = 10
ONE_GIGABYTE = 1e9


class BaseBenchmark:
    def __init__(self, name):
        self.name = name

    def _run(self):
        raise NotImplementedError

    def _to_csv(self):
        self.results_.to_csv(
            f"benchmarks/results/{self.name}.csv",
            mode="w+",
            index=False,
        )

    def run(self):
        self.results_ = pd.DataFrame()
        self._run()
        self._to_csv()


def load(path):
    splitted_path = path.split(".")
    module, func_name = ".".join(splitted_path[:-1]), splitted_path[-1]
    return getattr(importlib.import_module(module), func_name)
