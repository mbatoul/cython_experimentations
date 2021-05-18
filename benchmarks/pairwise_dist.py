import importlib
import time
from pprint import pprint

import numpy as np

from benchmarks.core import N_TRIALS, ONE_GIGABYTE, BaseBenchmark, load


class Benchmark(BaseBenchmark):
    def __init__(self, name, functions, datasets):
        self._functions = functions
        self._datasets = datasets
        super().__init__(name)

    def _run(self):
        for dataset in self._datasets:
            generator = load(dataset["generator"])
            n_samples = int(float(dataset["n_samples"]))
            n_features = int(float(dataset["n_features"]))
            X = generator(n_samples, n_features)
            Y = generator(n_samples, n_features)
            bytes_processed_data = X.nbytes + Y.nbytes
            for function in self._functions:
                implementation = function["implementation"]
                function = load(function["source"])
                times = []
                for _ in range(N_TRIALS):
                    start = time.perf_counter()
                    function(X, Y)
                    end = time.perf_counter()
                    time_elapsed = end - start
                    times.append(time_elapsed)
                time_elapsed = np.mean(times)
                throughput = bytes_processed_data / time_elapsed / ONE_GIGABYTE
                row = dict(
                    implementation=implementation,
                    time_elapsed=time_elapsed,
                    throughput=throughput,
                    n_samples=n_samples,
                    n_features=n_features,
                )
                self.results_ = self.results_.append(row, ignore_index=True)
                pprint(row)
                print("---")
                self._to_csv()
