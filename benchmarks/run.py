import json

import joblib
import yaml
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info
from threadpoolctl import threadpool_info

from benchmarks.core import load

with open("benchmarks/config.yml", "r") as f:
    config = yaml.full_load(f)

for benchmark, params in config.items():
    bench_class = load(f"benchmarks.{benchmark}.Benchmark")
    bench_instance = bench_class(**params)
    bench_instance.run()

env_info = {}
env_info["system_info"] = _get_sys_info()
env_info["dependencies_info"] = _get_deps_info()
env_info["threadpool_info"] = threadpool_info()
env_info["cpu_count"] = joblib.cpu_count(only_physical_cores=True)

with open("benchmarks/env_info.txt", "w") as f:
    json.dump(env_info, f)
