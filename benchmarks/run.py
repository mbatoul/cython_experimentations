from benchmarks.core import load
import yaml

with open("benchmarks/config.yml", "r") as f:
    config = yaml.full_load(f)

for benchmark, params in config.items():
    bench_class = load(f"benchmarks.{benchmark}.Benchmark")
    bench_instance = bench_class(**params)
    bench_instance.run()
