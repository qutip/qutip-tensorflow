import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import argparse

def unravel(data, key):
    for d in data:
        values = d.pop(key)
        for k,v in values.items():
            d[key+'_'+k] = v
    return data

def benchmark_to_datafrane(filepath, df_save_path):
    path = pathlib.Path(__file__).absolute().parents[1].joinpath(filepath)
    with open(path) as f:
        data = json.load(f)
        data = data['benchmarks']
        data = unravel(data, 'options')
        data = unravel(data, 'stats')
        data = unravel(data, 'params')
        data = unravel(data, 'extra_info')
        data = pd.DataFrame(data)

        # Set operation properly (for example: matmul instead of:
        # UNSERIALIZABLE[<function Qobj.__mammal__ at 0x...)
        # The name of the operation is obtained from the group name
        data.params_operation = data.group.str.split('-')
        data.params_operation = [d[0] for d in data.params_operation]

        data.to_csv(df_save_path)

        return data

def plot_benchmark(df_path, destination_folder):
    df = pd.read_csv(df_path, index_col=0)
    grouped = df.groupby([ 'params_operation', 'params_density'])
    for (operation,density), group in grouped:
        for dtype,g in group.groupby('extra_info_dtype'):
            plt.errorbar(g.params_size, g.stats_mean, g.stats_stddev, fmt='.-', label=dtype)

        plt.title(operation + str(density))
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f".benchmarks/figures/{operation}_density={density}.png")
        plt.close()

def run_benchmarks(args):
    pytest.main(["--benchmark-only",
                 "--benchmark-columns=Mean,StdDev",
                 "--benchmark-sort=name"] +
               args)


if __name__ == '__main__':
    filepath = ".benchmarks/Linux-CPython-3.8-64bit/0031_f6c63e6c1b95a648627a1cc6382f27c3a0181c78_20210618_142829_uncommited-changes.json"
    df_save_path = ".benchmarks/Linux-CPython-3.8-64bit/latest.csv"
    figures_save_path = ".benchmarks/figures/"

    parser = argparse.ArgumentParser()
    args, other_args = parser.parse_known_args()

    run_benchmarks(other_args)
    benchmark_to_datafrane(filepath, df_save_path)
    plot_benchmark(df_save_path, figures_save_path)
