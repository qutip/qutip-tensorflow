import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt

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

def run_benchmarks():
    pass


if __name__ == '__main__':
    filepath = ".benchmarks/Linux-CPython-3.8-64bit/0001_591960324f222f0d7dd1c471203d2347b8f28b52_20210617_160553_uncommited-changes.json"
    df_save_path = ".benchmarks/Linux-CPython-3.8-64bit/latest.csv"
    figures_save_path = ".benchmarks/figures/"

    run_benchmarks()
    benchmark_to_datafrane(filepath, df_save_path)
    plot_benchmark(df_save_path, figures_save_path)
