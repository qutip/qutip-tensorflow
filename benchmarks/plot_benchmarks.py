import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import argparse
import glob

def unravel(data, key):
    for d in data:
        values = d.pop(key)
        for k,v in values.items():
            d[key+'_'+k] = v
    return data

def benchmark_to_datafrane(filepath):
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

        return data

def plot_benchmark(df, destination_folder):
    grouped = df.groupby([ 'params_operation', 'params_density'])
    for (operation,density), group in grouped:
        for dtype,g in group.groupby('extra_info_dtype'):
            plt.errorbar(g.params_size, g.stats_mean, g.stats_stddev, fmt='.-', label=dtype)

        plt.title(f"{operation} {density}")
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f".benchmarks/figures/{operation}_density={density}.png")
        plt.close()

def run_benchmarks(args):
    pytest.main(["--benchmark-only",
                 "--benchmark-columns=Mean,StdDev",
                 "--benchmark-sort=name",
                 "--benchmark-autosave"] +
               args)

def get_latest_benchmark_path():
    """Returns the path to the latest benchmark run."""

    benchmark_paths = glob.glob("./.benchmarks/*/*.json")
    dates = [''.join(_b.split("/")[-1].split('_')[2:4])
             for _b in benchmark_paths]
    benchmarks = {date: value for date,value in zip(dates, benchmark_paths)}

    dates.sort()
    latest = dates[-1]
    benchmark_latest = benchmarks[latest]

    return benchmark_latest


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_csv", default=".benchmarks/latest.csv")
    parser.add_argument("--save_plots", default=".benchmarks/figures")
    parser.add_argument("--plot_only", action="store_true")
    args, other_args = parser.parse_known_args()

    if not args.plot_only:
        run_benchmarks(other_args)

    benchmark_latest = get_latest_benchmark_path()
    benchmark_latest = benchmark_to_datafrane(benchmark_latest)

    # Save results as csv
    if args.save_csv:
        benchmark_latest.to_csv(args.save_csv)

    if not args.save_plots:
        plot_benchmark(benchmark_latest, args.save_plots)

