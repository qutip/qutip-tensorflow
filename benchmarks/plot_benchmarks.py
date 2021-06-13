import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt

# with open('../bench') as f:
    # print(f)

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
        # UNSERIALIZABLE[<function Qobj.__matmul__ at 0x...)
        # The name of the operation is obtained from the group name
        data.params_operation = data.group.str.split('-')
        data.params_operation = [d[0] for d in data.params_operation]

        return data


if __name__ == '__main__':
    filepath = ".benchmarks/Linux-CPython-3.8-64bit/0029_05d4e025622c1975caabbb591b8ff35555cf830e_20210613_200950_uncommited-changes.json"
    df = benchmark_to_datafrane(filepath)

    save_path = filepath[:-4]+'csv'
    df.to_csv(save_path)

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


