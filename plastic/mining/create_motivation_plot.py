import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from util import get_palette

matplotlib.use('TkAgg')
import seaborn as sns

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{helvet}'
matplotlib.rc('font', family='sans-serif')


def plot(df: pd.DataFrame, x: str, y: str, ylabel="Accuracy [\%]") -> sns.FacetGrid:
    g = sns.relplot(df, x=x, y=y, kind="line", row="Dataset", hue="Approach",
                       facet_kws={"sharex": False}, lw=0.7, palette=get_palette())
    sns.move_legend(g, "upper center", title="", ncols=4)
    for ax in g.axes.flatten():
        ax.set_title("")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("\# instances")
    plt.tight_layout()
    plt.gcf().set_size_inches(5, 1.5)
    plt.subplots_adjust(top=0.795, bottom=0.28, right=0.987, left=0.12, wspace=.2, hspace=.5)


if __name__ == '__main__':
    rw_results_dir = os.path.join(os.getcwd(), "results", "motivation")
    plotted_datasets = [
        "RandomTree"
    ]

    all_dfs = []
    for dataset in plotted_datasets:
        data_path = os.path.join(rw_results_dir, dataset + ".parquet")
        df = pd.read_parquet(data_path)
        df["Dataset"] = dataset
        all_dfs.append(df)
        del df

    dataframe = pd.concat(all_dfs, ignore_index=True)
    # dataframe = dataframe[dataframe["Approach"] != "PLASTIC"]
    grid = plot(dataframe, x="classified instances", y="classifications correct (percent)")
    plt.savefig(os.path.join(os.getcwd(), "figures", "motivation.pdf"))
    plt.show()

    # grid = plot(dataframe, x="classified instances", y="Number of leaves")
    # plt.show()


