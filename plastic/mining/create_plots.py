import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from util import get_palette

matplotlib.use('TkAgg')
import seaborn as sns

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
matplotlib.rc('font', family='serif')


def plot(df: pd.DataFrame, x: str, y: str, errorbar=None, palette=None, col_wrap=2, figheight=5.0, sharex=False) -> sns.FacetGrid:
    g = sns.relplot(df, x=x, y=y, kind="line", col="Dataset", col_wrap=col_wrap, hue="Approach", errorbar=errorbar,
                       facet_kws={"sharex": sharex, "sharey": False}, lw=0.7, palette=palette)
    sns.move_legend(g, "upper center", title="", ncols=4)
    plt.tight_layout()
    plt.gcf().set_size_inches(9, figheight)
    plt.subplots_adjust(top=0.914, bottom=0.07, right=0.97, left=0.093, wspace=.2, hspace=.717)
    return g


if __name__ == '__main__':
    METRIC = "classifications correct (percent)"
    DATASET_COL = "Dataset"
    APPROACH_COL = "Approach"
    ORDER_COL = "Order"

    rw_results_dir = os.path.join(os.getcwd(), "results", "hopefully_final")

    all_dfs = []
    for file in os.listdir(rw_results_dir):
        dataset = file.split("_")[0]
        if dataset == "SUSY" or dataset == "POKER" or dataset == "FONTS":
            continue
        data_path = os.path.join(rw_results_dir, file, "unshuffled", dataset + ".parquet")
        df = pd.read_parquet(data_path)
        df["Dataset"] = dataset
        if dataset == "HIGGS" or dataset == "HEPMASS":
            df = df.iloc[::20]
        all_dfs.append(df)
        del df

    dataframe = pd.concat(all_dfs, ignore_index=True)
    dataframe = dataframe[dataframe["Approach"] != "HT"]

    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "PLASTIC"]
    # dataframe[APPROACH_COL][dataframe[APPROACH_COL] == "PLASTIC*"] = "PLASTIC"
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "EFDT"]
    # dataframe[APPROACH_COL][dataframe[APPROACH_COL] == "EFDT*"] = "EFDT"

    metric_to_plot = "classifications correct (percent)"
    diff_df_efdt = dataframe[dataframe["Approach"] == "EFDT"]
    diff_df_efdt[metric_to_plot] = dataframe[dataframe["Approach"] == "PLASTIC"][metric_to_plot].to_numpy() - diff_df_efdt[metric_to_plot].to_numpy()
    df = diff_df_efdt
    df["Approach"] = "PLASTIC - EFDT"
    df.rename({"classified instances": "\# instances"}, axis=1, inplace=True)
    pal = get_palette()
    pal["PLASTIC - EFDT"] = pal["PLASTIC-A"]
    grid = plot(df, x="\# instances", y=metric_to_plot,
                palette=pal,
                col_wrap=3)

    hlines = [
        [-10, -5, 5],
        [-12.5, 12.5, 25],
        [-50, 50, 100],
        [-50, 50, 100],
        [-1.25, 1.25, 2.5],
        [-5, 5],
        [-2.5, 2.5],
        [-12.5, 12.5, 25],
        [-50, 50],
        [10, 20],
        [25, 50],
        [-33.34, 33.33, 66.66, 100],
    ]

    for i, ax in enumerate(grid.axes.flatten()):
        ax.set_title(ax.get_title().split(" = ")[-1])
        ax.set_ylabel("")
        ax.axhline(0.0, color="black", zorder=0, lw=0.7)
        [ax.axhline(h, color="black", ls="--", zorder=0, lw=0.7, alpha=0.3) for h in hlines[i]]
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.gcf().supylabel("Difference in Accuracy (PLASTIC - EFDT) [\%]")
    grid.legend.remove()
    plt.tight_layout(pad=.5)
    plt.savefig(os.path.join(os.getcwd(), "figures", "difference_in_accuracy_rw_data.pdf"))
    plt.show()
