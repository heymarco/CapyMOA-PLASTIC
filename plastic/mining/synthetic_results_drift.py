import os

import numpy as np
import pandas as pd
from critdd import Diagram

from plastic.mining.rw_results_unshuffled import add_average
from util import get_palette, get_approaches_synthetic

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
matplotlib.rc('font', family='serif')


def add_diff_column(df: pd.DataFrame, column_to_differentiate) -> pd.Series:
    s = df[column_to_differentiate].diff()
    return s


if __name__ == '__main__':
    METRIC = "classifications correct (percent)"
    DATASET_COL = "Dataset"
    APPROACH_COL = "Approach"
    ORDER_COL = "Order"
    approach_order = {key: i for i, key in enumerate(get_approaches_synthetic())}

    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "artificial_drift")
    for dataset_name in os.listdir(main_result_dir):
        dataset = os.path.splitext(dataset_name)[0].split("_")[0]
        filepath = os.path.join(main_result_dir, dataset_name)
        if dataset == "Mixed" or "LED" in dataset:
            continue
        this_df = pd.read_parquet(filepath)
        this_df[DATASET_COL] = dataset

        this_df = this_df.loc[this_df[APPROACH_COL] != "APLASTIC"]
        this_df[ORDER_COL] = this_df[APPROACH_COL].apply(lambda x: approach_order[x])
        this_df["Drift"] = this_df[DATASET_COL].apply(lambda x: "gradual" if x.split("-")[1] == "g" else "abrupt")
        this_df[DATASET_COL] = this_df[DATASET_COL].apply(lambda x: x.split("-")[0])

        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    dataframe = dataframe[dataframe[DATASET_COL] != "STAGGER"]

    metric_to_plot = "classifications correct (percent)"
    plot_relevant_data = dataframe.copy()
    plot_relevant_data.rename({
        "classified instances": "\# instances",
        "classifications correct (percent)": "Accuracy [\%]"
    }, axis=1, inplace=True)

    grid = sns.relplot(plot_relevant_data, x="\# instances", y="Accuracy [\%]", hue=APPROACH_COL,
                       col="Dataset", row="Drift", kind="line", errorbar=None,
                       facet_kws={"sharex": True, "sharey": False}, lw=0.7, palette=get_palette())

    for i, ax in enumerate(grid.axes.flatten()):
        title = ax.get_title()
        a, b = title.split(" | ")
        drift_type = a.split(" = ")[-1]
        dataset = b.split(" = ")[-1]
        ax.set_ylabel(drift_type if i % 4 == 0 else "")
        ax.set_title(dataset if i < 4 else "")
        if dataset == "Hyperplane" and drift_type == "abrupt":
            ax.set_ylim(bottom=65)
        # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.gcf().supylabel("Accuracy [\%]")

    sns.move_legend(grid, "upper center", title="", ncols=5)
    plt.gcf().set_size_inches(6, 3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.15, right=0.97, left=0.13, wspace=.3)

    plt.savefig(os.path.join(os.getcwd(), "figures", "synth_data_acc_over_time_drift.pdf"))
    plt.show()

    # grid = plot(dataframe, x="classified instances", y="Number of leaves")
    # plt.show()

    def add_avg_rank(df: pd.DataFrame):
        ranks = Diagram(
            df[get_approaches_synthetic()].dropna(axis=0).to_numpy(),
            treatment_names=get_approaches_synthetic(),
            maximize_outcome=True
        ).average_ranks
        additional_df = pd.DataFrame(data=[list(ranks)], columns=get_approaches_synthetic()).round(2)
        return additional_df.style.relabel_index(["Rank"])

    # GET AVERAGE ACCURACY
    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC, ORDER_COL]]
                      .groupby([DATASET_COL, ORDER_COL, APPROACH_COL])
                      .mean()
                      # .drop(ORDER_COL, axis=1)
                      )
    average_acc_df.index = average_acc_df.index.droplevel(1)
    pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)
    pivot = pivot[get_approaches_synthetic()]

    # GET STANDARD DEVIATION
    std_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC, ORDER_COL]]
              .groupby([DATASET_COL, ORDER_COL, APPROACH_COL])
              .std())
    std_df.index = std_df.index.droplevel(1)
    std_pivot = std_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)
    std_pivot = std_pivot[get_approaches_synthetic()]

    # GET AVERAGE RUNTIME
    RT_METRIC = "Wallclock time [s]"
    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, RT_METRIC, ORDER_COL]]
                      .groupby([DATASET_COL, ORDER_COL, APPROACH_COL])
                      .mean())
    average_acc_df.index = average_acc_df.index.droplevel(1)
    rt_pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=RT_METRIC)
    rt_pivot = rt_pivot[get_approaches_synthetic()]

    print((pivot
           .round(1)
           .style
           .format(precision=1).highlight_max(axis=1, props="font-weight: bold;")
           .concat(add_average(pivot.dropna(axis=0), name="Accuracy", precision=1).highlight_max(axis=1, props="font-weight: bold;", subset=get_approaches_synthetic()).format(precision=1))
           .concat(add_average(std_pivot.dropna(axis=0), name="Standard dev.", precision=2).highlight_min(axis=1, props="font-weight: bold;",subset=get_approaches_synthetic()).format(precision=2))
           .concat(add_avg_rank(pivot.dropna(axis=0)).highlight_min(axis=1, props="font-weight: bold;", subset=get_approaches_synthetic()).format(precision=1))
           .concat(add_average(rt_pivot.dropna(axis=0), name="Runtime", precision=2).highlight_min(axis=1, props="font-weight: bold;", subset=get_approaches_synthetic()).format(precision=2))
           .to_latex(convert_css=True)))
    print(std_pivot.mean(axis=0))

    print((pivot
           .round(2)
           .style.format(precision=2).highlight_min(axis=1, props="font-weight: bold;")
           .concat(
        add_average(pivot, use_gmean=True).format(precision=2).highlight_min(axis=1, props="font-weight: bold;"))
           .to_latex(convert_css=True)))
