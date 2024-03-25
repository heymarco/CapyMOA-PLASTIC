import os

import numpy as np
import pandas as pd
from critdd import Diagram
from matplotlib import pyplot as plt

from plastic.mining.create_plots import plot
from plastic.mining.rw_results_unshuffled import add_average
from util import get_palette, get_approaches_synthetic
import seaborn as sns


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
    main_result_dir = os.path.join(os.getcwd(), "results", "artificial")
    for dataset_name in os.listdir(main_result_dir):
        dataset = os.path.splitext(dataset_name)[0].split("_")[0]
        filepath = os.path.join(main_result_dir, dataset_name)
        if dataset == "Mixed":
            continue
        this_df = pd.read_parquet(filepath)
        this_df[DATASET_COL] = dataset

        # this_df = this_df.loc[this_df[APPROACH_COL] != "PLASTIC"]
        # this_df[APPROACH_COL][this_df[APPROACH_COL] == "PLASTIC*"] = "PLASTIC"
        # this_df = this_df.loc[this_df[APPROACH_COL] != "EFDT"]
        # this_df[APPROACH_COL][this_df[APPROACH_COL] == "EFDT*"] = "EFDT"

        this_df = this_df.loc[this_df[APPROACH_COL] != "APLASTIC"]
        this_df[ORDER_COL] = this_df[APPROACH_COL].apply(lambda x: approach_order[x])

        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    dataframe = dataframe[dataframe[DATASET_COL] != "STAGGER"]

    metric_to_plot = "classifications correct (percent)"
    plot_relevant_data = dataframe.copy()
    diff_df_efdt = plot_relevant_data[plot_relevant_data["Approach"] != "HT"]
    diff_df_efdt = diff_df_efdt[diff_df_efdt["Approach"] == "EFDT"]
    diff_df_efdt[metric_to_plot] = plot_relevant_data[plot_relevant_data["Approach"] == "PLASTIC"][
                                       metric_to_plot].to_numpy() - diff_df_efdt[metric_to_plot].to_numpy()
    df = diff_df_efdt
    df["Approach"] = "PLASTIC - EFDT"
    df.rename({"classified instances": "\# instances"}, axis=1, inplace=True)
    pal = get_palette()
    pal["PLASTIC - EFDT"] = pal["PLASTIC-A"]
    grid = plot(df, x="\# instances", y=metric_to_plot, palette=pal, sharex=True,
                errorbar=("pi", 100), col_wrap=3, figheight=5 * 2.8/4)

    hlines = [
        [-5, 5],
        [10, 20],
        [25, 50],
        [-10, 10, 20, 30],
        [25, 50],
        [25, 50],
        [10, 20],
        [10, 20],
        [-5, 5, 10, 15],
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
    plt.savefig(os.path.join(os.getcwd(), "figures", "difference_in_accuracy_synth_data.pdf"))
    # plt.show()

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
