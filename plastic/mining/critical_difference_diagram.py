import os

import Orange
import numpy as np
import pandas as pd
from critdd import Diagram
from matplotlib import pyplot as plt
from scipy.stats import gmean

from util import get_palette, get_approaches_synthetic, get_all_approaches_wo_nochange

from Orange.evaluation import compute_CD, graph_ranks

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns


def add_diff_column(df: pd.DataFrame, column_to_differentiate) -> pd.Series:
    s = df[column_to_differentiate].diff()
    return s


if __name__ == '__main__':
    alpha = 0.05

    METRIC = "classifications correct (percent)"
    DATASET_COL = "id"
    APPROACH_COL = "Approach"
    approach_order = {key: i for i, key in enumerate(get_all_approaches_wo_nochange())}

    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "hopefully_final")
    for dataset_name in os.listdir(main_result_dir):
        if dataset_name == "aplastic":
            continue
        dataset_dir_unshuffled = os.path.join(main_result_dir, dataset_name, "unshuffled")
        ds = dataset_name.split("_")[0]
        if ds == "FONTS":
            continue
        filepath = os.path.join(dataset_dir_unshuffled, ds + ".parquet")
        this_df = pd.read_parquet(filepath)

        this_df[DATASET_COL] = ds
        this_df = this_df[this_df[APPROACH_COL] != "NoChange"]
        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)


    dataframes = [dataframe]
    main_result_dir = os.path.join(os.getcwd(), "results", "artificial")
    for dataset_name in os.listdir(main_result_dir):
        ds = os.path.splitext(dataset_name)[0].split("_")[0]
        if ds == "STAGGER":
            continue

        filepath = os.path.join(main_result_dir, dataset_name)
        this_df = pd.read_parquet(filepath)
        this_df[DATASET_COL] = ds
        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "PLASTIC"]
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "EFDT"]
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "PLASTIC*"] = "PLASTIC"
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "EFDT*"] = "EFDT"

    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC]]
                      .groupby([DATASET_COL, APPROACH_COL])
                      .mean()
                      )

    pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)
    pivot = pivot[get_all_approaches_wo_nochange()]

    # Critical difference diagram to get average ranks
    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    new_cols = []
    for i, c in enumerate(pivot.columns):
        new_c = str(c) + f" ({round(d.average_ranks[i], 2)})"
        new_cols.append(new_c)
    pivot.columns = new_cols

    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    print(d.average_ranks)  # the average rank of each treatment
    print(d.get_groups(alpha=alpha,
                       adjustment="holm"
                       ))

    # export the diagram to a file
    d.to_file(
        os.path.join(os.getcwd(), "figures", "cdd_all.tex"),
        alpha=alpha,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": ""},
    )

    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "hopefully_final")
    for dataset_name in os.listdir(main_result_dir):
        dataset_dir_unshuffled = os.path.join(main_result_dir, dataset_name, "unshuffled")
        ds = dataset_name.split("_")[0]
        if ds == "FONTS":
            continue
        filepath = os.path.join(dataset_dir_unshuffled, ds + ".parquet")
        this_df = pd.read_parquet(filepath)

        this_df = this_df[this_df[APPROACH_COL] != "NoChange"]
        this_df[DATASET_COL] = ds

        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "PLASTIC"]
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "EFDT"]
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "PLASTIC*"] = "PLASTIC"
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "EFDT*"] = "EFDT"

    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC]]
                      .groupby([DATASET_COL, APPROACH_COL])
                      .mean()
                      )

    pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)
    pivot = pivot[get_all_approaches_wo_nochange()]

    # Critical difference diagram to get average ranks
    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    new_cols = []
    for i, c in enumerate(pivot.columns):
        new_c = str(c) + f" ({round(d.average_ranks[i], 2)})"
        new_cols.append(new_c)
    pivot.columns = new_cols

    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    print(d.average_ranks)  # the average rank of each treatment
    print(d.get_groups(alpha=alpha,
                       adjustment="holm"
                       ))

    # export the diagram to a file
    d.to_file(
        os.path.join(os.getcwd(), "figures", "cdd_real.tex"),
        alpha=alpha,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": ""},
    )


    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "artificial")
    for dataset_name in os.listdir(main_result_dir):
        ds = os.path.splitext(dataset_name)[0].split("_")[0]
        if ds == "STAGGER":
            continue

        filepath = os.path.join(main_result_dir, dataset_name)
        this_df = pd.read_parquet(filepath)
        this_df[DATASET_COL] = ds
        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "PLASTIC"]
    # dataframe = dataframe.loc[dataframe[APPROACH_COL] != "EFDT"]
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "PLASTIC*"] = "PLASTIC"
    # dataframe[APPROACH_COL].loc[dataframe[APPROACH_COL] == "EFDT*"] = "EFDT"

    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC, "Seed"]]
                      .groupby([DATASET_COL, APPROACH_COL, "Seed"])
                      .mean()
                      )

    pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=[DATASET_COL, "Seed"], values=METRIC)
    pivot = pivot[get_approaches_synthetic()]

    # Critical difference diagram to get average ranks
    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    new_cols = []
    for i, c in enumerate(pivot.columns):
        new_c = str(c) + f" ({round(d.average_ranks[i], 2)})"
        new_cols.append(new_c)
    pivot.columns = new_cols

    d = Diagram(
        pivot.round(2).dropna(axis=0).to_numpy(),
        treatment_names=pivot.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    print(d.average_ranks)  # the average rank of each treatment
    print(d.get_groups(alpha=alpha,
                       adjustment="holm"
                       ))

    # export the diagram to a file
    d.to_file(
        os.path.join(os.getcwd(), "figures", "cdd_synth.tex"),
        alpha=alpha,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": ""},
    )
