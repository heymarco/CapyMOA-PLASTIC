import os
from functools import partial

import numpy as np
import pandas as pd
from critdd import Diagram
from pandas.io.formats.style import _highlight_value
from scipy.stats import gmean

from util import get_palette, get_all_approaches, get_approaches_synthetic, get_all_approaches_wo_nochange, \
    get_rw_datasets_in_paper_order


def add_diff_column(df: pd.DataFrame, column_to_differentiate) -> pd.Series:
    s = df[column_to_differentiate].diff()
    return s


def add_average(df: pd.DataFrame, use_gmean=False, name="Accuracy", precision=1):
    summary_styler = (df
                      .agg(["mean"] if not use_gmean else [lambda x: np.product(x) ** (1.0 / len(x[np.isnan(x) == False]))])
                      .round(precision)
                      .style.relabel_index([name]))
    return summary_styler


if __name__ == '__main__':
    METRIC = "classifications correct (percent)"
    DATASET_COL = "Dataset"
    APPROACH_COL = "Approach"
    ORDER_COL = "Order"
    approach_order = {key: i for i, key in enumerate(get_all_approaches())}

    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "hopefully_final")
    for dataset_name in os.listdir(main_result_dir):
        if dataset_name == "aplastic":
            continue
        if "FONTS" in dataset_name:
            continue
        dataset_dir_unshuffled = os.path.join(main_result_dir, dataset_name, "unshuffled")
        ds = dataset_name.split("_")[0]
        filepath = os.path.join(dataset_dir_unshuffled, ds + ".parquet")

        this_df = pd.read_parquet(filepath)
        this_df[DATASET_COL] = ds
        this_df[ORDER_COL] = this_df[APPROACH_COL].apply(lambda x: approach_order[x])

        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)


    def add_avg_rank(df: pd.DataFrame):
        ranks = Diagram(
            df[get_all_approaches_wo_nochange()].dropna(axis=0).to_numpy(),
            treatment_names=get_all_approaches_wo_nochange(),
            maximize_outcome=True
        ).average_ranks
        additional_df = pd.DataFrame(data=[list(ranks) + [-1]], columns=get_all_approaches()).round(2)
        return additional_df.style.relabel_index(["Rank"])

    # GET AVERAGE ACCURACY
    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC, ORDER_COL]]
                      .groupby([DATASET_COL, ORDER_COL, APPROACH_COL])
                      .mean()
                      # .drop(ORDER_COL, axis=1)
                      )
    # average_acc_df.index = average_acc_df.index.droplevel(1)
    pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)

    # GET AVERAGE RUNTIME
    METRIC = "Wallclock time [s]"
    average_acc_df = (dataframe[[DATASET_COL, APPROACH_COL, METRIC, ORDER_COL]]
                      .groupby([DATASET_COL, ORDER_COL, APPROACH_COL])
                      .mean()
                      # .drop(ORDER_COL, axis=1)
                      )
    average_acc_df.index = average_acc_df.index.droplevel(1)
    rt_pivot = average_acc_df.reset_index().pivot(columns=APPROACH_COL, index=DATASET_COL, values=METRIC)

    pivot = pivot.loc[get_rw_datasets_in_paper_order(), get_all_approaches()]
    rt_pivot = rt_pivot.loc[get_rw_datasets_in_paper_order(), get_all_approaches()]

    print(pivot.corr())

    print((pivot
           .dropna(axis=0)
           .round(1)
           .style.format(precision=0).highlight_max(axis=1, props="font-weight: bold;", subset=get_all_approaches_wo_nochange()).format(precision=1)
           .concat(add_average(pivot.dropna(axis=0), name="Accuracy").highlight_max(axis=1, props="font-weight: bold;",
                                                                   subset=get_all_approaches_wo_nochange()).format(precision=1))
           .concat(add_avg_rank(pivot.dropna(axis=0)).highlight_min(axis=1, props="font-weight: bold;",
                                                                   subset=get_all_approaches_wo_nochange()).format(precision=2))
           .concat(add_average(rt_pivot[get_all_approaches()].dropna(axis=0), name="Runtime", precision=1).highlight_min(axis=1, props="font-weight: bold;",
                                                                    subset=get_all_approaches_wo_nochange()).format(precision=1))
           .to_latex(convert_css=True)))

    print((rt_pivot
           .dropna(axis=0)
           .style.format(precision=2).highlight_min(axis=1, props="font-weight: bold;").format(precision=0)
           .concat(add_average(pivot, use_gmean=True).highlight_min(axis=1, props="font-weight: bold;").format(precision=0))
           .to_latex(convert_css=True)))
