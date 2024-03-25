import os.path

import arff
import pandas as pd
import seaborn as sns


def get_all_approaches():
    return [
        "HT",
        # "HT-c",
        "EFDT",
        # "EFDT-c",
        "EFHAT",
        "PLASTIC",
        "PLASTIC-A",
        "NoChange",
    ]


def get_all_approaches_wo_nochange():
    return [a for a in get_all_approaches() if a != "NoChange"]


def get_approaches_synthetic():
    return [
        "HT",
        "EFDT",
        "EFHAT",
        "PLASTIC",
        "PLASTIC-A",
    ]


def get_rw_datasets_in_paper_order():
    return [
        "RIALTO",
        "SENSORS",
        "COVTYPE",
        "HARTH",
        "PAMAP2",
        "WISDM",
        "HEPMASS",
        "HIGGS",
        "SUSY",
        "AIRLINES",
        "GAS",
        "INSECTS",
        "KDD",
        "POKER",
    ]


def get_palette() -> dict:
    colors = sns.cubehelix_palette(n_colors=len(get_all_approaches_wo_nochange()))
    return {a: c for a, c in zip(get_all_approaches_wo_nochange(), colors)}


def df_to_arff(df: pd.DataFrame, filepath: str, attributes=None):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    if attributes is None:
        attributes = [(str(j), 'NUMERIC') if df[j].dtypes not in ['str', "category"]
                      else (str(j), df[j].unique().astype(str).tolist()) for j in df]

    data = df.to_numpy()

    arff_dic = {
        'attributes': attributes,
        'data': data,
        'relation': 'myRel',
        'description': ''
    }

    with open(filepath, "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)
