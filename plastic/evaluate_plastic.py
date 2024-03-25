import os

import matplotlib
import pandas as pd
from tqdm import tqdm

from capymoa.evaluation import prequential_evaluation_fast
from capymoa.learner import MOAClassifier
from capymoa.learner.classifier import CustomEFDT, PLASTIC, CustomHT, EFHAT
from capymoa.learner.classifier.aplastic import APLASTIC
from capymoa.stream import stream_from_file

import moa.classifiers.drift as clf_drift
import moa.classifiers.functions as clf_functions

matplotlib.use('TkAgg')


if __name__ == '__main__':
    datasets = [
        "WISDM",
        "RIALTO",
        "COVTYPE",
        "PAMAP2",
        "SENSORS",
        "GAS",
        "HARTH",
        "KDD",
        "AIRLINES",
        "INSECTS",
        "POKER",
        "HEPMASS",
        "SUSY",
        "HIGGS",
        # "FONTS",
    ]

    versions = [
        "unshuffled",
    ]

    for dataset in tqdm(datasets, position=0, desc="Datasets"):
        for v in tqdm(versions, position=1, desc=dataset):
            rel_path_inside_data = os.path.join("processed", dataset, v)
            data_dir = os.path.join(os.getcwd(), "data", rel_path_inside_data)
            for file in tqdm(os.listdir(data_dir), position=2, desc=v):
                data_path = os.path.join(data_dir, file)
                filename = os.path.splitext(file)[0]
                results_path = os.path.join(os.getcwd(), "results",
                                            os.path.join("hopefully_final", dataset, v),  # rel_path_inside_data,
                                            filename + ".parquet")

                assert os.path.exists(data_path)
                stream = stream_from_file(path_to_csv_or_arff=data_path)

                aplastic = APLASTIC(schema=stream.schema, min_samples_reevaluate=20)
                plastic = PLASTIC(schema=stream.schema)
                custom_efdt = CustomEFDT(schema=stream.schema)
                ht = CustomHT(schema=stream.schema)
                efhat = EFHAT(schema=stream.schema)

                learners = {
                    'NoChange': MOAClassifier(schema=stream.schema,
                                              moa_learner=clf_functions.NoChange),
                    'HT': ht,
                    'EFDT': custom_efdt,
                    # 'HT-c': MOAClassifier(schema=stream.schema,
                    #                          moa_learner=clf_drift.BackgroundLearnerClassifier,
                    #                          CLI="-l trees.CustomHT"),
                    # 'EFDT-c': MOAClassifier(schema=stream.schema,
                    #                       moa_learner=clf_drift.BackgroundLearnerClassifier,
                    #                       CLI="-l trees.EFDT"),
                    'EFHAT': efhat,
                    'PLASTIC': plastic,
                    'PLASTIC-A': MOAClassifier(schema=stream.schema,
                                                  moa_learner=clf_drift.BackgroundLearnerClassifier,
                                                  CLI="-l trees.PLASTIC"),
                }

                df = []
                results = {}
                for learner_name, learner in learners.items():
                    stream.restart()
                    results[learner_name] = prequential_evaluation_fast(stream, learner, window_size=1000)

                for approach in results:
                    data = results[approach]["windowed"].metrics_per_window()
                    data["Wallclock time [s]"] = results[approach]["wallclock"]
                    data["CPU time [s]"] = results[approach]["cpu_time"]
                    data["Approach"] = approach
                    df.append(data)

                df = pd.concat(df).reset_index()

                results_dir = os.path.split(results_path)[0]
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                df.to_parquet(results_path)

                for learner in learners:
                    print(f"{filename}: {learner} final accuracy = {results[learner]['cumulative'].accuracy()}")

                # plot_windowed_results(results['HT'], results['EFDT'], results['Own EFDT'], results['PLASTIC'], metric="classifications correct (percent)")

