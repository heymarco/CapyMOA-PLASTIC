import os.path

import pandas as pd
from tqdm import tqdm

from capymoa.evaluation import prequential_evaluation_multiple_learners, prequential_evaluation_fast
from capymoa.learner.classifier import PLASTIC, CustomEFDT, CustomHT
from capymoa.stream import Stream

from moa.streams.generators import AgrawalGenerator, HyperplaneGenerator, LEDGenerator, RandomRBFGenerator, \
    RandomTreeGenerator, SEAGenerator, STAGGERGenerator, WaveformGenerator


if __name__ == '__main__':
    reps = 1
    max_instances = 200_000
    generators = {
        "RandomTree": [Stream(moa_stream=RandomTreeGenerator(), CLI=f"-c 5 -o 10 -u 0 -v 2") for i in range(reps)],
    }

    for generator_name in tqdm(generators, position=0):

        df = []
        for seed, stream in enumerate(tqdm(generators[generator_name], position=1)):
            plastic = PLASTIC(schema=stream.schema, relative_min_merit=0.5, max_branch_length=20)
            custom_efdt = CustomEFDT(schema=stream.schema)
            ht = CustomHT(schema=stream.schema)

            learners = {
                'HT': ht,
                'EFDT': custom_efdt,
                'PLASTIC': plastic,
            }

            results = {}
            for learner_name, learner in learners.items():
                stream.restart()
                results[learner_name] = prequential_evaluation_fast(stream, learner, window_size=500,
                                                                    max_instances=max_instances,
                                                                    record_tree_revisions=True,
                                                                    record_number_of_leaves=True)

            for approach in results:
                data = results[approach]["windowed"].metrics_per_window()
                data["Approach"] = approach
                data["Seed"] = seed
                data["Number of leaves"] = results[approach]["number_of_leaves"]
                data["Revision"] = False
                data.loc[
                    data["classified instances"].apply(lambda x: x in results[approach]["tree_revisions"]),
                    "Tree revision"] = True
                df.append(data)

            for learner in learners:
                print(f"{generator_name}-{seed}: {learner} final accuracy = {results[learner]['cumulative'].accuracy()}")

        df = pd.concat(df).reset_index()

        results_dir = os.path.join(os.getcwd(), "results", "motivation")
        results_path = os.path.join(results_dir, generator_name + ".parquet")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        df.to_parquet(results_path)
