import os.path

import pandas as pd
from tqdm import tqdm

from capymoa.evaluation import prequential_evaluation_multiple_learners
from capymoa.learner.classifier import PLASTIC, CustomEFDT, CustomHT
from capymoa.stream import Stream

from moa.streams.generators import AgrawalGenerator, HyperplaneGenerator, LEDGenerator, RandomRBFGenerator, \
    RandomTreeGenerator, SEAGenerator, STAGGERGenerator, WaveformGenerator


if __name__ == '__main__':
    reps = 1
    max_instances = 100_000_000
    generators = {
        "RandomTree": [Stream(moa_stream=RandomTreeGenerator(), CLI=f"-c 5 -o 10 -u 0 -v 2") for i in range(reps)],
    }

    for generator_name in tqdm(generators, position=0):

        df = []
        for seed, stream in enumerate(tqdm(generators[generator_name], position=1)):
            plastic = PLASTIC(schema=stream.schema, relative_min_merit=0.5)
            custom_efdt = CustomEFDT(schema=stream.schema)
            ht = CustomHT(schema=stream.schema)

            learners = {
                'HT': ht,
                'EFDT': custom_efdt,
                'PLASTIC': plastic,
            }

            results = prequential_evaluation_multiple_learners(stream, learners, window_size=1000,
                                                               show_progress=False, max_instances=max_instances)

            for approach in results:
                data = results[approach]["windowed"].metrics_per_window()
                data["Approach"] = approach
                data["Seed"] = seed
                df.append(data)

            for learner in learners:
                print(f"{generator_name}-{seed}: {learner} final accuracy = {results[learner]['cumulative'].accuracy()}")

        df = pd.concat(df).reset_index()

        results_dir = os.path.join(os.getcwd(), "results", "motivation_extended")
        results_path = os.path.join(results_dir, generator_name + ".parquet")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        df.to_parquet(results_path)
