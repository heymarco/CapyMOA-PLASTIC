import copy
import os.path

import pandas as pd
from tqdm import tqdm

from capymoa.evaluation import prequential_evaluation_multiple_learners, prequential_evaluation_fast
from capymoa.learner.classifier import PLASTIC, CustomEFDT, CustomHT, EFHAT
from capymoa.stream import Stream

from moa.streams.generators import AgrawalGenerator, HyperplaneGenerator, LEDGenerator, RandomRBFGenerator, \
    RandomTreeGenerator, SEAGenerator, STAGGERGenerator, WaveformGenerator


if __name__ == '__main__':
    reps = 10
    max_instances = int(5e5)
    generators = {
        f"RandomTree ({n_cat})": [Stream(moa_stream=RandomTreeGenerator(), CLI=f"-o 10 -u 0 -v {n_cat}") for i in range(reps)]
        for n_cat in [2, 4, 10]
    }

    for generator_name in tqdm(generators, position=0):

        df = []
        for seed, stream in enumerate(tqdm(generators[generator_name], position=1)):
            learners = {
                f'PLASTIC ({b})': PLASTIC(schema=stream.schema, relative_min_merit=0.5, max_branch_length=b)
                for b in [1, 2, 3, 5, 10, 20]
            }

            results = {}
            for learner_name, learner in learners.items():
                stream.restart()
                results[learner_name] = prequential_evaluation_fast(stream, learner, window_size=500, max_instances=max_instances)

            for approach in results:
                data = results[approach]["windowed"].metrics_per_window()
                data["Wallclock time [s]"] = results[approach]["wallclock"]
                data["CPU time [s]"] = results[approach]["cpu_time"]
                data["Approach"] = approach
                data["Seed"] = seed
                df.append(data)

        df = pd.concat(df).reset_index()

        results_dir = os.path.join(os.getcwd(), "../results/results", "sensitivity_study")
        results_path = os.path.join(results_dir, generator_name + ".parquet")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        df.to_parquet(results_path)
