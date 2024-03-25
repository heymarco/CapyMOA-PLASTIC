import copy
import os.path

import pandas as pd
from tqdm import tqdm

from capymoa.evaluation import prequential_evaluation_multiple_learners, prequential_evaluation_fast
from capymoa.learner import MOAClassifier
from capymoa.learner.classifier import PLASTIC, CustomEFDT, CustomHT, EFHAT
from capymoa.learner.classifier.aplastic import APLASTIC
from capymoa.stream import Stream

import moa.streams.generators as gntr
from moa.streams import ConceptDriftStream
from moa.streams.generators import AgrawalGenerator, HyperplaneGenerator, LEDGenerator, RandomRBFGenerator, \
    RandomTreeGenerator, SEAGenerator, STAGGERGenerator, WaveformGenerator, MixedGenerator
import moa.classifiers.drift as clf_drift


if __name__ == '__main__':
    reps = 30
    max_instances = int(2e5)

    gradual_length = 50000
    drift_generators = {
        # "Agrawal-g": [Stream(moa_stream=ConceptDriftStream(),
        #                    CLI=f"-s (generators.AgrawalGenerator -f 1 -i {rep}) -d (generators.AgrawalGenerator -f 2 -i {rep + 1}) -w {gradual_length} -p 100000")
        #             for rep in range(reps)],
        # "LED-g": [Stream(moa_stream=ConceptDriftStream(),
        #                      CLI=f"-s (generators.LEDGeneratorDrift -d 1 -i {rep}) -d (generators.LEDGeneratorDrift -d 3 -i {rep + 1}) -w {gradual_length} -p 100000")
        #           for rep in range(reps)],
        # "LED-a": [Stream(moa_stream=ConceptDriftStream(),
        #                  CLI=f"-s (generators.LEDGeneratorDrift -d 1 -i {rep}) -d (generators.LEDGeneratorDrift -d 3 -i {rep + 1}) -w 50 -p 100000")
        #           for rep in range(reps)],
        # "Agrawal-a": [Stream(moa_stream=ConceptDriftStream(),
        #                      CLI=f"-s (generators.AgrawalGenerator -f 1 -i {rep}) -d (generators.AgrawalGenerator -f 2 -i {rep + 1}) -w 50 -p 100000")
        #               for rep in range(reps)],
        # "RandomTree (c)-d": [Stream(moa_stream=ConceptDriftStream(),
        #                             CLI=f"-s (generators.RandomTreeGenerator -c 5 -r {rep} -i {rep} -o 10 -u 0 -v 2) -d (generators.RandomTreeGenerator -c 5 -r {rep + 1} -i {rep + 1} -o 10 -u 0 -v 2) -w 50 -p 100000")
        #           for rep in range(reps)],
        # "RandomTree (n)-d": [Stream(moa_stream=ConceptDriftStream(),
        #                             CLI=f"-s (generators.RandomTreeGenerator -c 5 -i {rep} -r {rep} -v 2) -d (generators.RandomTreeGenerator -c 5 -r {rep + 1} -i {rep + 1} -v 2) -w 50 -p 100000")
        #                      for rep in range(reps)],
        # "RandomTree-a": [Stream(moa_stream=ConceptDriftStream(),
        #                             CLI=f"-s (generators.RandomTreeGenerator -c 5 -r {rep} -i {rep} -o 0 -u 10) -d (generators.RandomTreeGenerator -c 5 -r {rep + 1} -i {rep + 1} -o 0 -u 10) -w 50 -p 100000")
        #                      for rep in range(reps)],
        # "RandomTree-g": [Stream(moa_stream=ConceptDriftStream(),
        #                         CLI=f"-s (generators.RandomTreeGenerator -c 5 -r {rep} -i {rep} -o 0 -u 10) -d (generators.RandomTreeGenerator -c 5 -r {rep + 1} -i {rep + 1} -o 0 -u 10) -w {gradual_length} -p 100000")
        #                  for rep in range(reps)],
        "RBF-g": [Stream(moa_stream=gntr.RandomRBFGeneratorDrift(), CLI=f"-s 0.0001 -r {rep} -i {rep}") for rep in
                  range(reps)],
        # "RBF-a": [Stream(moa_stream=ConceptDriftStream(),
        #                  CLI=f"-s (generators.RandomRBFGeneratorDrift -s 0.0 -r {rep} -i {rep}) -d (generators.RandomRBFGeneratorDrift -s 0.0 -r {rep + 1} -i {rep + 1}) -w 50 -p 100000")
        #           for rep in range(reps)]
        # "Hyperplane-a": [Stream(moa_stream=ConceptDriftStream(),
        #                  CLI=f"-s (generators.HyperplaneGenerator -i {rep}) -d (generators.HyperplaneGenerator -i {rep + 1}) -w 50 -p 100000")
        #           for rep in range(reps)],
        # "Hyperplane-g": [Stream(moa_stream=gntr.HyperplaneGenerator(), CLI=f"-t 0.01 -i {rep}") for rep in range(reps)],
    }

    for generator_name in tqdm(drift_generators, position=0):

        df = []
        for seed, stream in enumerate(tqdm(drift_generators[generator_name], position=1)):
            plastic = PLASTIC(schema=stream.schema, relative_min_merit=0.5)
            custom_efdt = CustomEFDT(schema=stream.schema)
            ht = CustomHT(schema=stream.schema)
            efhat = EFHAT(schema=stream.schema, min_samples_reevaluate=1)
            plastic_a = MOAClassifier(schema=stream.schema,
                                       moa_learner=clf_drift.BackgroundLearnerClassifier,
                                       CLI="-l trees.PLASTIC")

            learners = {
                'HT': ht,
                'EFDT': custom_efdt,
                'EFHAT': efhat,
                'PLASTIC': plastic,
                'PLASTIC-A': plastic_a,
            }

            results = {}
            for learner_name, learner in learners.items():
                stream.restart()
                results[learner_name] = prequential_evaluation_fast(stream, learner,
                                                                    window_size=500, max_instances=max_instances,
                                                                    record_number_of_leaves=False,
                                                                    record_tree_revisions=False)

            for approach in results:
                data = results[approach]["windowed"].metrics_per_window()
                data["Wallclock time [s]"] = results[approach]["wallclock"]
                data["CPU time [s]"] = results[approach]["cpu_time"]
                data["Approach"] = approach
                data["Seed"] = seed
                # data["Number of leaves"] = results[approach]["number_of_leaves"]
                # data["Revision"] = False
                # data.loc[
                #     data["classified instances"].apply(lambda x: x in results[approach]["tree_revisions"]),
                #     "Tree revision"] = True
                df.append(data)

        df = pd.concat(df).reset_index()

        results_dir = os.path.join(os.getcwd(), "results", "artificial_drift")
        results_path = os.path.join(results_dir, generator_name + ".parquet")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        df.to_parquet(results_path)
