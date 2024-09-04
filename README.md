# CapyMOA-PLASTIC

This code contains a python version of plastic using CapyMOA, and the scripts for running the experiments from the paper and plotting the results. The java code of plastic can be found under [github.com/heymarco/PLASTIC](https://github.com/heymarco/PLASTIC)

## Abstract

Commonly used incremental decision trees for mining data streams include Hoeffding Trees (HT) and Extremely Fast Decision Trees (EFDT). EFDT exhibits faster learning than HT. However, due to its split revision procedure, EFDT suffers from sudden and unpredictable accuracy decreases caused by subtree pruning. To overcome this, we propose PLASTIC, an incremental decision tree that restructures the otherwise pruned subtree. This is possible due to *decision tree plasticity*: one can alter a tree's structure without affecting its predictions. We conduct extensive evaluations comparing PLASTIC with state-of-the-art methods on synthetic and real-world data streams. 
Our results show that PLASTIC improves EFDT's worst-case accuracy by up to 50 % and outperforms the current state of the art on real-world data. 
We provide an open-source implementation of PLASTIC within the MOA framework for mining high-speed data streams.

## Citing
If you want to cite this paper, use
```
@inproceedings{heyden2024leveraging,
  title={Leveraging Plasticity in Incremental Decision Trees},
  author={Heyden, Marco and Gomes, Heitor Murilo and Fouch{\'e}, Edouard and Pfahringer, Bernhard and B{\"o}hm, Klemens},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={38--54},
  year={2024},
  organization={Springer}
}
```

# CapyMOA
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API


# Developer Installation

1. **(Optional)** It is recommended to use a conda environment since using a
   conda environment isolates project dependencies and avoids conflicts.
   > Follow the instructions [at this link](https://docs.conda.io/projects/miniconda/en/latest/) to install miniconda.

   Setup the conda environment by running one of the following commands:
   ```sh
   conda env create -f environment.yml # For linux
   conda env create -f environment_wds.yml # For windows
   ```
   Ensure the environment is activated:
   ```sh
   conda activate CapyMOA
   ```
2. Use pip to install the project in editable mode with the development dependencies:
   ```bash
   pip install --editable '.[dev]'
   ```
3. Make sure the environment variable `JAVA_HOME` is set.
4. Run the tests to make sure everything is working:
   ```bash
   python -m pytest
   ```
5. **(Optional)** Try the example notebooks
   1. Download extra datasets with `make download`.
   2. Run the DEMO notebook ```python -m jupyter notebook notebooks/DEMO.ipynb```. The notebook must be
   started with the correct ```JAVA_HOME``` variable set.
   To double check run ```echo $JAVA_HOME``` in the
   terminal before starting the notebook.


# Functionality
* Full support for classification, regression and semi-supervised classification. 
* Read CSV or ARFF files, or use synthetic generators from MOA.

# Tutorial notebooks
These notebooks show how to do things. Data is available in the ```/data/``` directory (some of which will need to be downloaded, see instrucitons there). 

* **DEMO.ipynb**: Contains simple examples on how to execute classification and regression, using MOA objets to configure synthetic generators or classifiers/regressors. 
* **Evaluation_and_Data_Reading.ipynb**: Many examples showing how to perform different evaluations for classification and regression using different methods (i.e. a loop or buildin functions). 
* **Learners_API_Examples.ipynb**: Similar to the DEMO, but shows more capabilities of the evaluator and learner objects.
* **Using_sklearn_pytorch.ipynb**: Shows how one can use the API to run sklearn algorithms (those that implement ```partial_fit```) and PyTorch models. 

# Test notebooks
These show how some parts of the library were developed and provide comparisons of different options on how to do things. 

* **Efficient_Evaluation.ipynb**: Some simple benchmarks comparing different versions of test_then_train_evaluation and prequential_evaluation. Interesting to developers looking to improve that aspect of the platform. 
* **Using_jpype_MOA_example.ipynb**: Example using MOA directly from jpype without the library in-between. Interesting to developers looking for a full example of how it is done without the library. 
* **Data_Reading.ipynb**: Data reading examples. More interesting to developers looking to improve the data capabilities. 

**Updated all the notebooks on 16/01/2024, removed some that were outdated**
