from __future__ import annotations

import inspect

from capymoa.learner import MOAClassifier
import moa.classifiers.trees as moa_trees
from capymoa.stream import Schema


class PLASTIC(MOAClassifier):
    """PLASTIC algorithm.

    TODO

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    min_samples_reevaluate
        Number of instances a node should observe before reevaluating the best split.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    confidence
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    tau_reevaluate
        Threshold below which a split will be forced to break ties during reevaluation.
    relative_min_merit
        Minimum information gain above which tie breaking occurs. Relative, will be multiplied by tau_reevaluate.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    numeric_attribute_observer
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_byte_size
        The max size of the tree, in bytes.
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    disable_prepruning
        If True, disable merit-based tree pre-pruning.
    """

    MAJORITY_CLASS = 0
    NAIVE_BAYES = 1
    NAIVE_BAYES_ADAPTIVE = 2

    def __init__(
            self,
            schema: Schema | None = None,
            random_seed: int = 0,
            grace_period: int = 200,
            min_samples_reevaluate: int = 200,
            split_criterion: str = "InfoGainSplitCriterion",
            confidence: float = 1e-3,
            tie_threshold: float = 0.05,
            tie_threshold_reevaluate: float = 0.05,
            relative_min_merit: float = 0.5,
            max_branch_length: int = 5,
            leaf_prediction: str = MAJORITY_CLASS,
            binary_split: bool = False
    ):
        # Example configuration string:
        # "trees.EFDT -R 2001 -m 33554433 -n FIMTDDNumericAttributeClassObserver -e 10003000 -g 201 -s GiniSplitCriterion -c 0.002 -t 0.051 -b -z -r -p -l NB -q 1"

        mappings = {
            "grace_period": "-g",
            "min_samples_reevaluate": "-R",
            "split_criterion": "-s",
            "confidence": "-c",
            "tie_threshold": "-t",
            "tie_threshold_reevaluate": "-T",
            "relative_min_merit": "-G",
            "binary_split": "-b",
            "leaf_prediction": "-l",
            "max_branch_length": "-B",
        }

        config_str = ""
        parameters = inspect.signature(self.__init__).parameters
        for key in mappings:
            if key not in parameters:
                continue
            this_parameter = parameters[key]
            default_value = this_parameter.default
            set_value = locals()[key]
            is_bool = type(set_value) == bool
            if is_bool:
                if set_value:
                    str_extension = mappings[key] + " "
                else:
                    str_extension = ""
            else:
                str_extension = f"{mappings[key]} {set_value} "
            config_str += str_extension

        super(PLASTIC, self).__init__(moa_learner=moa_trees.PLASTIC,
                                      schema=schema,
                                      CLI=config_str,
                                      random_seed=random_seed)
