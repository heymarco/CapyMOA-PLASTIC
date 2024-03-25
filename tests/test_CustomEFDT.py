from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier import CustomEFDT
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest


@pytest.mark.parametrize(
    "stream, expectation",
    [
        (ElectricityTiny(), 46.0),
        (CovtypeTiny(), 46.0),
    ],
    ids=["ElectricityTiny", "CovtypeTiny"]
)
def test_CustomEFDT(stream, expectation):
    # The optimizer steps are set to 10 to speed up the test
    learner = CustomEFDT(
        schema=stream.schema,
        grace_period=201,
        min_samples_reevaluate=21,
        # split_criterion="gini",
        confidence=1e-3,
        tie_threshold=0.055,
        # leaf_prediction="mc",
        # numeric_attribute_observer="FIMTDDNumericAttributeClassObserver",
        binary_split=True,
        min_branch_fraction=0.02,
        max_share_to_split=0.98,
        disable_prepruning=False,
    )
    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
    )
