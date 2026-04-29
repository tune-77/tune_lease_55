import random

from evaluators.selector import EpsilonGreedySelector
from evaluators.types import EvaluatorParams


def test_selector_epsilon_zero_exploit():
    random.seed(1)
    s = EpsilonGreedySelector(EvaluatorParams(epsilon=0.0, exploit_engine='quantum', exploration_pool=['mahalanobis']))
    assert s._choose() == 'quantum'


def test_selector_epsilon_one_explore():
    random.seed(1)
    s = EpsilonGreedySelector(EvaluatorParams(epsilon=1.0, exploit_engine='quantum', exploration_pool=['mahalanobis']))
    assert s._choose() == 'mahalanobis'
