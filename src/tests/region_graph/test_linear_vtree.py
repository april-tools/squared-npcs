import itertools

import pytest

from region_graph.linear_vtree import LinearVTree
from tests.region_graph.test_region_graph import (
    check_region_graph_save_load,
    check_region_partition_layers,
)


@pytest.mark.parametrize(
    "num_variables,num_repetitions,randomize", itertools.product([1, 2, 5, 12], [1, 3], [False, True])
)
def test_linear_vtree(num_variables: int, num_repetitions: int, randomize: bool) -> None:
    rg = LinearVTree(num_variables, num_repetitions=num_repetitions, randomize=randomize)
    assert rg.num_variables == num_variables
    assert rg.is_smooth
    assert rg.is_decomposable
    if num_repetitions == 1 or not randomize:
        assert rg.is_structured_decomposable
    check_region_partition_layers(rg, bottom_up=True)
    check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)
