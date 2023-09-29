import numpy as np

from .region_graph import RegionGraph
from .rg_node import PartitionNode, RegionNode


def RandomBinaryTree(num_vars: int, num_repetitions: int = 1, seed: int = 42, depth: int = -1) -> RegionGraph:
    vs = list(range(num_vars))
    root = RegionNode(vs)
    graph = RegionGraph()
    graph.add_node(root)
    random_state = np.random.RandomState(seed)
    if depth < 0:
        depth = float('inf')

    for replica_idx in range(num_repetitions):
        q = list()
        q.append(root)
        while q and len(graph.topological_layers(bottom_up=False)) <= depth:
            region_node = q.pop()
            rvs = list(region_node.scope)
            if len(rvs) == 1:
                continue
            random_state.shuffle(rvs)
            rvs_left, rvs_right = rvs[:len(rvs) // 2], rvs[len(rvs) // 2:]
            partition_node = PartitionNode(rvs)
            region_left = RegionNode(rvs_left, replica_idx=replica_idx)
            region_right = RegionNode(rvs_right, replica_idx=replica_idx)
            graph.add_edge(partition_node, region_node)
            graph.add_edge(region_left, partition_node)
            graph.add_edge(region_right, partition_node)
            q.append(region_left)
            q.append(region_right)

    return graph
