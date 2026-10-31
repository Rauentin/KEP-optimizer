# random compatibility graphs for testing and comparing formulations

import random

from graph_methods.comp_graph import CompGraph, Edge, Vertex


def random_graph(n_pairs, p, n_ndds=0, weights=(1, 1), seed=None):
    # pairs get ids 0..n_pairs-1, ndds the ids after that
    # every possible edge exists with probability p, ndds never get incoming edges
    # edge weights are integers drawn uniformly from the closed interval weights
    rng = random.Random(seed)
    vertices = [Vertex(i, None, None, 0.0, False) for i in range(n_pairs)]
    vertices += [Vertex(n_pairs + i, None, None, 0.0, True) for i in range(n_ndds)]

    edges = []
    for u in vertices:
        for v in vertices[:n_pairs]:
            if u is not v and rng.random() < p:
                edges.append(Edge(u, v, rng.randint(*weights)))

    g = CompGraph()
    g.add_nodes(vertices)
    g.add_edges(edges)
    return g
