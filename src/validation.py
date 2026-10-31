# solver independent checks, used to validate the formulations against each other


def check_solution(graph, cycles, chains, K, objective=None, tol=1e-6):
    # raises ValueError if the solution is infeasible for cycle cap K or its weight does not match objective
    # returns the total weight of the solution
    if chains:
        raise ValueError("chains are not supported by the checker yet")

    used = set()
    total = 0
    for c in cycles:
        if len(set(c)) != len(c):
            raise ValueError(f"cycle {c} repeats a vertex")
        if K is not None and len(c) > K:
            raise ValueError(f"cycle {c} is longer than K={K}")
        if used & set(c):
            raise ValueError(f"cycle {c} shares vertices {used & set(c)} with another cycle")
        used |= set(c)
        total += graph.cycle_weight(c)

    if objective is not None and abs(total - objective) > tol:
        raise ValueError(f"solution weight {total} does not match objective {objective}")
    return total


def brute_force(graph, K):
    # optimal total weight by trying every edge set in which each vertex has at most one chosen incoming
    # and outgoing edge, independent of get_cycles and of any ILP, only usable for tiny graphs
    edges = list(graph.Graph.edges(data='weight'))
    best = 0

    def search(i, out_used, in_used, chosen, total):
        nonlocal best
        if i == len(edges):
            # equal sets means every used vertex has one in and one out edge, i.e. disjoint cycles
            if out_used == in_used and total > best:
                if K is None or all(len(c) <= K for c in graph.edges_to_cycles(chosen)):
                    best = total
            return
        search(i + 1, out_used, in_used, chosen, total)
        u, v, w = edges[i]
        if u not in out_used and v not in in_used:
            search(i + 1, out_used | {u}, in_used | {v}, chosen + [(u, v)], total + w)

    search(0, frozenset(), frozenset(), [], 0)
    return best
