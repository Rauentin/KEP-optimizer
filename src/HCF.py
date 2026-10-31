import math

import gurobipy as gp
from gurobipy import GRB

# implementation of the Half-Cycle Formulation (cycles only), introduced in
# Delorme et al.: "Half-cycle: A new formulation for modelling kidney exchange problems."
# every cycle is split into two half-cycles, i.e. simple paths with at most ceil(K/2) edges, that meet at
# their start and end vertices, the objective uses edge weights w_h instead of |V_m(h)| + 1

class HCF():
    def __init__(self):
        # filled by build, half_cycles[i] belongs to half_cycle_vars[i]
        self.half_cycles = None
        self.half_cycle_vars = None
        self.graph = None

    def build(self, model, graph, K):
        if K is None or K < 2:
            raise ValueError("HCF needs a cycle length K >= 2")
        # the set H_K, all simple paths with at most ceil(K/2) edges
        half_cycles = graph.get_paths(math.ceil(K / 2))
        half_cycle_vars = []
        end_mappings = {}       # v -> half-cycles with v = v_s(h) or v = v_e(h)
        middle_mappings = {}    # v -> half-cycles with v in V_m(h)
        pair_mappings = {}      # (v_s, v_e) -> half-cycles from v_s to v_e

        for i, h in enumerate(half_cycles):
            half_cycle_vars.append(model.addVar(vtype=GRB.BINARY, name=f"half_cycle_{i}"))
            for v in (h[0], h[-1]):
                if v not in end_mappings:
                    end_mappings[v] = []
                end_mappings[v].append(i)
            for v in h[1:-1]:
                if v not in middle_mappings:
                    middle_mappings[v] = []
                middle_mappings[v].append(i)
            if (h[0], h[-1]) not in pair_mappings:
                pair_mappings[h[0], h[-1]] = []
            pair_mappings[h[0], h[-1]].append(i)

        # a vertex is start or end of at most two chosen half-cycles, or in the middle of at most one
        for v in graph.order:
            ends = end_mappings.get(v, [])
            middles = middle_mappings.get(v, [])
            if ends or middles:
                model.addConstr(gp.quicksum(0.5 * half_cycle_vars[i] for i in ends)
                                + gp.quicksum(half_cycle_vars[i] for i in middles) <= 1, name=f"capacity_{v}")

        # for every chosen half-cycle from v to u a half-cycle from u back to v has to be chosen, u > v
        pairs = set()
        for a, b in pair_mappings:
            pairs.add((a, b) if graph.index(a) < graph.index(b) else (b, a))
        for v, u in pairs:
            model.addConstr(gp.quicksum(half_cycle_vars[i] for i in pair_mappings.get((v, u), []))
                            == gp.quicksum(half_cycle_vars[i] for i in pair_mappings.get((u, v), [])),
                            name=f"balance_{v}_{u}")

        # for odd K a half-cycle of maximum length must not start at the larger of its end vertices,
        # otherwise two of them could combine into a cycle of length K+1
        if K % 2 == 1:
            for i, h in enumerate(half_cycles):
                if graph.index(h[0]) > graph.index(h[-1]) and len(h[1:-1]) == (K - 1) // 2:
                    model.addConstr(half_cycle_vars[i] == 0, name=f"odd_{i}")

        weights = [graph.path_weight(h) for h in half_cycles]

        model.setObjective(gp.quicksum(weights[i] * half_cycle_vars[i] for i, _ in enumerate(half_cycle_vars)), GRB.MAXIMIZE)

        self.half_cycles = half_cycles
        self.half_cycle_vars = half_cycle_vars
        self.graph = graph

    def extract_solution(self, model):
        # returns (cycles, chains) like all formulations, chains stay empty since HCF is cycle only
        if self.half_cycles is None:
            raise RuntimeError("build must be called before extract_solution")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model not solved to optimality, status {model.Status}")
        chosen = [e for h, x in zip(self.half_cycles, self.half_cycle_vars) if x.X > 0.5 for e in self.graph.path_edges(h)]
        return self.graph.edges_to_cycles(chosen), []
