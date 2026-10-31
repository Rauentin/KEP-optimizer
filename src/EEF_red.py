import networkx as nx

import gurobipy as gp
from gurobipy import GRB

# implementation of the reduced Extended-Edge Formulation (cycles only),
# introduced in Constantino et al.: "New insights on integer-programming models for the kidney exchange problem."
# one copy G^l per vertex l that only contains vertices >= l, l has to be the lowest vertex of the cycle in
# its copy, vertices and edges that cannot lie on a cycle of length <= K through l are removed using the
# shortest path distances d^l_ij

class EEF_red():
    def __init__(self):
        # filled by build, keys are (l, (i, j))
        self.edge_vars = None
        self.graph = None

    def build(self, model, graph, K):
        if K is None or K < 2:
            raise ValueError("EEF_red needs a cycle length K >= 2")
        G = graph.Graph
        inf = float('inf')

        # reduced sets V~^l and A~^l, the set L only keeps copies that still contain edges
        copies = {}
        for idx, l in enumerate(graph.order):
            G_l = G.subgraph(graph.order[idx:])
            d_from_l = nx.single_source_shortest_path_length(G_l, l)
            d_to_l = nx.single_source_shortest_path_length(nx.reverse_view(G_l), l)
            V_l = [i for i in G_l.nodes if d_from_l.get(i, inf) + d_to_l.get(i, inf) <= K]
            A_l = [(i, j) for i, j in G_l.edges
                   if i in V_l and j in V_l and d_from_l.get(i, inf) + 1 + d_to_l.get(j, inf) <= K]
            if A_l:
                copies[l] = (V_l, A_l)

        edge_vars = {}
        in_mappings = {}     # (l, i) -> variables of edges into i in copy l
        out_mappings = {}    # (l, i) -> variables of edges out of i in copy l
        for l, (V_l, A_l) in copies.items():
            for i in V_l:
                in_mappings[l, i] = []
                out_mappings[l, i] = []
            for i, j in A_l:
                x = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{i}_{j}")
                edge_vars[l, (i, j)] = x
                out_mappings[l, i].append(x)
                in_mappings[l, j].append(x)

        for l, (V_l, A_l) in copies.items():
            for i in V_l:
                # flow conservation in every copy
                model.addConstr(gp.quicksum(in_mappings[l, i]) == gp.quicksum(out_mappings[l, i]), name=f"flow_{l}_{i}")
                # a vertex can only be used in copy l if l is used as well
                model.addConstr(gp.quicksum(out_mappings[l, i]) <= gp.quicksum(out_mappings[l, l]), name=f"lowest_{l}_{i}")
            # at most K chosen edges per copy
            model.addConstr(gp.quicksum(edge_vars[l, a] for a in A_l) <= K, name=f"length_{l}")

        # a pair receives at most one kidney over all copies
        for i in graph.order:
            incoming = [x for l in copies if (l, i) in in_mappings for x in in_mappings[l, i]]
            if incoming:
                model.addConstr(gp.quicksum(incoming) <= 1, name=f"capacity_{i}")

        model.setObjective(gp.quicksum(G[i][j]['weight'] * x for (l, (i, j)), x in edge_vars.items()), GRB.MAXIMIZE)

        self.edge_vars = edge_vars
        self.graph = graph

    def extract_solution(self, model):
        # returns (cycles, chains) like all formulations, chains stay empty since EEF is cycle only
        if self.edge_vars is None:
            raise RuntimeError("build must be called before extract_solution")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model not solved to optimality, status {model.Status}")
        chosen = [a for (l, a), x in self.edge_vars.items() if x.X > 0.5]
        return self.graph.edges_to_cycles(chosen), []
