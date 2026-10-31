import gurobipy as gp
from gurobipy import GRB

# implementation of the unreduced Extended-Edge Formulation (cycles only),
# introduced in Constantino et al.: "New insights on integer-programming models for the kidney exchange problem."
# n_copies copies of the whole graph (default |V|), every copy holds at most K chosen edges,
# so every cycle in it has length at most K

class EEF():
    def __init__(self, n_copies=None):
        self.n_copies = n_copies
        # filled by build, keys are (l, (i, j))
        self.edge_vars = None
        self.graph = None

    def build(self, model, graph, K):
        if K is None or K < 2:
            raise ValueError("EEF needs a cycle length K >= 2")
        G = graph.Graph
        n_copies = self.n_copies if self.n_copies is not None else G.number_of_nodes()
        copies = range(1, n_copies + 1)
        edge_vars = {}
        for l in copies:
            for i, j in G.edges:
                edge_vars[l, (i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{i}_{j}")

        # flow conservation in every copy
        for l in copies:
            for i in G.nodes:
                model.addConstr(gp.quicksum(edge_vars[l, (j, i)] for j in G.predecessors(i))
                                == gp.quicksum(edge_vars[l, (i, j)] for j in G.successors(i)), name=f"flow_{l}_{i}")

        # a pair receives at most one kidney over all copies
        for i in G.nodes:
            model.addConstr(gp.quicksum(edge_vars[l, (j, i)] for l in copies for j in G.predecessors(i)) <= 1,
                            name=f"capacity_{i}")

        # at most K chosen edges per copy
        for l in copies:
            model.addConstr(gp.quicksum(edge_vars[l, a] for a in G.edges) <= K, name=f"length_{l}")

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
