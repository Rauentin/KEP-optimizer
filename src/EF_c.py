import gurobipy as gp
from gurobipy import GRB

# implementation of the Edge Formulation (cycles only), see Abraham et al.:
# "Clearing algorithms for barter exchange markets: enabling nationwide kidney exchanges."
# one binary variable per edge, chosen edges form vertex disjoint cycles through flow conservation
# cycle length is restricted by constraints on all simple paths with K-1 edges, whose number grows
# quickly with K and edge density

class EF_c():
    def __init__(self):
        # filled by build
        self.edge_vars = None
        self.graph = None

    def build(self, model, graph, K):
        if K is None or K < 2:
            raise ValueError("EF_c needs a cycle length K >= 2")
        G = graph.Graph
        edge_vars = {}
        for u, v in G.edges:
            edge_vars[u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

        for i in G.nodes:
            inflow = gp.quicksum(edge_vars[j, i] for j in G.predecessors(i))
            outflow = gp.quicksum(edge_vars[i, j] for j in G.successors(i))
            # a pair that receives a kidney also donates one
            model.addConstr(inflow == outflow, name=f"flow_{i}")
            # a pair donates at most once
            model.addConstr(outflow <= 1, name=f"capacity_{i}")

        # if all edges of a simple path p = (v_0, ..., v_K-1) are chosen, the closing edge (v_K-1, v_0)
        # has to exist and be chosen as well
        for n, p in enumerate(graph.get_paths(K - 1, min_len=K - 1)):
            lhs = gp.quicksum(edge_vars[a] for a in graph.path_edges(p))
            if G.has_edge(p[-1], p[0]):
                lhs -= edge_vars[p[-1], p[0]]
            model.addConstr(lhs <= K - 2, name=f"path_{n}")

        model.setObjective(gp.quicksum(G[u][v]['weight'] * x for (u, v), x in edge_vars.items()), GRB.MAXIMIZE)

        self.edge_vars = edge_vars
        self.graph = graph

    def extract_solution(self, model):
        # returns (cycles, chains) like all formulations, chains stay empty since EF is cycle only
        if self.edge_vars is None:
            raise RuntimeError("build must be called before extract_solution")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model not solved to optimality, status {model.Status}")
        chosen = [e for e, x in self.edge_vars.items() if x.X > 0.5]
        return self.graph.edges_to_cycles(chosen), []
