import gurobipy as gp
from gurobipy import GRB

# implementation of the Position-Indexed Edge Formulation (cycles only),
# introduced in Dickerson et al.: "Position-indexed formulations for kidney exchange."
# one copy G^l per vertex l that only contains vertices >= l, every edge additionally gets the position k
# at which it is used in the cycle, the possible positions are given by kappa^l

class PIEF():
    def __init__(self):
        # filled by build, keys are (l, (i, j), k)
        self.edge_vars = None
        self.graph = None

    def kappa(self, l, i, j, K):
        # possible positions of edge (i, j) in copy l, a cycle in copy l starts and ends at l
        if i == l:
            return [1]
        if j == l:
            return list(range(2, K + 1))
        return list(range(2, K))

    def build(self, model, graph, K):
        if K is None or K < 2:
            raise ValueError("PIEF needs a cycle length K >= 2")
        G = graph.Graph
        edge_vars = {}
        in_mappings = {}     # (l, i, k) -> variables of edges into i at position k in copy l
        out_mappings = {}    # (l, i, k) -> variables of edges out of i at position k in copy l
        capacity_mappings = {}  # j -> variables of all edges into j over all copies and positions

        for idx, l in enumerate(graph.order):
            V_l = set(graph.order[idx:])
            for i, j in G.edges:
                if i not in V_l or j not in V_l:
                    continue
                for k in self.kappa(l, i, j, K):
                    x = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{i}_{j}_{k}")
                    edge_vars[l, (i, j), k] = x
                    if (l, j, k) not in in_mappings:
                        in_mappings[l, j, k] = []
                    in_mappings[l, j, k].append(x)
                    if (l, i, k) not in out_mappings:
                        out_mappings[l, i, k] = []
                    out_mappings[l, i, k].append(x)
                    if j not in capacity_mappings:
                        capacity_mappings[j] = []
                    capacity_mappings[j].append(x)

        # a pair receives at most one kidney over all copies and positions
        for j, incoming in capacity_mappings.items():
            model.addConstr(gp.quicksum(incoming) <= 1, name=f"capacity_{j}")

        # an edge into i at position k has to be followed by an edge out of i at position k+1
        for idx, l in enumerate(graph.order):
            for i in graph.order[idx + 1:]:
                for k in range(1, K):
                    incoming = in_mappings.get((l, i, k), [])
                    outgoing = out_mappings.get((l, i, k + 1), [])
                    if incoming or outgoing:
                        model.addConstr(gp.quicksum(incoming) == gp.quicksum(outgoing), name=f"flow_{l}_{i}_{k}")

        model.setObjective(gp.quicksum(G[i][j]['weight'] * x for (l, (i, j), k), x in edge_vars.items()), GRB.MAXIMIZE)

        self.edge_vars = edge_vars
        self.graph = graph

    def extract_solution(self, model):
        # returns (cycles, chains) like all formulations, chains stay empty since this PIEF is cycle only
        if self.edge_vars is None:
            raise RuntimeError("build must be called before extract_solution")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model not solved to optimality, status {model.Status}")
        chosen = [a for (l, a, k), x in self.edge_vars.items() if x.X > 0.5]
        return self.graph.edges_to_cycles(chosen), []
