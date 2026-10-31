import gurobipy as gp
from gurobipy import GRB

# implementation of the Cycle Formulation first introduced in Abraham et al.: "Clearing algorithms
# for barter exchange markets: enabling nationwide kidney exchanges."

class CF():
    def __init__(self):
        # filled by build, cycles[i] belongs to cycle_vars[i]
        self.cycles = None
        self.cycle_vars = None

    def build(self, model, graph, K):
        cycles = graph.get_cycles(K)
        cycle_mappings = {}
        cycle_vars = []

        for i, cycle in enumerate(cycles):
            cycle_vars.append(model.addVar(vtype=GRB.BINARY, name=f"cycle_{i}"))
            for v in cycle:
                if v not in cycle_mappings.keys():
                    cycle_mappings[v] = []
                cycle_mappings[v].append(i)


        for v, idcs in cycle_mappings.items():
            model.addConstr(gp.quicksum(cycle_vars[i] for i in idcs) <= 1, name=f"vertex_{v}")

        weights = [graph.cycle_weight(c) for c in cycles]

        model.setObjective(gp.quicksum(weights[i] * cycle_vars[i] for i,_ in enumerate(cycle_vars)), GRB.MAXIMIZE)

        self.cycles = cycles
        self.cycle_vars = cycle_vars

    def extract_solution(self, model):
        # returns (cycles, chains) like all formulations, chains stay empty since CF is cycle only
        if self.cycles is None:
            raise RuntimeError("build must be called before extract_solution")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"model not solved to optimality, status {model.Status}")
        chosen = [c for c, x in zip(self.cycles, self.cycle_vars) if x.X > 0.5]
        return chosen, []


    



        



        

