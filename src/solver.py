# runs a formulation on a graph and collects the solution together with model statistics

import time

import gurobipy as gp
from gurobipy import GRB


def solve(formulation, graph, K, time_limit=None, verbose=False):
    model = gp.Model()
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    start = time.perf_counter()
    formulation.build(model, graph, K)
    build_time = time.perf_counter() - start

    model.optimize()
    cycles, chains = formulation.extract_solution(model) if model.Status == GRB.OPTIMAL else (None, None)

    return {
        "status": model.Status,
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "cycles": cycles,
        "chains": chains,
        "build_time": build_time,
        "solve_time": model.Runtime,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        # keeps the model alive, otherwise the variable values of the formulation are lost
        "model": model,
    }
