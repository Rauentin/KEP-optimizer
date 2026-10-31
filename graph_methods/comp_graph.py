# wrapping for networkx methods suited to a compatibility graph method

import networkx as nx


class Vertex:
    def __init__(self, id, p_bt, d_bt, perc_pra, ndd):
        self.id = id
        self.p_bt = p_bt
        self.d_bt = d_bt
        self.perc_pra = perc_pra
        self.is_ndd = ndd

    
class Edge:
    def __init__(self, src, tgt, w):
        self.src = src
        self.tgt = tgt  
        self.w = w

class CompGraph:
    def __init__(self):
        self.Graph = nx.DiGraph()
        self.order = []    
        self._index = {}   

    def add_nodes(self, v_list):
        self.Graph.add_nodes_from(
            (v.id, {'p_bt': v.p_bt, 'd_bt': v.d_bt, 'perc_pra': v.perc_pra, 'is_ndd': v.is_ndd})
            for v in v_list
        )
        self._refresh_order()

    def add_edges(self, e_list):
        # adjust tgt and src since they are  Vertex objects
        # networkx would silently create missing endpoints without attributes and outside of _index,
        # so all edges are validated first and nothing is added if one of them is invalid
        l = []
        for e in e_list:
            missing = [v.id for v in (e.src, e.tgt) if v.id not in self.Graph]
            if missing:
                raise ValueError(f"edge ({e.src.id}, {e.tgt.id}) references unknown vertices {missing}")
            l.append((e.src.id, e.tgt.id, {'weight': e.w}))

        self.Graph.add_edges_from(l)

    def _refresh_order(self):
        self.order = sorted(self.Graph.nodes)
        self._index = {v_id: i for i, v_id in enumerate(self.order)}

    def index(self, v):
        # position of v in the canonical order, u counts as smaller than v iff index(u) < index(v)
        return self._index[v]

    def min_vertex(self, vertices):
        # lowest-indexed vertex among `vertices` per the canonical order, i.e. l = min(V(c)) used by EEF/PIEF/HCF
        return min(vertices, key=lambda v: self._index[v])

    def cycle_edges(self, cycle):
        # edges of a cycle given as vertex list, including the closing edge, e.g. [0, 1, 2] -> [(0, 1), (1, 2), (2, 0)]
        return list(zip(cycle, cycle[1:] + cycle[:1]))

    def cycle_weight(self, cycle):
        # sum of edge weights along the cycle, raises ValueError if an edge is not in the graph
        total = 0
        for u, v in self.cycle_edges(cycle):
            if not self.Graph.has_edge(u, v):
                raise ValueError(f"cycle {cycle} uses edge ({u}, {v}) which is not in the graph")
            total += self.Graph[u][v]['weight']
        return total

    def path_edges(self, path):
        # edges of a path given as vertex list, without closing edge, e.g. [0, 1, 2] -> [(0, 1), (1, 2)]
        return list(zip(path, path[1:]))

    def path_weight(self, path):
        # sum of edge weights along the path, raises ValueError if an edge is not in the graph
        total = 0
        for u, v in self.path_edges(path):
            if not self.Graph.has_edge(u, v):
                raise ValueError(f"path {path} uses edge ({u}, {v}) which is not in the graph")
            total += self.Graph[u][v]['weight']
        return total

    def get_paths(self, max_len, min_len=1):
        # returns all simple paths with between min_len and max_len edges, each as a list of vertex ids
        # used for the path constraints of EF and the half-cycles of HCF
        paths = []

        def extend(path, on_path):
            if len(path) - 1 >= min_len:
                paths.append(list(path))
            if len(path) - 1 == max_len:
                return
            for w in self.Graph.successors(path[-1]):
                if w not in on_path:
                    path.append(w)
                    on_path.add(w)
                    extend(path, on_path)
                    path.pop()
                    on_path.remove(w)

        for v in self.order:
            extend([v], {v})
        return paths

    def edges_to_cycles(self, edges):
        # turns chosen edges (u, v) into cycles, each starting at its lowest vertex per the canonical order
        # every vertex needs exactly one chosen incoming and outgoing edge, otherwise ValueError
        succ = {}
        for u, v in edges:
            if u in succ:
                raise ValueError(f"vertex {u} has more than one outgoing edge")
            succ[u] = v
        if len(set(succ.values())) != len(succ):
            raise ValueError("a vertex has more than one incoming edge")

        cycles = []
        seen = set()
        for start in sorted(succ, key=lambda v: self._index[v]):
            if start in seen:
                continue
            cycle = [start]
            seen.add(start)
            v = succ[start]
            while v != start:
                if v not in succ:
                    raise ValueError(f"edges starting at {start} do not close into a cycle")
                cycle.append(v)
                seen.add(v)
                v = succ[v]
            cycles.append(cycle)
        return cycles

    def get_cycles(self, K):
        # returns all simple cycles of length at most K, each as a list of vertex ids
        return list(nx.simple_cycles(self.Graph, length_bound=K))

    # note that the networkx function for getting all simple cycles is sufficient
    def get_chains(self, L):
        # returns a list of chains of length at most L, i.e. list ids corresponding to the vertices in the chain
        # length is counted in edges (transplants), so a chain has at most L + 1 vertices including the ndd
        # vertices cannot repeat and chains must start at vertices with ndds
        ndds = [v for v, is_ndd in self.Graph.nodes(data='is_ndd') if is_ndd]
        non_ndds = [v for v, is_ndd in self.Graph.nodes(data='is_ndd') if not is_ndd]
        chains = []
        for ndd in ndds:
            # search only the pairs plus this ndd, otherwise a chain could pass through a second ndd
            sub = self.Graph.subgraph(non_ndds + [ndd])
            paths = list(nx.all_simple_paths(sub, ndd, non_ndds, cutoff=L))
            chains += paths

        return chains





        
    
