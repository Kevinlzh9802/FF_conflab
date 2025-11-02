import math
import os
from typing import List, Tuple

import numpy as np


def _dbg_enabled() -> bool:
    return os.environ.get("GCFF_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _dbg_print(prefix: str, msg: str):
    if _dbg_enabled():
        print(f"[graphcut.{prefix}] {msg}")


def _dbg_detail() -> bool:
    return os.environ.get("GCFF_DEBUG_DETAIL", "").lower() in {"1", "true", "yes", "on"}


# class _Dinic:
#     def __init__(self):
#         self.n = 0
#         self.adj = []  # list of lists of edge indices
#         self.to = []
#         self.cap = []
#         self.rev = []  # index of reverse edge

#     def add_node(self, count: int = 1) -> int:
#         start = self.n
#         for _ in range(count):
#             self.adj.append([])
#             self.n += 1
#         return start

#     def _add_edge_dir(self, u: int, v: int, c: float):
#         self.adj[u].append(len(self.to))
#         self.to.append(v)
#         self.cap.append(float(c))
#         self.rev.append(len(self.to) + 1)  # reverse will be next append

#         self.adj[v].append(len(self.to))
#         self.to.append(u)
#         self.cap.append(0.0)
#         self.rev.append(len(self.to) - 2)

#     def add_edge(self, u: int, v: int, c_uv: float, c_vu: float):
#         self._add_edge_dir(u, v, c_uv)
#         self._add_edge_dir(v, u, c_vu)

#     def maxflow(self, s: int, t: int) -> float:
#         flow = 0.0
#         while True:
#             level = [-1] * self.n
#             q = [s]
#             level[s] = 0
#             for u in q:
#                 for ei in self.adj[u]:
#                     v = self.to[ei]
#                     if self.cap[ei] > 1e-12 and level[v] < 0:
#                         level[v] = level[u] + 1
#                         q.append(v)
#             if level[t] < 0:
#                 break
#             it = [0] * self.n

#             def dfs(u: int, f: float) -> float:
#                 if u == t:
#                     return f
#                 i = it[u]
#                 while i < len(self.adj[u]):
#                     ei = self.adj[u][i]
#                     v = self.to[ei]
#                     if self.cap[ei] > 1e-12 and level[u] + 1 == level[v]:
#                         d = dfs(v, min(f, self.cap[ei]))
#                         if d > 1e-12:
#                             self.cap[ei] -= d
#                             rev_i = self.rev[ei]
#                             self.cap[rev_i] += d
#                             return d
#                     i += 1
#                     it[u] = i
#                 return 0.0

#             while True:
#                 pushed = dfs(s, float('inf'))
#                 if pushed <= 1e-12:
#                     break
#                 flow += pushed
#         return flow

#     def reachable_from(self, s: int) -> List[bool]:
#         vis = [False] * self.n
#         stack = [s]
#         vis[s] = True
#         while stack:
#             u = stack.pop()
#             for ei in self.adj[u]:
#                 if self.cap[ei] > 1e-12:
#                     v = self.to[ei]
#                     if not vis[v]:
#                         vis[v] = True
#                         stack.append(v)
#         return vis


# def _select_backend() -> str:
#     b = os.environ.get("GCFF_MAXFLOW", "").lower()
#     if b.startswith("bk"):
#         return "bk"
#     return "dinic"


class _BKGraph:
    def __init__(self):
        try:
            import maxflow  # type: ignore
        except Exception as e:
            raise RuntimeError("BK backend requested but 'maxflow' package is not installed") from e
        self._maxflow = maxflow
        self.g = maxflow.Graph[float]()
        self.n = 0

    def add_node(self, count: int = 1) -> int:
        start = self.n
        self.g.add_nodes(int(count))
        self.n += int(count)
        return start

    def add_edge(self, i: int, j: int, w1: float, w2: float):
        if i == j:
            return
        if w1 < 0 or w2 < 0:
            raise ValueError("Edge capacities must be >= 0")
        self.g.add_edge(int(i), int(j), float(w1), float(w2))

    def add_tweights(self, i: int, w_source: float, w_sink: float):
        if w_source < 0 or w_sink < 0:
            raise ValueError("Terminal capacities must be >= 0")
        self.g.add_tedge(int(i), float(w_source), float(w_sink))

    def maxflow(self) -> float:
        return float(self.g.maxflow())

    def what_segment(self, i: int) -> int:
        seg = self.g.get_segment(int(i))
        # maxflow.SINK is 1
        return int(seg)


# class _DinicGraph:
#     def __init__(self):
#         self._g = _Dinic()
#         self.source = self._g.add_node(1)
#         self.sink = self._g.add_node(1)

#     def add_node(self, count: int = 1) -> int:
#         return self._g.add_node(count)

#     def add_edge(self, i: int, j: int, w1: float, w2: float):
#         if i == j:
#             return
#         if w1 < 0 or w2 < 0:
#             raise ValueError("Edge capacities must be >= 0")
#         self._g.add_edge(i, j, w1, w2)

#     def add_tweights(self, i: int, w_source: float, w_sink: float):
#         if w_source < 0 or w_sink < 0:
#             raise ValueError("Terminal capacities must be >= 0")
#         if w_source > 0:
#             self._g._add_edge_dir(self.source, i, w_source)
#         if w_sink > 0:
#             self._g._add_edge_dir(i, self.sink, w_sink)

#     def maxflow(self) -> float:
#         return self._g.maxflow(self.source, self.sink)

#     def what_segment(self, i: int) -> int:
#         reach = self._g.reachable_from(self.source)
#         return 1 if not reach[i] else 0

#     def reset(self):
#         self.__init__()


class _Graph:
    SINK = 1

    def __init__(self, max_nodes_hint: int = 0):
        self._g = _BKGraph()
        # self._backend = _select_backend()
        # if self._backend == "bk":
        #     try:
        #         self._g = _BKGraph()
        #         if _dbg_enabled():
        #             _dbg_print("backend", "using BK (pymaxflow)")
        #     except RuntimeError:
        #         # fallback to dinic
        #         self._backend = "dinic"
        #         self._g = _DinicGraph()
        #         print("warning: BK not available, falling back to Dinic")
        # else:
        #     self._g = _DinicGraph()
        #     if _dbg_enabled():
        #         _dbg_print("backend", "using Dinic")

    def reset(self):
        self.__init__()

    def add_node(self, count: int = 1) -> int:
        return self._g.add_node(count)

    def add_edge(self, i: int, j: int, w1: float, w2: float):
        self._g.add_edge(i, j, w1, w2)

    def add_tweights(self, i: int, w_source: float, w_sink: float):
        self._g.add_tweights(i, w_source, w_sink)

    def maxflow(self) -> float:
        return self._g.maxflow()

    def what_segment(self, i: int, which) -> int:
        # return 1 for sink, 0 for source
        return self._g.what_segment(i)


class Hypothesis:
    def __init__(self, nodes: int, maxn_overlap: int, hyp: int, lam: float, hypweight: np.ndarray, maxn_pair: int):
        self.max_neigh_p = int(maxn_pair)
        self.max_neigh_m = int(maxn_overlap)
        self.nodes = int(nodes)
        self.hyp = int(hyp)
        self.label = np.zeros(self.nodes, dtype=np.int64)
        self.un = None  # to be assigned: shape (nodes, hyp)
        self.lambda_ = float(lam)
        self.hypweight = np.asarray(hypweight, dtype=float).reshape(-1)
        self.neigh_p = np.full((self.nodes, self.max_neigh_p), -1, dtype=np.int64) if self.max_neigh_p > 0 else np.zeros((self.nodes, 0), dtype=np.int64)
        self.neigh_m = np.full((self.nodes, self.max_neigh_m), -1, dtype=np.int64) if self.max_neigh_m > 0 else np.zeros((self.nodes, 0), dtype=np.int64)
        self._ncost = None  # to be assigned: shape (nodes, max_neigh_p)
        self.g = _Graph()
        self._base0 = None  # base for data nodes
        self._base1 = None  # base for auxiliary nodes allocated in construct_multi

    def _abs_index(self, idx: int) -> int:
        # Map C++-style indices to absolute indices in our graph
        # [0 .. nodes-1]          -> base0 + idx
        # [nodes .. 2*nodes-1]    -> base1 + (idx - nodes)
        if 0 <= idx < self.nodes:
            return (self._base0 if self._base0 is not None else 0) + idx
        if self.nodes <= idx < 2 * self.nodes and self._base1 is not None:
            return self._base1 + (idx - self.nodes)
        return idx

    def add_tweights(self, i: int, w1: float, w2: float, _id: str):
        if (w1 == 0) and (w2 == 0):
            return
        self.g.add_tweights(self._abs_index(i), float(w1), float(w2))

    def add_edge(self, i: int, j: int, w1: float, w2: float, _id: str):
        if i == j:
            return
        self.g.add_edge(self._abs_index(i), self._abs_index(j), float(w1), float(w2))

    def unary(self, node: int, lab: int) -> float:
        return float(self.un[node, lab])

    def hweight(self, lab: int) -> float:
        return float(self.hypweight[lab])

    def overlap_neighbour(self, n: int, n2: int) -> int:
        if self.max_neigh_m == 0:
            return -1
        return int(self.neigh_m[n, n2])

    def pair_neighbour(self, n: int, n2: int) -> int:
        if self.max_neigh_p == 0:
            return -1
        return int(self.neigh_p[n, n2])

    def ncost(self, n: int, n2: int) -> float:
        if self._ncost is None or self.max_neigh_p == 0:
            return 0.0
        return float(self._ncost[n, n2])

    def construct_multi(self, alpha: int):
        # Allocate auxiliary nodes block as in C++ (g->add_node(nodes))
        self._base1 = self.g.add_node(self.nodes)
        for i in range(self.nodes):
            if self.label[i] != alpha:
                nalpha = False
                for j in range(self.max_neigh_m):
                    nb = self.overlap_neighbour(i, j)
                    if nb == -1:
                        break
                    nalpha = nalpha or (self.label[nb] == alpha)
                if not nalpha:
                    # costs for taking alpha
                    self.add_tweights(self.nodes + i, self.unary(i, alpha) * self.lambda_, 0.0, 'd')
                    self.add_edge(i, self.nodes + i, 0.0, self.unary(i, alpha) * self.lambda_, 'a')
                    for j in range(self.max_neigh_m):
                        nb = self.overlap_neighbour(i, j)
                        if nb == -1:
                            break
                        self.add_edge(nb, self.nodes + i, 0.0, self.lambda_ * self.unary(i, alpha), 'b')

        nclass = np.zeros(self.hyp, dtype=bool)
        for i in range(self.nodes):
            nclass[:] = False
            for j in range(self.max_neigh_m):
                nb = self.overlap_neighbour(i, j)
                if nb == -1:
                    break
                nclass[self.label[nb]] = True
            nclass[self.label[i]] = True

            for j in range(self.hyp):
                if nclass[j] and (j != alpha):
                    temp = self.g.add_node(1)
                    self.add_tweights(temp, 0.0, self.unary(i, j) * self.lambda_, 'e')
                    for k in range(self.max_neigh_m):
                        nb = self.overlap_neighbour(i, k)
                        if nb == -1:
                            break
                        if self.label[nb] == j:
                            self.add_edge(nb, temp, self.unary(i, j) * self.lambda_, 0.0, 'c')
                    if self.label[i] == j:
                        self.add_edge(i, temp, self.unary(i, j) * self.lambda_, 0.0, 'd')
                        self.add_tweights(i, self.unary(i, alpha) * (1.0 - self.lambda_), self.unary(i, j) * (1.0 - self.lambda_), 'b')

    def construct_mdl_boykov(self, alpha: int):
        nclass = np.zeros(self.hyp, dtype=bool)
        for i in range(self.nodes):
            nclass[self.label[i]] = True
        temp_nodes = 0
        edge_count = 0
        min_w = float('inf')
        max_w = 0.0
        sum_w = 0.0
        for i in range(self.hyp):
            if (i != alpha) and nclass[i] and self.hweight(i) > 0:
                temp = self.g.add_node(1)
                w = self.hweight(i)
                self.add_tweights(temp, 0.0, w, 'z')
                temp_nodes += 1
                min_w = min(min_w, w)
                max_w = max(max_w, w)
                sum_w += w
                for j in range(self.nodes):
                    if self.label[j] == i:
                        self.add_edge(j, temp, w, 0.0, 'l')
                        edge_count += 1
        if _dbg_enabled():
            if temp_nodes == 0:
                _dbg_print("mdl", f"alpha={alpha} MDL: temps=0 edges=0")
            else:
                _dbg_print("mdl", f"alpha={alpha} MDL: temps={temp_nodes} edges={edge_count} w[min={min_w:.6g}, max={max_w:.6g}, sum={sum_w:.6g}]")

    def construct_pairwise(self, alpha: int):
        pos_cnt = 0
        neg_cnt = 0
        for i in range(self.nodes):
            if self.label[i] != alpha:
                cost = (1.0 - self.lambda_) * (self.unary(i, self.label[i]) - self.unary(i, alpha))
                for j in range(self.max_neigh_p):
                    nb = self.pair_neighbour(i, j)
                    if nb == -1:
                        break
                    if self.label[nb] == alpha:
                        cost += self.ncost(i, j)
                    else:
                        if self.label[nb] == self.label[i]:
                            self.add_edge(i, nb, self.ncost(i, j), self.ncost(i, j), 'a')
                        else:
                            cost += self.ncost(i, j)
                            self.add_edge(i, nb, 0.0, self.ncost(i, j), 'a')
                if self.label[i] != alpha:
                    # add unary backward
                    self.add_tweights(i, max(-cost, 0.0), max(cost, 0.0), 'z')
                    if cost >= 0:
                        pos_cnt += 1
                    else:
                        neg_cnt += 1
        if _dbg_enabled():
            _dbg_print("pairwise", f"alpha={alpha} unary-back counts: pos={pos_cnt} neg={neg_cnt}")

    def expand(self, alpha: int):
        # Optional pre-cost debug
        if _dbg_enabled():
            try:
                _dbg_print("solve", f"before alpha={alpha} cost={self.cost():.6g}")
            except Exception:
                pass
        self.g.reset()
        self._base0 = self.g.add_node(self.nodes)
        self._base1 = None
        if self.max_neigh_m > 0:
            self.construct_multi(alpha)
        self.construct_pairwise(alpha)
        self.construct_mdl_boykov(alpha)
        self.g.maxflow()
        # update labels: nodes at sink side take alpha
        flips = 0
        flipped_idx = []
        for i in range(self.nodes):
            sink_side = self.g.what_segment(self._abs_index(i), _Graph.SINK) == _Graph.SINK
            if self.label[i] != alpha and sink_side:
                self.label[i] = alpha
                flips += 1
                if _dbg_detail():
                    flipped_idx.append(i)
        if _dbg_enabled():
            try:
                _dbg_print("expand", f"alpha={alpha} flips={flips} cost={self.cost():.6g}")
                if _dbg_detail() and flipped_idx:
                    _dbg_print("expand", f"flipped idx (first 20): {flipped_idx[:20]}")
            except Exception:
                pass

    def annotate(self, thresh: float) -> np.ndarray:
        out = np.zeros((self.nodes, self.hyp), dtype=float)
        for i in range(self.nodes):
            for j in range(self.max_neigh_m):
                nb = self.overlap_neighbour(i, j)
                if nb == -1:
                    break
                out[i, self.label[nb]] = max(out[i, self.label[nb]], self.lambda_)
            out[i, self.label[i]] = 1.0
        for j in range(self.hyp):
            for i in range(self.nodes):
                if self.unary(i, j) == thresh:
                    out[i, j] = 0.0
        return out

    def cost(self) -> float:
        cost = 0.0
        lweight = np.zeros(self.hyp, dtype=float)
        for i in range(self.nodes):
            lweight[:] = 0
            for j in range(self.max_neigh_m):
                nb = self.overlap_neighbour(i, j)
                if nb == -1:
                    break
                lweight[self.label[nb]] = self.lambda_
            lweight[self.label[i]] = 1.0
            for j in range(self.hyp):
                if lweight[j] != 0:
                    cost += lweight[j] * self.unary(i, j)

        for i in range(self.nodes):
            for j in range(self.max_neigh_p):
                nb = self.pair_neighbour(i, j)
                if nb == -1:
                    break
                if self.label[i] != self.label[nb]:
                    cost += self.ncost(i, j)

        lcost = np.zeros(self.hyp, dtype=int)
        for i in range(self.nodes):
            lcost[self.label[i]] = 1
        for i in range(self.hyp):
            cost += lcost[i] * self.hweight(i)
        return cost

    def fast_solve(self):
        c = self.cost()
        c2 = math.inf
        oldlabel = self.label.copy()
        i = 0
        while c2 > c or math.isinf(c):
            c2 = c
            for j in range(self.hyp):
                self.expand(j)
                temp = self.cost()
                if temp > c + 1e-9:
                    # revert if cost increases
                    self.label[:] = oldlabel
                    temp = c
                else:
                    if temp < c - 1e-9:
                        i = 0
                    oldlabel[:] = self.label
                    c = temp
                i += 1
                if i == self.hyp:
                    c2 = c
                    return

    def solve(self):
        self.fast_solve()


def _check_nonneg(name: str, arr: np.ndarray):
    if np.any(arr < 0):
        raise ValueError(f"{name} must be elementwise >= 0")


def expand(unary: np.ndarray,
           neigh_pair: np.ndarray,
           pair_costs: np.ndarray,
           current_labels: np.ndarray,
           hypweight: np.ndarray,
           thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    unary = np.asarray(unary, dtype=float)
    if _dbg_enabled():
        _dbg_print("expand", f"unary shape={unary.shape} dtype={unary.dtype} min={np.min(unary):.6g} max={np.max(unary):.6g}")
    points, hyp = unary.shape
    neigh_pair = np.asarray(neigh_pair, dtype=float)
    maxn = neigh_pair.shape[0]
    if neigh_pair.shape[1] != points:
        raise ValueError("neigh_pair must have shape (maxn, points)")
    pair_costs = np.asarray(pair_costs, dtype=float)
    if pair_costs.shape != (maxn, points):
        raise ValueError("pair_costs must have same shape as neigh_pair")
    current_labels = np.asarray(current_labels, dtype=float).reshape(-1)
    if current_labels.shape[0] != points:
        raise ValueError("current_labels length mismatch")
    hypweight = np.asarray(hypweight, dtype=float).reshape(-1)
    if hypweight.shape[0] != hyp:
        raise ValueError("hypweight length must equal number of labels (hyp)")

    _check_nonneg("unary", unary)
    _check_nonneg("pair_costs", pair_costs)
    _check_nonneg("hypweight", hypweight)

    H = Hypothesis(points, 0, hyp, 0.0, hypweight, maxn)
    H.un = np.minimum(unary, float(thresh))
    H._ncost = pair_costs.T.copy()  # shape (points, maxn)
    H.label = np.asarray(current_labels, dtype=np.int64).copy()
    # neigh_pair provided in MATLAB is 1-based, may include zeros for -1; expect Python 0-based inputs
    # Here we accept 1-based neighbors and convert to 0-based with -1 for 0
    n_p = np.asarray(neigh_pair, dtype=np.int64).T  # to (points, maxn)
    n_p = n_p - 1
    if _dbg_enabled():
        _dbg_print(
            "expand",
            f"neigh_pair shape={neigh_pair.shape} maxn={maxn} points={points} min={int(np.min(neigh_pair)) if neigh_pair.size else 'NA'} max={int(np.max(neigh_pair)) if neigh_pair.size else 'NA'}",
        )
        _dbg_print(
            "expand",
            f"converted neigh_p stats: min={int(np.min(n_p)) if n_p.size else 'NA'} max={int(np.max(n_p)) if n_p.size else 'NA'} count(-1)={(n_p==-1).sum() if n_p.size else 0}")
        _dbg_print(
            "expand",
            f"pair_costs min={np.min(pair_costs):.6g} max={np.max(pair_costs):.6g} hypweight min={np.min(hypweight):.6g} max={np.max(hypweight):.6g} thresh={thresh}",
        )
    H.neigh_p = n_p
    if _dbg_enabled():
        _dbg_print("expand", f"initial labels unique={np.unique(H.label)[:10]}")
    H.solve()
    if _dbg_enabled():
        _dbg_print("expand", f"final labels unique={np.unique(H.label)[:10]}")
    annot = H.annotate(float(thresh))
    return annot, H.label.copy()


def multi(unary: np.ndarray,
          neigh_overlap: np.ndarray,
          interior_labels: np.ndarray,
          lam: float,
          hypweight: np.ndarray,
          thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    unary = np.asarray(unary, dtype=float)
    points, hyp = unary.shape
    neigh_overlap = np.asarray(neigh_overlap, dtype=float)
    maxn = neigh_overlap.shape[0]
    if _dbg_enabled():
        _dbg_print("multi", f"unary shape={unary.shape} maxn={maxn} thresh={thresh} lambda={lam}")
    if neigh_overlap.shape[1] != points:
        raise ValueError("neigh_overlap must have shape (maxn, points)")
    interior_labels = np.asarray(interior_labels, dtype=float).reshape(-1)
    if interior_labels.shape[0] != points:
        raise ValueError("interior_labels length mismatch")
    hypweight = np.asarray(hypweight, dtype=float).reshape(-1)
    if hypweight.shape[0] != hyp:
        raise ValueError("hypweight length must equal number of labels (hyp)")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda must be in [0,1]")

    _check_nonneg("unary", unary)
    _check_nonneg("hypweight", hypweight)

    H = Hypothesis(points, maxn, hyp, float(lam), hypweight, 0)
    H.un = np.minimum(unary, float(thresh))
    H.label = np.asarray(interior_labels, dtype=np.int64).copy()
    n_m = np.asarray(neigh_overlap, dtype=np.int64).T  # (points, maxn)
    n_m = n_m - 1
    H.neigh_m = n_m
    H.solve()
    annot = H.annotate(float(thresh))
    if _dbg_enabled():
        _dbg_print("expand", f"annot sum={float(np.sum(annot)):.6g}")
    return annot, H.label.copy()


def allgc(unary: np.ndarray,
          neigh_overlap: np.ndarray,
          neigh_pair: np.ndarray,
          pair_costs: np.ndarray,
          current_labels: np.ndarray,
          lam: float,
          hypweight: np.ndarray,
          thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unary = np.asarray(unary, dtype=float)
    points, hyp = unary.shape
    if _dbg_enabled():
        _dbg_print("allgc", f"unary shape={unary.shape} lam={lam} thresh={thresh}")
    neigh_overlap = np.asarray(neigh_overlap, dtype=float)
    maxn_m = neigh_overlap.shape[0]
    if neigh_overlap.shape[1] != points:
        raise ValueError("neigh_overlap must have shape (maxn, points)")
    neigh_pair = np.asarray(neigh_pair, dtype=float)
    maxn_p = neigh_pair.shape[0]
    if neigh_pair.shape[1] != points:
        raise ValueError("neigh_pair must have shape (maxn, points)")
    pair_costs = np.asarray(pair_costs, dtype=float)
    if pair_costs.shape != (maxn_p, points):
        raise ValueError("pair_costs must have same shape as neigh_pair")
    current_labels = np.asarray(current_labels, dtype=float).reshape(-1)
    if current_labels.shape[0] != points:
        raise ValueError("current_labels length mismatch")
    hypweight = np.asarray(hypweight, dtype=float).reshape(-1)
    if hypweight.shape[0] != hyp:
        raise ValueError("hypweight length must equal number of labels (hyp)")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda must be in [0,1]")

    _check_nonneg("unary", unary)
    _check_nonneg("pair_costs", pair_costs)
    _check_nonneg("hypweight", hypweight)

    H = Hypothesis(points, maxn_m, hyp, float(lam), hypweight, maxn_p)
    H.un = np.minimum(unary, float(thresh))
    H._ncost = pair_costs.T.copy()
    H.label = np.asarray(current_labels, dtype=np.int64).copy()
    H.neigh_m = np.asarray(neigh_overlap, dtype=np.int64).T - 1
    H.neigh_p = np.asarray(neigh_pair, dtype=np.int64).T - 1
    if _dbg_enabled():
        _dbg_print("allgc", f"neigh_m: maxn={maxn_m} min={int(np.min(neigh_overlap)) if neigh_overlap.size else 'NA'} max={int(np.max(neigh_overlap)) if neigh_overlap.size else 'NA'}")
        _dbg_print("allgc", f"neigh_p: maxn={maxn_p} min={int(np.min(neigh_pair)) if neigh_pair.size else 'NA'} max={int(np.max(neigh_pair)) if neigh_pair.size else 'NA'}")
    H.solve()
    annot = H.annotate(float(thresh))
    # For compatibility with MEX, return a third matrix for "outliers" even if zeros
    outliers = np.zeros_like(annot)
    return annot, H.label.copy(), outliers


# Convenience wrapper matching the call seen in vis_group.py (unary, neigh, weight, mdl, seg)
def expand_simple(unary: np.ndarray,
                  neigh: np.ndarray,
                  weight: np.ndarray,
                  mdl: np.ndarray,
                  seg: np.ndarray,
                  thresh: float = float('inf')) -> Tuple[np.ndarray, np.ndarray]:
    return expand(unary, neigh, weight, seg, mdl, thresh)

