"""
Adaptive Variable Neighborhood Search (AVNS) for an Electric Vehicle/Truck Routing Problem (EVRP/ETRP)

This scripts implements the AVNS algorithm following the algorithmic description of  Erdem 2022:
- Greedy construction for initial solution S
- Guided shaking phase over neighborhood structures Nk (selected by roulette wheel weights)
- Local search intensification to produce S''
- Neighborhood change + SA-like acceptance exp(-(c(S'')-c(S))/T)
- Adaptive operator weights (scoring-based) + dynamic penalty factors for constraint violations

IMPORTANT (per your request):
- This does NOT treat ALNS-style "destroy+repair" as neighborhoods.
- Neighborhoods are classic VNS/VRP operators (swap/relocate/2-opt + charging-station operators).


"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# ----------------------------- PARAMETERS ------------------------------------
# =============================================================================

# EVRP parameters
R = 45.0        # max SoC
r = 5.0         # min SoC
Q = 28000.0     # capacity
alpha = 0.5     # SoC gain per unit time (consistent with Delta units)
beta0 = 2.0     # distance cost factor
beta1 = 0.6     # time/service cost factor
beta2 = 0.8     # charging time cost factor
Delta_min = 5.0
Delta_max = 120.0
K_fleet = 6
M = 1000.0
###############################################################################

OMEGA_MAX = 4000          # max iterations
K_MAX = 8                 # number of neighborhood structures Nk
EPSILON = 0.95          # T <- T * EPSILON
SIGMA_PCT = 0.30    # T0 chosen so sigma% deterioration accepted w/ prob
RNG_SEED = 42

# Search controls
LOCAL_SEARCH_TRIES = 150 # # per iteration
SHAKING_TRIES = 100 # attempts to generate a neighbor in shaking
ALLOW_INFEASIBLE = True

# Dynamic penalty settings (start values)
LAM_CAP_INIT = 1.0
LAM_SOC_INIT = 1.0
LAM_CHG_INIT = 1.0

LAM_MIN = 1e-3
LAM_MAX = 1e6

PEN_INCREASE = 1.05  # multiply when violations persist
PEN_DECREASE = 0.95   # multiply when no violations (relax)
###############################################################################


# =============================================================================
# --------------------------- DATA LOADING ------------------------------------
# =============================================================================

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

# adjust this to match your layout
base_dir = script_dir.parent.parent / "synthetic_instance" / "instance_3"

def read_csv(filename: str) -> pd.DataFrame:
    fp = base_dir / filename
    if not fp.exists():
        avail = ", ".join(p.name for p in sorted(base_dir.glob("*.csv")))
        raise FileNotFoundError(f"Missing `{fp}`. Available CSVs: {avail or 'none'}")
    return pd.read_csv(fp)

complete_nodes = read_csv("complete_nodes.csv")
waste_demand = read_csv("waste_demand.csv")
service_time_df = read_csv("case1_service_time.csv")
complete_travel_data = read_csv("complete_travel_data.csv")

# Process nodes
complete_nodes.set_index("Node ID", inplace=True)
complete_nodes.index = complete_nodes.index.map(int)

node_list: List[int] = complete_nodes.index.tolist()
node_types: Dict[int, str] = complete_nodes["Label"].to_dict()

depot_idx: int = [int(k) for k, v in node_types.items() if v == "D"][0]

chargers_on = [n for n in node_list if node_types[n] == "C+"]
chargers_off = [n for n in node_list if node_types[n] == "C-"]
charging_pairs: List[Tuple[int, int]] = list(zip(chargers_on, chargers_off))
charging_pair_set = set(charging_pairs)

# Demand and service time
q: Dict[int, float] = waste_demand.set_index("Node")["Demand"].rename(int).to_dict()
Tserv: Dict[int, float] = service_time_df.set_index("Node")["Service Time"].rename(int).to_dict()

# Distance and time matrices
dist_matrix: Dict[Tuple[int, int], float] = {}
time_matrix: Dict[Tuple[int, int], float] = {}

for _, row in complete_travel_data.iterrows():
    i = int(row["from_node ID"])
    j = int(row["to_node ID"])
    d = row["Distance (miles)"]
    t = row["Time (minutes)"]

    if pd.notna(d) and pd.notna(t):
        # keep your disallowed arc rule
        if (i == depot_idx and j in chargers_off) or (i in chargers_on and j == depot_idx):
            continue
        dist_matrix[(i, j)] = float(d)
        time_matrix[(i, j)] = float(t)

# =============================================================================
# ----------------------------- HELPERS ---------------------------------------
# =============================================================================

def is_customer(n: int) -> bool:
    return node_types.get(n) == "V"

def is_cplus(n: int) -> bool:
    return node_types.get(n) == "C+"

def is_cminus(n: int) -> bool:
    return node_types.get(n) == "C-"

def compute_charge_time(charge_gain: float) -> float:
    return charge_gain / alpha

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def plot_routes(state: "EVRPState", title: str = "AVNS EVRP"):
    plt.figure(figsize=(8, 6))
    for route in state.routes:
        coords = [complete_nodes.loc[n, ["X Coordinate", "Y Coordinate"]].values for n in route]
        xs, ys = zip(*coords)
        plt.plot(xs, ys, marker="o")
        for node, x, y in zip(route, xs, ys):
            plt.text(x, y + 0.15, str(node), fontsize=12, ha="center", va="bottom")
    plt.title(f"{title} | cost={state.objective():.1f} | penalized={state.penalized_cost():.1f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
# ----------------------------- STATE -----------------------------------------
# =============================================================================

@dataclass
class PenaltyWeights:
    lam_cap: float = LAM_CAP_INIT
    lam_soc: float = LAM_SOC_INIT
    lam_chg: float = LAM_CHG_INIT

class EVRPState:
    """
    Route-based state:
      - routes: list of routes; each is a list of node IDs and must start/end at depot.
      - charge_gains: per route, a dict mapping index_of_Cplus_in_route -> SoC gain charged at that station.
        Convention: route contains ... C+ at position p, C- at position p+1, and the charge_gain is applied at C-.
      - socs: recomputed SoC values at each node (for diagnostics). Not required to be perfectly aligned after edits;
        use normalize() to recompute.
    """
    def __init__(
        self,
        routes: List[List[int]],
        charge_gains: Optional[List[Dict[int, float]]] = None,
        socs: Optional[List[List[float]]] = None,
        pen: Optional[PenaltyWeights] = None
    ):
        self.routes = [[int(x) for x in rt] for rt in routes]
        self.charge_gains = charge_gains if charge_gains is not None else [dict() for _ in self.routes]
        self.socs = socs if socs is not None else [[] for _ in self.routes]
        self.pen = copy.deepcopy(pen) if pen is not None else PenaltyWeights()

    def copy(self) -> "EVRPState":
        return EVRPState(
            routes=copy.deepcopy(self.routes),
            charge_gains=copy.deepcopy(self.charge_gains),
            socs=copy.deepcopy(self.socs),
            pen=self.pen
        )

    # ----------------- objective and constraint evaluation -----------------

    def route_load(self, rt: List[int]) -> float:
        return sum(q.get(n, 0.0) for n in rt if is_customer(n))

    def objective(self) -> float:
        """
        True objective (no penalties):
          beta0 * distance + beta1 * (travel time + service time) + beta2 * (charging time)
        Charging time is derived from charge_gains.
        """
        total = 0.0
        for k, rt in enumerate(self.routes):
            total += self._route_objective(rt, self.charge_gains[k])
        return total

    def _route_objective(self, rt: List[int], cg: Dict[int, float]) -> float:
        cost = 0.0
        # travel + service
        for i in range(len(rt) - 1):
            u, v = rt[i], rt[i + 1]
            if (u, v) not in dist_matrix:
                return float("inf")
            cost += beta0 * dist_matrix[(u, v)] + beta1 * time_matrix[(u, v)]
            if is_customer(u):
                cost += beta1 * Tserv.get(u, 0.0)

        # charging cost
        for pos_cplus, gain in cg.items():
            # charging duration bounds are enforced via penalties/feasibility; objective counts the time cost
            cost += beta2 * compute_charge_time(gain)

        return cost

    def violations(self) -> Tuple[float, float, float, float]:
        """
        Returns (cap_violation, soc_violation, chg_time_violation, arc_violation_count)
        cap_violation: sum over routes of max(0, load - Q)
        soc_violation: sum of battery shortfall below r along routes (integral-ish)
        chg_time_violation: sum of charging time bounds violations
        arc_violation_count: number of missing arcs
        """
        cap_v = 0.0
        soc_v = 0.0
        chg_v = 0.0
        arc_miss = 0.0

        for k, rt in enumerate(self.routes):
            # capacity
            cap_v += max(0.0, self.route_load(rt) - Q)

            # arc existence + SoC simulation
            socs, soc_short, arc_bad = simulate_soc(rt, self.charge_gains[k])
            if socs is None:
                arc_miss += arc_bad
            else:
                arc_miss += arc_bad
                soc_v += soc_short

            # charging time bounds
            for pos_cplus, gain in self.charge_gains[k].items():
                t = compute_charge_time(gain)
                if t < Delta_min:
                    chg_v += (Delta_min - t)
                elif t > Delta_max:
                    chg_v += (t - Delta_max)

        return cap_v, soc_v, chg_v, arc_miss

    def penalized_cost(self) -> float:
        """
        Penalized cost used for acceptance and operator scoring.
        """
        obj = self.objective()
        cap_v, soc_v, chg_v, arc_miss = self.violations()

        # hard arc-missing penalty
        if arc_miss > 0:
            return float("inf")

        return (
            obj
            + self.pen.lam_cap * cap_v
            + self.pen.lam_soc * soc_v
            + self.pen.lam_chg * chg_v
        )

    def feasible(self) -> bool:
        cap_v, soc_v, chg_v, arc_miss = self.violations()
        return (cap_v <= 1e-9) and (soc_v <= 1e-9) and (chg_v <= 1e-9) and (arc_miss == 0)

    def normalize(self) -> Optional["EVRPState"]:
        """
        Recompute SoC arrays and clean obvious charger inconsistencies.
        If ALLOW_INFEASIBLE is False, returns None if infeasible.
        """
        # clean degenerate routes
        new_routes = []
        new_cg = []

        for k, rt in enumerate(self.routes):
            rt2, cg2 = cleanup_route_and_chargers(rt, self.charge_gains[k])
            if len(rt2) <= 2:
                continue
            new_routes.append(rt2)
            new_cg.append(cg2)

        self.routes = new_routes
        self.charge_gains = new_cg
        self.socs = []

        # recompute socs
        for k, rt in enumerate(self.routes):
            socs, soc_short, arc_bad = simulate_soc(rt, self.charge_gains[k])
            if socs is None:
                if not ALLOW_INFEASIBLE:
                    return None
                self.socs.append([])
            else:
                self.socs.append(socs)
                if (not ALLOW_INFEASIBLE) and (soc_short > 1e-9):
                    return None

        if (not ALLOW_INFEASIBLE) and (not self.feasible()):
            return None
        return self

# =============================================================================
# ----------------------- SOC SIMULATION + CLEANUP ----------------------------
# =============================================================================

def simulate_soc(route: List[int], cg: Dict[int, float]) -> Tuple[Optional[List[float]], float, int]:
    """
    Simulates SoC along the route with charging modeled by (C+ at pos p, C- at pos p+1).
    Charge gain at p is applied after reaching C-.

    Returns:
      (socs, total_soc_shortfall_below_r, missing_arc_count)
      soc_shortfall sums max(0, r - soc_at_node) at each step.
    """
    socs: List[float] = [R]
    soc = R
    soc_short = 0.0
    arc_bad = 0

    i = 0
    while i < len(route) - 1:
        u, v = route[i], route[i + 1]
        if (u, v) not in dist_matrix:
            arc_bad += 1
            return None, soc_short, arc_bad

        # drive
        soc -= dist_matrix[(u, v)]
        socs.append(soc)

        if soc < r:
            soc_short += (r - soc)

        # if v is C+ and next is C-, apply charge at that pair
        if is_cplus(v):
            if i + 2 >= len(route):
                # dangling
                i += 1
                continue
            v2 = route[i + 2]
            if not is_cminus(v2):
                i += 1
                continue
            # ensure pair is valid
            if (v, v2) not in charging_pair_set:
                i += 1
                continue
            # travel v->v2 might exist (often ~0); apply it
            if (v, v2) in dist_matrix:
                soc -= dist_matrix[(v, v2)]
                socs.append(soc)
                if soc < r:
                    soc_short += (r - soc)

            gain = cg.get(i + 1, 0.0)  # pos of C+ is i+1
            # clamp gain physically to not exceed R
            gain = clamp(gain, 0.0, max(0.0, R - soc))
            soc += gain
            # overwrite last SoC (at C-) after charging
            socs[-1] = soc

        i += 1

    return socs, soc_short, arc_bad

def cleanup_route_and_chargers(route: List[int], cg: Dict[int, float]) -> Tuple[List[int], Dict[int, float]]:
    """
    Removes inconsistent charger nodes and fixes cg indexing.
    Keeps only patterns: ... -> C+ -> C- -> ...
    and stores charge gain keyed by the index position of C+ in the cleaned route.
    """
    if not route:
        return route, {}

    cleaned = [route[0]]
    new_cg: Dict[int, float] = {}

    i = 1
    while i < len(route) - 1:
        n = route[i]
        nxt = route[i + 1] if (i + 1) < len(route) else None

        if is_cplus(n):
            # must be followed by C-
            if nxt is not None and is_cminus(nxt) and (n, nxt) in charging_pair_set:
                pos_cplus = len(cleaned)  # index in cleaned where C+ will be placed
                cleaned.append(n)
                cleaned.append(nxt)

                # map old cg: old position of this C+ in original route is i
                gain = cg.get(i, 0.0)
                new_cg[pos_cplus] = gain

                i += 2
                continue
            # drop dangling C+
            i += 1
            continue

        if is_cminus(n):
            # drop orphan C-
            i += 1
            continue

        cleaned.append(n)
        i += 1

    cleaned.append(route[-1])
    return cleaned, new_cg

# =============================================================================
# -------------------------- INITIAL SOLUTION ---------------------------------
# =============================================================================

def nearest_neighbor_init() -> EVRPState:
    """
    Greedy construction (problem-specific), similar spirit to your NN build,
    but now supports partial charging via charge_gains.
    """
    unvisited = [n for n in node_list if is_customer(n)]

    routes: List[List[int]] = []
    gains: List[Dict[int, float]] = []

    while unvisited:
        rt = [depot_idx]
        cg: Dict[int, float] = {}
        load = 0.0
        soc = R

        while unvisited:
            cur = rt[-1]
            feasible = [(c, dist_matrix[(cur, c)]) for c in unvisited if (cur, c) in dist_matrix]
            if not feasible:
                break
            cust = min(feasible, key=lambda x: x[1])[0]

            if load + q.get(cust, 0.0) > Q:
                break

            d = dist_matrix[(cur, cust)]
            if soc - d >= r - 1e-6:
                rt.append(cust)
                soc -= d
                load += q.get(cust, 0.0)
                unvisited.remove(cust)
                continue

            # need charging: insert best station pair that can get us to cust
            best = None
            best_cost = None

            for c_on, c_off in charging_pairs:
                if (cur, c_on) not in dist_matrix or (c_off, cust) not in dist_matrix:
                    continue
                d1 = dist_matrix[(cur, c_on)]
                if soc - d1 < r - 1e-6:
                    continue

                # choose minimum charge gain needed to reach cust safely
                soc_at_cplus = soc - d1
                need = dist_matrix[(c_off, cust)] + r - soc_at_cplus
                gain = clamp(need, 0.0, R - soc_at_cplus)

                t_ch = compute_charge_time(gain)
                if t_ch < Delta_min or t_ch > Delta_max:
                    continue

                # approximate insertion cost
                d2 = dist_matrix.get((c_on, c_off), 0.0)
                d3 = dist_matrix[(c_off, cust)]
                t1 = time_matrix[(cur, c_on)]
                t2 = time_matrix.get((c_on, c_off), 0.0)
                t3 = time_matrix[(c_off, cust)]

                ins_cost = beta0 * (d1 + d2 + d3) + beta1 * (t1 + t2 + t3) + beta2 * t_ch
                if best_cost is None or ins_cost < best_cost:
                    best_cost = ins_cost
                    best = (c_on, c_off, gain)

            if best is None:
                break

            c_on, c_off, gain = best
            # insert ... cur -> C+ -> C- -> cust
            rt.append(c_on)
            pos_cplus = len(rt) - 1
            cg[pos_cplus] = gain
            rt.append(c_off)
            rt.append(cust)

            # update soc approximately
            soc = soc - dist_matrix[(cur, c_on)]  # arrive at C+
            soc += gain                           # after charging at C-
            soc -= dist_matrix[(c_off, cust)]     # reach customer

            load += q.get(cust, 0.0)
            unvisited.remove(cust)

        # close route back to depot if possible
        last = rt[-1]
        if last != depot_idx and (last, depot_idx) in dist_matrix:
            d_back = dist_matrix[(last, depot_idx)]
            if soc - d_back >= r - 1e-6:
                rt.append(depot_idx)
            else:
                # attempt to insert charging before depot
                inserted = False
                for c_on, c_off in charging_pairs:
                    if (last, c_on) not in dist_matrix or (c_off, depot_idx) not in dist_matrix:
                        continue
                    d1 = dist_matrix[(last, c_on)]
                    if soc - d1 < r - 1e-6:
                        continue
                    soc_at_cplus = soc - d1
                    need = dist_matrix[(c_off, depot_idx)] + r - soc_at_cplus
                    gain = clamp(need, 0.0, R - soc_at_cplus)
                    t_ch = compute_charge_time(gain)
                    if t_ch < Delta_min or t_ch > Delta_max:
                        continue
                    rt.append(c_on)
                    pos_cplus = len(rt) - 1
                    cg[pos_cplus] = gain
                    rt.append(c_off)
                    rt.append(depot_idx)
                    inserted = True
                    break
                if not inserted:
                    # allow infeasible return only if ALLOW_INFEASIBLE
                    rt.append(depot_idx)

        routes.append(rt)
        gains.append(cg)

    S = EVRPState(routes, gains).normalize()
    if S is None:
        # if strict feasible required, fallback to trivial per-customer routes
        routes = []
        gains = []
        for c in [n for n in node_list if is_customer(n)]:
            if (depot_idx, c) in dist_matrix and (c, depot_idx) in dist_matrix:
                routes.append([depot_idx, c, depot_idx])
                gains.append({})
        S = EVRPState(routes, gains).normalize()
    return S if S is not None else EVRPState(routes, gains)

# =============================================================================
# ---------------------- NEIGHBORHOOD OPERATORS Nk ----------------------------
# =============================================================================
# AVNS paper: shaking uses a neighborhood Nk; local search intensifies.
# We provide a set of classic VRP neighborhoods + charging-station ops.

def get_customer_positions(state: EVRPState) -> List[Tuple[int, int, int]]:
    pos = []
    for ri, rt in enumerate(state.routes):
        for idx, n in enumerate(rt):
            if is_customer(n):
                pos.append((ri, idx, n))
    return pos

def swap_vertical(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Inter-route swap: choose two customers on possibly different routes and swap."""
    s = state.copy()
    pos = get_customer_positions(s)
    if len(pos) < 2:
        return None
    a = pos[rng.integers(0, len(pos))]
    b = pos[rng.integers(0, len(pos))]
    if a == b:
        return None
    (r1, i1, _), (r2, i2, _) = a, b
    s.routes[r1][i1], s.routes[r2][i2] = s.routes[r2][i2], s.routes[r1][i1]
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def swap_horizontal(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Intra-route swap: swap two customers within one route."""
    s = state.copy()
    if not s.routes:
        return None
    ri = rng.integers(0, len(s.routes))
    rt = s.routes[ri]
    cust_idx = [i for i, n in enumerate(rt) if is_customer(n)]
    if len(cust_idx) < 2:
        return None
    i1, i2 = rng.choice(cust_idx, size=2, replace=False)
    rt[i1], rt[i2] = rt[i2], rt[i1]
    s.routes[ri] = rt
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def relocate_vertical(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Inter-route relocate: take one customer and insert into another route at random position."""
    s = state.copy()
    pos = get_customer_positions(s)
    if not pos or len(s.routes) < 1:
        return None
    r_from, i_from, cust = pos[rng.integers(0, len(pos))]
    s.routes[r_from].pop(i_from)

    # remove empty/degenerate route later in normalize
    if not s.routes:
        return None

    # choose target route (could be same; avoid trivial)
    r_to = rng.integers(0, len(s.routes))
    rt_to = s.routes[r_to]
    ins = rng.integers(1, len(rt_to))  # not at 0
    rt_to.insert(ins, cust)
    s.routes[r_to] = rt_to
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def relocate_horizontal(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Intra-route relocate: move one customer to another position in same route."""
    s = state.copy()
    if not s.routes:
        return None
    ri = rng.integers(0, len(s.routes))
    rt = s.routes[ri]
    cust_idx = [i for i, n in enumerate(rt) if is_customer(n)]
    if len(cust_idx) < 2:
        return None
    i_from = int(rng.choice(cust_idx))
    cust = rt.pop(i_from)
    i_to = rng.integers(1, len(rt))
    rt.insert(i_to, cust)
    s.routes[ri] = rt
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def two_opt_intra(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Classic 2-opt within a route (reverse segment)."""
    s = state.copy()
    if not s.routes:
        return None
    ri = rng.integers(0, len(s.routes))
    rt = s.routes[ri]
    if len(rt) < 6:
        return None
    i = rng.integers(1, len(rt) - 3)
    j = rng.integers(i + 1, len(rt) - 1)
    rt2 = rt[:i] + list(reversed(rt[i:j])) + rt[j:]
    s.routes[ri] = rt2
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def insert_station_near_violation(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """
    Insert a charging station pair (C+,C-) in a route at a place likely to help SoC feasibility.
    We pick a random route, scan for the first arc where SoC would drop below r, and insert a station before it.
    """
    s = state.copy()
    if not s.routes:
        return None

    ri = rng.integers(0, len(s.routes))
    rt = s.routes[ri]
    cg = s.charge_gains[ri]

    socs, _, arc_bad = simulate_soc(rt, cg)
    if socs is None or arc_bad > 0:
        return None

    # find first index where soc after arriving is below r (needs help)
    bad_pos = None
    soc = R
    for i in range(len(rt) - 1):
        u, v = rt[i], rt[i + 1]
        if (u, v) not in dist_matrix:
            return None
        soc -= dist_matrix[(u, v)]
        if soc < r - 1e-6:
            bad_pos = i + 1  # insert before v, i.e. between u and v
            break

        # apply charging if u->... pattern; we handle via simulate_soc but keep simple here
        if is_cplus(v) and i + 2 < len(rt) and is_cminus(rt[i + 2]) and (v, rt[i + 2]) in charging_pair_set:
            # jump handled by normalize later
            pass

    if bad_pos is None:
        # still allow random insert for diversification
        bad_pos = rng.integers(1, len(rt))

    pre = rt[bad_pos - 1]
    post = rt[bad_pos]

    # try a few stations and pick one that provides a feasible-ish gain
    best = None
    best_cost = None

    for _ in range(25):
        c_on, c_off = charging_pairs[rng.integers(0, len(charging_pairs))]
        if (pre, c_on) not in dist_matrix or (c_off, post) not in dist_matrix:
            continue

        # estimate soc at pre using simulation
        soc_pre = socs[bad_pos - 1] if (bad_pos - 1) < len(socs) else None
        if soc_pre is None:
            continue

        # can we reach C+?
        if soc_pre - dist_matrix[(pre, c_on)] < r - 1e-6:
            continue

        soc_at_cplus = soc_pre - dist_matrix[(pre, c_on)]
        need = dist_matrix[(c_off, post)] + r - soc_at_cplus
        gain = clamp(need, 0.0, R - soc_at_cplus)
        t_ch = compute_charge_time(gain)
        if t_ch < Delta_min or t_ch > Delta_max:
            continue

        candidate = s.copy()
        # insert C+,C-
        candidate.routes[ri] = rt[:bad_pos] + [c_on, c_off] + rt[bad_pos:]
        # shift old cg keys that are >= bad_pos
        candidate.charge_gains[ri] = shift_charge_gains(cg, insert_at=bad_pos, shift_by=2)
        # add gain at new C+ position
        new_pos_cplus = bad_pos
        candidate.charge_gains[ri][new_pos_cplus] = gain

        cand_norm = candidate.normalize()
        if cand_norm is None and not ALLOW_INFEASIBLE:
            continue
        cand_cost = (cand_norm.penalized_cost() if cand_norm is not None else candidate.penalized_cost())
        if best_cost is None or cand_cost < best_cost:
            best_cost = cand_cost
            best = cand_norm if cand_norm is not None else candidate

    return best

def remove_station(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Remove one random charging pair occurrence from a route."""
    s = state.copy()
    occurrences = []
    for ri, rt in enumerate(s.routes):
        for i in range(len(rt) - 1):
            if is_cplus(rt[i]) and is_cminus(rt[i + 1]) and (rt[i], rt[i + 1]) in charging_pair_set:
                occurrences.append((ri, i))
    if not occurrences:
        return None
    ri, i = occurrences[rng.integers(0, len(occurrences))]
    rt = s.routes[ri]
    cg = s.charge_gains[ri]

    # remove nodes i and i+1
    rt2 = rt[:i] + rt[i + 2:]
    cg2 = remove_charge_gain_at(cg, pos_cplus=i)
    cg2 = shift_charge_gains(cg2, insert_at=i, shift_by=-2)

    s.routes[ri] = rt2
    s.charge_gains[ri] = cg2
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def exchange_station(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """
    Replace an existing charging station pair with another station pair.
    Keep the gain but clamp as needed after normalize.
    """
    s = state.copy()
    occ = []
    for ri, rt in enumerate(s.routes):
        for i in range(len(rt) - 1):
            if is_cplus(rt[i]) and is_cminus(rt[i + 1]) and (rt[i], rt[i + 1]) in charging_pair_set:
                occ.append((ri, i))
    if not occ:
        return None

    ri, i = occ[rng.integers(0, len(occ))]
    rt = s.routes[ri]
    old_gain = s.charge_gains[ri].get(i, 0.0)

    new_on, new_off = charging_pairs[rng.integers(0, len(charging_pairs))]
    rt[i] = new_on
    rt[i + 1] = new_off
    if (new_on, new_off) not in charging_pair_set:
        return None

    s.routes[ri] = rt
    # keep gain at same pos
    s.charge_gains[ri][i] = old_gain
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def adjust_charge_amount(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """
    Modify charge gain at one station (like "remove charger amount" or adjust to reduce time violations).
    """
    s = state.copy()
    stations = []
    for ri, rt in enumerate(s.routes):
        for i in range(len(rt) - 1):
            if is_cplus(rt[i]) and is_cminus(rt[i + 1]) and (rt[i], rt[i + 1]) in charging_pair_set:
                stations.append((ri, i))
    if not stations:
        return None

    ri, pos = stations[rng.integers(0, len(stations))]
    gain = s.charge_gains[ri].get(pos, 0.0)

    # perturb gain
    # random factor in [0.5, 1.5]
    factor = rng.uniform(0.5, 1.5)
    new_gain = max(0.0, gain * factor)

    # also ensure the corresponding time can be moved toward feasibility
    t = compute_charge_time(new_gain)
    if t < Delta_min:
        # lift to min time
        new_gain = Delta_min * alpha
    elif t > Delta_max:
        # reduce to max time
        new_gain = Delta_max * alpha

    s.charge_gains[ri][pos] = new_gain
    return s.normalize() if s.normalize() is not None else (s if ALLOW_INFEASIBLE else None)

def swap_routes(state: EVRPState, rng: np.random.Generator) -> Optional[EVRPState]:
    """Swap the order of two routes (mostly neutral, but can interact with penalties/repair logic)."""
    s = state.copy()
    if len(s.routes) < 2:
        return None
    i, j = rng.choice(len(s.routes), size=2, replace=False)
    s.routes[i], s.routes[j] = s.routes[j], s.routes[i]
    s.charge_gains[i], s.charge_gains[j] = s.charge_gains[j], s.charge_gains[i]
    s.socs[i], s.socs[j] = s.socs[j], s.socs[i]
    return s

# Helper for charge gain index shifting/removal
def shift_charge_gains(cg: Dict[int, float], insert_at: int, shift_by: int) -> Dict[int, float]:
    out = {}
    for pos, gain in cg.items():
        if pos >= insert_at:
            out[pos + shift_by] = gain
        else:
            out[pos] = gain
    return out

def remove_charge_gain_at(cg: Dict[int, float], pos_cplus: int) -> Dict[int, float]:
    return {p: g for p, g in cg.items() if p != pos_cplus}

# List of neighborhoods Nk (shaking operators)
NEIGHBORHOODS = [
    swap_vertical,
    swap_horizontal,
    relocate_vertical,
    relocate_horizontal,
    two_opt_intra,
    insert_station_near_violation,
    remove_station,
    exchange_station,
    # (optional extras)
    # adjust_charge_amount,  # enable if you want more charging decision variation
    # swap_routes,
]

# =============================================================================
# ------------------------------ LOCAL SEARCH ---------------------------------
# =============================================================================

def local_search(state: EVRPState, rng: np.random.Generator, tries: int = LOCAL_SEARCH_TRIES) -> EVRPState:
    """
    Intensification:
    Apply a sequence of improving moves. This is kept simple but matches AVNS idea:
    - explore known operators and keep the best improvements
    - allow infeasible during the walk (penalized cost)
    """
    cur = state.copy()
    best = cur.copy()
    best_pc = best.penalized_cost()

    for _ in range(tries):
        op = NEIGHBORHOODS[rng.integers(0, len(NEIGHBORHOODS))]
        cand = op(cur, rng)
        if cand is None:
            continue

        cand_pc = cand.penalized_cost()
        if cand_pc < best_pc - 1e-9:
            best = cand
            best_pc = cand_pc

        # take first-improvement step sometimes to diversify
        if cand_pc < cur.penalized_cost() - 1e-9:
            cur = cand

    return best

# =============================================================================
# ------------------------------ AVNS CORE ------------------------------------
# =============================================================================

@dataclass
class OperatorStat:
    weight: float = 1.0
    score: float = 0.0
    freq: int = 0

def roulette_select(stats: List[OperatorStat], rng: np.random.Generator) -> int:
    w = np.array([max(1e-12, st.weight) for st in stats], dtype=float)
    p = w / w.sum()
    return int(rng.choice(len(stats), p=p))

def initial_temperature(c0: float, sigma_pct: float = SIGMA_PCT) -> float:
    # exp(-(sigma*c0)/T0) = 0.5 => T0 = sigma*c0 / ln(2)
    return (sigma_pct * c0) / math.log(2.0) if c0 > 0 else 1.0

def accept_sa(c_new: float, c_cur: float, T: float, rng: np.random.Generator) -> bool:
    if c_new <= c_cur:
        return True
    if T <= 1e-12:
        return False
    return rng.random() < math.exp(-(c_new - c_cur) / T)

def update_operator_weights(stats: List[OperatorStat], eta: float = 0.8):
    """
    Weight update (paper-style):
      rho_k <- rho_k*(1-eta) + eta*(score_k/freq_k)
    """
    for st in stats:
        if st.freq > 0:
            st.weight = st.weight * (1.0 - eta) + eta * (st.score / st.freq)
        st.score = 0.0
        st.freq = 0

def update_penalties(state: EVRPState) -> None:
    """
    Dynamic penalty mechanism:
    - If constraint violations exist, increase corresponding lambdas
    - If not, decrease slightly
    """
    cap_v, soc_v, chg_v, arc_miss = state.violations()

    if arc_miss > 0:
        # keep infeasible arcs effectively forbidden
        state.pen.lam_cap = clamp(state.pen.lam_cap, LAM_MIN, LAM_MAX)
        state.pen.lam_soc = clamp(state.pen.lam_soc, LAM_MIN, LAM_MAX)
        state.pen.lam_chg = clamp(state.pen.lam_chg, LAM_MIN, LAM_MAX)
        return

    if cap_v > 1e-9:
        state.pen.lam_cap = clamp(state.pen.lam_cap * PEN_INCREASE, LAM_MIN, LAM_MAX)
    else:
        state.pen.lam_cap = clamp(state.pen.lam_cap * PEN_DECREASE, LAM_MIN, LAM_MAX)

    if soc_v > 1e-9:
        state.pen.lam_soc = clamp(state.pen.lam_soc * PEN_INCREASE, LAM_MIN, LAM_MAX)
    else:
        state.pen.lam_soc = clamp(state.pen.lam_soc * PEN_DECREASE, LAM_MIN, LAM_MAX)

    if chg_v > 1e-9:
        state.pen.lam_chg = clamp(state.pen.lam_chg * PEN_INCREASE, LAM_MIN, LAM_MAX)
    else:
        state.pen.lam_chg = clamp(state.pen.lam_chg * PEN_DECREASE, LAM_MIN, LAM_MAX)

# =============================================================================
# ------------------------------ SHAKING --------------------------------------
# =============================================================================

def shaking(S: EVRPState, Nk_idx: int, rng: np.random.Generator) -> Optional[EVRPState]:
    """
    Guided shaking: apply the chosen neighborhood operator.
    We attempt multiple times to get a valid neighbor (especially important when strict feasibility is used).
    """
    op = NEIGHBORHOODS[Nk_idx]
    for _ in range(SHAKING_TRIES):
        cand = op(S, rng)
        if cand is not None:
            return cand
    return None

# =============================================================================
# ------------------------------ RUN AVNS -------------------------------------
# =============================================================================

def run_avns() -> Tuple[EVRPState, EVRPState, Dict[str, float]]:
    rng = np.random.default_rng(RNG_SEED)

    # Initialize operator stats (roulette weights)
    op_stats = [OperatorStat(weight=1.0) for _ in range(len(NEIGHBORHOODS))]

    # Initial solution S
    S = nearest_neighbor_init()
    if S is None:
        raise RuntimeError("Could not build an initial solution. Check data / arc connectivity.")
    S_best_feasible = S.copy() if S.feasible() else None
    S_best_any = S.copy()

    history_best = []
    history_current = []

    # Initial temperature based on initial solution's penalized cost
    Tcur = initial_temperature(S.penalized_cost(), SIGMA_PCT)

    start = time.time()

    omega = 1
    while omega <= OMEGA_MAX:
        k = 1
        while k <= min(K_MAX, len(NEIGHBORHOODS)):
            # select neighborhood by roulette
            idx = roulette_select(op_stats, rng)
            op_stats[idx].freq += 1

            # Shaking => S'
            S_prime = shaking(S, idx, rng)
            if S_prime is None:
                k += 1
                continue

            # Local search => S''
            S_dblprime = local_search(S_prime, rng, tries=LOCAL_SEARCH_TRIES)

            c_cur = S.penalized_cost()
            c_new = S_dblprime.penalized_cost()

            accepted = accept_sa(c_new, c_cur, Tcur, rng)
            if accepted:
                prev_cur = S.copy()
                S = S_dblprime

                # scoring system:
                # 10 if new global best feasible (or best-any if infeasible allowed)
                # 7 if improves current
                # 2 if worse but accepted
                improved_current = c_new < c_cur - 1e-9

                # Update best-any
                if c_new < S_best_any.penalized_cost() - 1e-9:
                    S_best_any = S.copy()

                # Update best-feasible (tracked by true objective, not penalized)
                if S.feasible():
                    if S_best_feasible is None or (S.objective() < S_best_feasible.objective() - 1e-9):
                        S_best_feasible = S.copy()
                        op_stats[idx].score += 10.0
                        k = 1
                    elif improved_current:
                        op_stats[idx].score += 7.0
                        k = 1
                    else:
                        op_stats[idx].score += 2.0
                        k += 1
                else:
                    # infeasible acceptance: still score based on improved current penalized cost
                    if improved_current:
                        op_stats[idx].score += 7.0
                        k = 1
                    else:
                        op_stats[idx].score += 2.0
                        k += 1

                # dynamic penalties update (paper step)
                update_penalties(S)

            else:
                # rejected -> neighborhood change
                k += 1

        # Update weights at end of omega-iteration (paper step)
        update_operator_weights(op_stats, eta=0.8) # eta=0.8

        # Track TRUE objective values (NOT penalized)

        current_cost = S.objective()

        if S_best_feasible is not None:
            best_cost = S_best_feasible.objective()
        else:
            best_cost = S_best_any.objective()

        history_current.append(current_cost)
        history_best.append(best_cost)

        # Optional: print progress
        print(f"Iter {omega}: "f"Current (penalized) = {S.penalized_cost():.2f}, "f"Current (true) = {S.objective():.2f}, "f"Best (true) = {S_best_any.objective():.2f}")

        # cool temperature
        Tcur *= EPSILON
        omega += 1

    end = time.time()

    # Choose final reported solution: prefer best feasible; fallback to best-any
    final_best = S_best_feasible if S_best_feasible is not None else S_best_any

    totals = final_best.compute_totals()
    totals["runtime_sec"] = end - start
    totals["best_feasible_found"] = 1 if (S_best_feasible is not None) else 0
    totals["pen_lam_cap"] = final_best.pen.lam_cap
    totals["pen_lam_soc"] = final_best.pen.lam_soc
    totals["pen_lam_chg"] = final_best.pen.lam_chg

    # return final_best, S_best_any, totals
    return final_best, S_best_any, totals, history_best, history_current

# =============================================================================
# ------------------------------ METRICS --------------------------------------
# =============================================================================

def compute_route_totals(rt: List[int], cg: Dict[int, float]) -> Tuple[float, float, float, float]:
    """
    (distance, travel_time, service_time, charging_time)
    """
    dist = 0.0
    ttime = 0.0
    stime = 0.0
    chtime = 0.0

    for i in range(len(rt) - 1):
        u, v = rt[i], rt[i + 1]
        if (u, v) not in dist_matrix:
            continue
        dist += dist_matrix[(u, v)]
        ttime += time_matrix[(u, v)]
        if is_customer(u):
            stime += Tserv.get(u, 0.0)

    for _, gain in cg.items():
        chtime += compute_charge_time(gain)

    return dist, ttime, stime, chtime

def compute_totals_for_state(state: EVRPState) -> Dict[str, float]:
    total_distance = 0.0
    total_travel_time = 0.0
    total_service_time = 0.0
    total_charging_time = 0.0

    for k, rt in enumerate(state.routes):
        d, tt, st, ct = compute_route_totals(rt, state.charge_gains[k])
        total_distance += d
        total_travel_time += tt
        total_service_time += st
        total_charging_time += ct

    return {
        "distance": total_distance,
        "travel_time": total_travel_time,
        "service_time": total_service_time,
        "charging_time": total_charging_time,
    }

# attach totals method to EVRPState (so output formatting matches your style)
def _compute_totals_method(self: EVRPState) -> Dict[str, float]:
    return compute_totals_for_state(self)


EVRPState.compute_totals = _compute_totals_method  # type: ignore

# =============================================================================
# ------------------------------ MAIN -----------------------------------------
# =============================================================================

if __name__ == "__main__":
    # final_best, best_any, totals = run_avns()
    final_best, best_any, totals, history_best, history_current = run_avns()

    print("\n==================== AVNS RESULTS ====================")
    print(f"ALLOW_INFEASIBLE: {ALLOW_INFEASIBLE}")
    print(f"✅ Best Reported Cost (true objective): ${final_best.objective():.2f}")
    print(f"   Penalized cost used in search       : {final_best.penalized_cost():.2f}")
    print(f"   Feasible?                           : {final_best.feasible()}")

    print("\n📦 Routes:")
    for i, rt in enumerate(final_best.routes):
        print(f"  Route {i+1}: {rt}")

    print("\n🔋 SoC (recomputed):")
    for i, socs in enumerate(final_best.socs):
        if socs:
            print(f"  Route {i+1} SoC: {['{:.1f}'.format(s) for s in socs]}")
        else:
            print(f"  Route {i+1} SoC: [not available]")

    print("\n⚡ Charging decisions (gain at each C+ position):")
    for i, cg in enumerate(final_best.charge_gains):
        if not cg:
            print(f"  Route {i+1}: none")
        else:
            entries = ", ".join([f"(pos {pos}: gain {gain:.1f}, time {compute_charge_time(gain):.1f})" for pos, gain in sorted(cg.items())])
            print(f"  Route {i+1}: {entries}")

    print("\n📊 Totals:")
    print(f"  ➤ Total Distance     : {totals['distance']:.1f} miles")
    print(f"  ➤ Total Travel Time  : {totals['travel_time']:.1f} minutes")
    print(f"  ➤ Total Service Time : {totals['service_time']:.1f} minutes")
    print(f"  ➤ Total Charging Time: {totals['charging_time']:.1f} minutes")
    print(f"\n🕒 Runtime: {totals['runtime_sec']:.2f} seconds")
    print(f"Best feasible found? {bool(totals['best_feasible_found'])}")
    print(f"Final penalties: lam_cap={totals['pen_lam_cap']:.3g}, lam_soc={totals['pen_lam_soc']:.3g}, lam_chg={totals['pen_lam_chg']:.3g}")

    plot_routes(final_best, title="Final AVNS EVRP Route")
