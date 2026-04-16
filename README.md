# ANVS-Algorithm-Implementation

# Adaptive Variable Neighborhood Search (AVNS) for Electric Vehicle Routing

This repository contains a Python implementation of the Adaptive Variable Neighborhood Search (AVNS) algorithm for solving the Electric Vehicle Routing Problem (EVRP), following the algorithmic description by **Erdem (2022)**.

## Overview

The AVNS algorithm is designed to optimize routing for electric vehicles under capacity, battery, and charging constraints. Key features include:

- **Greedy initial solution construction** using problem-specific heuristics.
- **Guided shaking phase** across multiple neighborhood structures \(N_k\), selected adaptively using roulette-wheel probabilities based on past performance.
- **Local search intensification** to improve the perturbed solution.
- **Simulated annealing acceptance criterion** to allow occasional acceptance of worse solutions:
  \[
  P(\text{accept}) = \exp\Big(-\frac{c(S'') - c(S)}{T}\Big)
  \]
- **Adaptive operator weights and scoring** to prioritize effective neighborhoods.
- **Dynamic penalties** for capacity, state-of-charge (SoC), and charging-time violations.

This implementation follows classical VNS neighborhoods, including swaps, relocations, 2-opt moves, and charging station insertions/removals, rather than ALNS-style destroy/repair.

## Parameter Settings

The AVNS algorithm can be configured using the following key parameters (default values used in our experiments):

| Parameter                  | Description                                                                 | Value       |
|-----------------------------|-----------------------------------------------------------------------------|------------|
| `OMEGA_MAX`                 | Maximum number of iterations                                                 | 4,000      |
| `K_MAX`                     | Number of neighborhood structures                                           | 8          |
| `EPSILON`                   | Cooling rate for simulated annealing                                         | 0.95       |
| `SIGMA_PCT`                 | Exploration probability / system parameter (chance of random neighborhood)  | 0.30       |
| `LOCAL_SEARCH_TRIES`        | Number of local search trials per iteration                                  | 250        |
| `SHAKING_TRIES`             | Number of attempts to generate a valid neighbor during shaking               | 40         |
| `ALLOW_INFEASIBLE`          | Flag allowing temporary infeasible solutions with dynamic penalties          | True       |

Dynamic penalty weights for constraint violations are initialized as follows:  
`LAM_CAP_INIT = LAM_SOC_INIT = LAM_CHG_INIT = 1.0`  

## Running the Algorithm

1. Adjust the file paths for your dataset CSVs (`network_nodes.csv`, `waste_demand_data.csv`, `service_time_data.csv`, `travel_cost_data.csv`) in the script.
2. Run `python run_avns.py` to execute the algorithm.
3. Outputs include:
   - Final routes for each vehicle
   - State-of-charge (SoC) per route
   - Charging decisions and durations
   - Total distance, travel time, service time, and charging time
   - Runtime and dynamic penalties

## References

- Erdem, M. (2022). *Optimization of sustainable urban recycling waste collection and routing with heterogeneous electric vehicles*. Sustainable Cities and Society, 80, 103785.
- Liu, W., Dridi, M., Ren, J., El Hassani, A. H., & Li, S. (2023). *A double-adaptive general variable neighborhood search for an unmanned electric vehicle routing and scheduling problem in green manufacturing systems*. Engineering Applications of Artificial Intelligence, 126, 107113.

---

> ⚠️ Note: This code is intended for research purposes. Users are encouraged to verify data compatibility and parameter choices for their specific instances.
