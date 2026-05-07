# 🐝 Artificial Bee Colony — Network Bandwidth Allocation

A Python implementation of the **Artificial Bee Colony (ABC)** algorithm applied to **Network Bandwidth Allocation**, optimizing resource distribution across clients using proportional fairness.

---

## 📌 Problem

Allocate **750 Gbps** of total bandwidth across 6 clients to **maximize total weighted satisfaction**, respecting per-client upper bounds and a global bandwidth constraint.

**Objective Function (Maximization):**
```
f(x) = Σ priority_i × ln(bandwidth_i)
```

**Client Configuration:**

| Client | Tier | Type | Base BW | Max Allowed | Priority |
|--------|------|------|---------|-------------|----------|
| 1 | 2 | Special | 100 | 130.0 Gbps | 3 |
| 2 | 3 | Regular | 25 | 32.5 Gbps | 2 |
| 3 | 1 | Special | 200 | 260.0 Gbps | 4 |
| 4 | 4 | Regular | 10 | 10.0 Gbps | 1 |
| 5 | 1 | Special | 200 | 200.0 Gbps | 4 |
| 6 | 2 | Special | 100 | 122.0 Gbps | 3 |

> **Priority** = `5 - Tier` &nbsp;|&nbsp; **Max BW** = `Base_BW × (1 + Additional%)`

---

## ⚙️ Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Bees | 30 | Colony size (food sources) |
| Iterations | 2000 | Number of ABC cycles |
| Limit | 25 | Trial threshold before scout replaces source |
| Penalty | 10,000 | Penalty for constraint violation |
| Lower Bound | 0.1 Gbps | Minimum allocation per client |
| Upper Bound | 750 Gbps | Total bandwidth pool |

---

## 🔁 How It Works

1. **Initialization** — Generate random feasible solutions (food sources) respecting per-client bounds
2. **Employed Phase** — Each bee explores its source via neighbour generation; greedy selection keeps the better solution *(offline update)*
3. **Onlooker Phase** — Bees select sources probabilistically via roulette wheel; better neighbours replace current source *(online update)*
4. **Elitism** — Best solution found so far is preserved before scout phase
5. **Scout Phase** — If a source's trial counter exceeds the limit, abandon it and generate a new random source
6. **Repeat** — Track global best across all iterations

---

## 💻 Code

```python
import numpy as np
import random
import math

# ================================================================== #
#  Problem Configuration                                              #
# ================================================================== #
TOTAL_BW = 750   # Total bandwidth pool (Gbps)

clients = {
    'Client':     [1, 2, 3, 4, 5, 6],
    'Tier':       [2, 3, 1, 4, 1, 2],
    'Type':       ['Special', 'Regular', 'Special', 'Regular', 'Special', 'Special'],
    'Base_BW':    [100, 25, 200, 10, 200, 100],
    'Additional': [0.30, 0.30, 0.30, 0.0, 0.0, 0.22],
}

# Per-client upper bounds
ub_per_client = [
    min(clients['Base_BW'][i] * (1 + clients['Additional'][i]), TOTAL_BW)
    for i in range(6)
]

# Priority = 5 - Tier
priorities = [5 - t for t in clients['Tier']]

# ================================================================== #
#  Hyperparameters                                                    #
# ================================================================== #
NO_OF_BEES = 30
ITERATIONS = 2000
LIMIT      = 25
PENALTY    = 10_000
LB         = 0.1    # minimum per-client bandwidth (Gbps)


# ================================================================== #
#  ABC Class                                                          #
# ================================================================== #
class networking:

    def __init__(self, users_priority, no_of_bees, iterations, limit,
                 penalty=None, lb=None, ub=None):
        self.__users_priority  = users_priority
        self.__lb              = lb
        self.__ub              = ub
        self.__pen             = penalty
        self.__ub_per_client   = ub_per_client
        self.__initialize_parameters(no_of_bees, iterations, limit)

    # ------------------------------------------------------------------ #
    #  Parameter Setup                                                    #
    # ------------------------------------------------------------------ #
    def __initialize_parameters(self, no_of_bees, iterations, limit):
        self.__no_of_bees = no_of_bees
        self.__iterations = iterations
        self.__limit      = limit
        self.__dimensions = len(self.__users_priority)
        self.__dims       = range(self.__dimensions)

        self.__population = np.array([
            self.__random_solution() for _ in range(self.__no_of_bees)
        ])
        self.__objective_vec = np.array([
            self.__objective_function(x) for x in self.__population
        ])
        penalties = np.array([self.__penalty(x) for x in self.__population])
        self.__fitness_vec = np.array([
            self.__fitness_function((self.__objective_vec[i], penalties[i]))
            for i in range(self.__no_of_bees)
        ])
        self.__trials = np.zeros(self.__no_of_bees)

    # ------------------------------------------------------------------ #
    #  STEP 1 — Random Feasible Solution                                  #
    # ------------------------------------------------------------------ #
    def __random_solution(self):
        """Generate a random solution clamped to per-client and total bounds."""
        sol = np.array([
            np.random.uniform(self.__lb, self.__ub_per_client[d])
            for d in self.__dims
        ])
        if sum(sol) > self.__ub:
            sol = sol * (self.__ub / sum(sol))
        return sol

    # ------------------------------------------------------------------ #
    #  STEP 2 — Objective Function                                        #
    # ------------------------------------------------------------------ #
    def __objective_function(self, var):
        """f(x) = Σ priority_i × ln(bandwidth_i)  — maximise."""
        return sum(self.__users_priority[d] * math.log(var[d]) for d in self.__dims)

    # ------------------------------------------------------------------ #
    #  STEP 3 — Penalty & Fitness                                         #
    # ------------------------------------------------------------------ #
    def __penalty(self, var):
        """Return +1 if total BW exceeds pool, else -1."""
        return 1 if sum(var) > self.__ub else -1

    def __fitness_function(self, obj_fun):
        """
        Fitness for maximisation:
        - f >= 0, no penalty  → fitness = f(x)
        - f >= 0, penalised   → fitness = 1 / (f(x) + penalty)
        - f <  0              → fitness = 1 / (1 + |f(x)|)
        """
        if obj_fun[0] >= 0:
            return 1 / (obj_fun[0] + self.__pen) if obj_fun[1] == 1 else obj_fun[0]
        return 1 / (1 + abs(obj_fun[0]))

    # ------------------------------------------------------------------ #
    #  STEP 4 — Greedy Selection                                          #
    # ------------------------------------------------------------------ #
    def __greedy_selection(self, solutions, objectives, fitness, trial):
        """Keep the better of current vs new solution."""
        if fitness[1] > fitness[0]:
            return solutions[1], objectives[1], fitness[1], 0
        return solutions[0], objectives[0], fitness[0], trial + 1

    # ------------------------------------------------------------------ #
    #  STEP 5 — Neighbour Generation                                      #
    # ------------------------------------------------------------------ #
    def __new_solution(self, current, partner, dim):
        """X_new = X + φ*(X - X_partner), φ ∈ [-1, 1], clamped to bounds."""
        phi  = np.random.uniform(-1, 1)
        Xnew = current[dim] + phi * (current[dim] - partner[dim])
        Xnew = max(self.__lb, min(Xnew, self.__ub_per_client[dim]))
        new  = current.copy()
        new[dim]    = Xnew
        new_obj     = self.__objective_function(new)
        new_pen     = self.__penalty(new)
        new_fit     = self.__fitness_function((new_obj, new_pen))
        return new, new_obj, new_fit

    def __pick_partner(self, current):
        pool = [p for p in self.__population if not np.array_equal(p, current)]
        return random.choice(pool) if pool else random.choice(self.__population)

    # ------------------------------------------------------------------ #
    #  Employed Phase (offline update)                                    #
    # ------------------------------------------------------------------ #
    def __employed_phase(self):
        pop, obj_vec, fit_vec, trials = [], [], [], []
        for ix in range(self.__no_of_bees):
            partner = self.__pick_partner(self.__population[ix])
            dim     = np.random.choice(self.__dims)
            new_sol, new_obj, new_fit = self.__new_solution(
                self.__population[ix], partner, dim
            )
            sol, o, f, t = self.__greedy_selection(
                (self.__population[ix], new_sol),
                (self.__objective_vec[ix], new_obj),
                (self.__fitness_vec[ix],   new_fit),
                self.__trials[ix]
            )
            pop.append(sol); obj_vec.append(o)
            fit_vec.append(f); trials.append(t)
        self.__population    = np.array(pop)
        self.__objective_vec = np.array(obj_vec)
        self.__fitness_vec   = np.array(fit_vec)
        self.__trials        = np.array(trials)

    # ------------------------------------------------------------------ #
    #  Onlooker Phase (online update)                                     #
    # ------------------------------------------------------------------ #
    def __onlooker_phase(self):
        probs = self.__fitness_vec / self.__fitness_vec.sum()
        ix = 0
        for _ in range(self.__no_of_bees):
            while True:
                if np.random.random() <= probs[ix]:
                    partner = self.__pick_partner(self.__population[ix])
                    dim     = np.random.choice(self.__dims)
                    new_sol, new_obj, new_fit = self.__new_solution(
                        self.__population[ix], partner, dim
                    )
                    sol, o, f, t = self.__greedy_selection(
                        (self.__population[ix], new_sol),
                        (self.__objective_vec[ix], new_obj),
                        (self.__fitness_vec[ix],   new_fit),
                        self.__trials[ix]
                    )
                    self.__population[ix]    = sol
                    self.__objective_vec[ix] = o
                    self.__fitness_vec[ix]   = f
                    self.__trials[ix]        = t
                    ix = (ix + 1) % self.__no_of_bees
                    break
                ix = (ix + 1) % self.__no_of_bees

    # ------------------------------------------------------------------ #
    #  Scout Phase                                                        #
    # ------------------------------------------------------------------ #
    def __scout_phase(self):
        for ix in range(self.__no_of_bees):
            if self.__trials[ix] > self.__limit:
                self.__population[ix]    = self.__random_solution()
                self.__objective_vec[ix] = self.__objective_function(self.__population[ix])
                pen = self.__penalty(self.__population[ix])
                self.__fitness_vec[ix]   = self.__fitness_function(
                    (self.__objective_vec[ix], pen)
                )
                self.__trials[ix] = 0

    # ------------------------------------------------------------------ #
    #  Main Loop                                                          #
    # ------------------------------------------------------------------ #
    def solution(self):
        """Run full ABC: Employed → Onlooker → Elitism → Scout."""
        best_sol, best_obj, best_fit = None, None, None

        for iteration in range(self.__iterations):
            self.__employed_phase()
            self.__onlooker_phase()

            # Elitism
            idx = list(self.__fitness_vec).index(max(self.__fitness_vec))
            if best_fit is None or self.__fitness_vec[idx] > best_fit:
                best_sol = self.__population[idx].copy()
                best_obj = self.__objective_vec[idx]
                best_fit = self.__fitness_vec[idx]

            self.__scout_phase()

            if (iteration + 1) % 200 == 0 or iteration == 0:
                print(f"\rIteration {iteration+1:>5}/{self.__iterations} | "
                      f"Best f(x) = {best_obj:.4f} | "
                      f"Fitness = {best_fit:.4f} | "
                      f"Total BW used = {sum(best_sol):.2f} Gbps", end="")
        print()
        return best_sol, best_obj


# ================================================================== #
#  Run                                                                #
# ================================================================== #
problem = networking(
    users_priority = priorities,
    no_of_bees     = NO_OF_BEES,
    iterations     = ITERATIONS,
    limit          = LIMIT,
    penalty        = PENALTY,
    lb             = LB,
    ub             = TOTAL_BW
)

best_sol, best_obj = problem.solution()

# ================================================================== #
#  Results                                                            #
# ================================================================== #
print("\n" + "=" * 65)
print("                    FINAL RESULTS")
print("=" * 65)
print(f"\n{'Client':<10} {'Priority':>10} {'Max Allowed':>13} {'Allocated BW':>14} {'% of Max':>10}")
print("-" * 60)
for i in range(6):
    pct = (best_sol[i] / ub_per_client[i]) * 100
    print(f"  {i+1:<8} {priorities[i]:>10} {ub_per_client[i]:>12.1f}  {best_sol[i]:>12.2f}   {pct:>8.1f}%")
print("-" * 60)
print(f"  {'TOTAL':<8} {'':>10} {'':>13}  {sum(best_sol):>12.2f}   (pool: {TOTAL_BW} Gbps)")
print(f"\n  Objective f(x) = {best_obj:.6f}")
print(f"  Constraint satisfied: {sum(best_sol) <= TOTAL_BW}")

# Proportional fairness verification
print("\n" + "=" * 65)
print("  Proportional Fairness Verification  (xi = B × wi / Σw)")
print("=" * 65)
print(f"\n{'Client':<10} {'Priority':>10} {'Theoretical':>13} {'ABC Result':>12} {'Diff':>8}")
print("-" * 55)
for i in range(6):
    theoretical = TOTAL_BW * priorities[i] / sum(priorities)
    diff = abs(best_sol[i] - theoretical)
    print(f"  {i+1:<8} {priorities[i]:>10} {theoretical:>12.2f}  {best_sol[i]:>11.2f}  {diff:>7.2f}")
```

---

## 📤 Sample Output

```
Iteration  2000/2000 | Best f(x) = 81.6458 | Fitness = 81.6458 | Total BW used = 749.99 Gbps

=================================================================
                    FINAL RESULTS
=================================================================

Client     Priority   Max Allowed   Allocated BW   % of Max
------------------------------------------------------------
  1               3       130.0        130.00      100.0%
  2               2        32.5         32.50      100.0%
  3               4       260.0        255.49       98.3%
  4               1        10.0         10.00      100.0%
  5               4       200.0        200.00      100.0%
  6               3       122.0        122.00      100.0%
------------------------------------------------------------
  TOTAL                               749.99   (pool: 750 Gbps)

  Objective f(x) = 81.645803
  Constraint satisfied: True
```

---

## 🛠️ Requirements

```bash
pip install numpy
```

> Run with **Python 3.x** — only NumPy and standard library required.

---

## 📚 Concepts Used

- `Artificial Bee Colony (ABC)`
- `Proportional Fairness (log utility)`
- `Employed / Onlooker / Scout Phases`
- `Elitism`
- `Penalty Function for Constraints`
- `Roulette Wheel Selection`
