# 🐜 Ant Colony Optimization — Quadratic Assignment Problem (QAP)

A Python implementation of **Ant Colony Optimization (ACO)** applied to the **Quadratic Assignment Problem**, finding the optimal assignment of facilities to locations.

---

## 📌 Problem

Find the **optimal assignment** of facilities to locations that minimizes total cost based on flow between facilities and distance between locations.

**Objective Function:**
```
Cost = ∑i ∑j Distance(i, j) × Flow(facility_i, facility_j)
```

**Flow Matrix** — interaction between facilities:

|  | F0 | F1 | F2 | F3 | F4 |
|--|----|----|----|----|-----|
| **F0** | 0 | 11 | 3 | 5 | 1 |
| **F1** | 11 | 0 | 7 | 6 | 5 |
| **F2** | 3 | 7 | 0 | 2 | 8 |
| **F3** | 5 | 6 | 2 | 0 | 1 |
| **F4** | 1 | 5 | 8 | 1 | 0 |

**Distance Matrix** — physical distance between locations:

|  | L0 | L1 | L2 | L3 | L4 |
|--|----|----|----|----|-----|
| **L0** | 0 | 3 | 5 | 3 | 2 |
| **L1** | 3 | 0 | 25 | 1 | 8 |
| **L2** | 5 | 25 | 0 | 13 | 9 |
| **L3** | 3 | 1 | 13 | 0 | 7 |
| **L4** | 2 | 8 | 9 | 7 | 0 |

---

## ⚙️ Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size | 30 | Number of ants per iteration |
| Iterations | 2000 | Number of ACO cycles |
| Q | 3 | Pheromone deposit constant |
| Rho (ρ) | 0.26 | Evaporation rate |
| Alpha (α) | 1.2 | Pheromone influence weight |
| Beta (β) | 0.8 | Heuristic cost influence weight |

---

## 🔁 How It Works

1. **Initialization** — Place each ant at a random starting facility; set pheromone matrix to 1.0
2. **Transition** — Each ant picks the next facility using pheromone strength × cost heuristic
3. **Fitness Evaluation** — Compute total cost `∑ Distance × Flow` for each ant's assignment
4. **Pheromone Update** — Evaporate all trails, then reinforce the best ant's path
5. **Repeat** — Track global best across all iterations

---

## 💻 Code

```python
import numpy as np
import random


class Quadratic:
    def __init__(self, dim, flow_matrix, dist_matrix, pop_size, no_of_iterations, Q, rho, beta, alpha):
        # --- Problem Data ---
        self.__flow_matrix = flow_matrix
        self.__dist_matrix = dist_matrix
        self.__facilities  = list(range(dim))
        self.__dim         = dim

        self.__initialize_parameters(pop_size, no_of_iterations, Q, rho, beta, alpha)

    # ------------------------------------------------------------------ #
    #  Parameter Setup                                                    #
    # ------------------------------------------------------------------ #
    def __initialize_parameters(self, pop_size, no_of_iterations, Q, rho, beta, alpha):
        self.__pop_size          = pop_size
        self.__no_of_iterations  = no_of_iterations
        self.__Q                 = Q
        self.__rho               = rho
        self.__beta              = beta
        self.__alpha             = alpha
        self.__pheromone         = np.full((self.__dim, self.__dim), fill_value=1.0)

    # ------------------------------------------------------------------ #
    #  STEP 1 — Initialize Ants                                           #
    # ------------------------------------------------------------------ #
    def __starting_ants(self):
        """Place each ant at a random starting facility."""
        population = []
        for _ in range(self.__pop_size):
            ant = [random.choice(self.__facilities)]
            population.append(ant)
        return population

    # ------------------------------------------------------------------ #
    #  STEP 2 — Transition (Probabilistic Next Facility)                  #
    # ------------------------------------------------------------------ #
    def __transition(self, ant):
        """Pick next facility using pheromone strength and cost heuristic."""
        probabilities = np.zeros(self.__dim)

        for facility in self.__facilities:
            if facility in ant:
                continue
            cost = sum(
                self.__dist_matrix[loc][len(ant)] * self.__flow_matrix[ant[loc]][facility]
                for loc in range(len(ant))
            )
            tau  = self.__pheromone[ant[-1], facility] ** self.__alpha
            eta  = (1 / (cost + 1)) ** self.__beta
            probabilities[facility] = tau * eta

        probabilities /= probabilities.sum()
        return [np.random.choice(self.__facilities, p=probabilities).item()]

    # ------------------------------------------------------------------ #
    #  STEP 3 — Fitness Evaluation                                        #
    # ------------------------------------------------------------------ #
    def __calculate_fitness(self, ant):
        """Compute total assignment cost: ∑i ∑j Dist(i,j) × Flow(fi, fj)."""
        return sum(
            self.__dist_matrix[i][j] * self.__flow_matrix[ant[i]][ant[j]]
            for i in range(len(ant))
            for j in range(len(ant))
        )

    # ------------------------------------------------------------------ #
    #  STEP 4 — Pheromone Update                                          #
    # ------------------------------------------------------------------ #
    def __update_pheromone(self, best_ant, fitness, count):
        """Evaporate trails then reinforce the best ant's path."""
        self.__pheromone *= (1 - self.__rho)
        for i in range(len(best_ant) - 1):
            self.__pheromone[best_ant[i], best_ant[i + 1]] += (self.__Q / fitness) * count

    # ------------------------------------------------------------------ #
    #  Main Loop — Run ACO                                                #
    # ------------------------------------------------------------------ #
    def solution(self):
        """Evolve ants across iterations and return best assignment found."""
        best_fitness_overall = None
        best_solution        = None

        for iteration in range(self.__no_of_iterations):
            population = self.__starting_ants()

            for ant in population:
                for _ in range(self.__dim - 1):
                    ant.extend(self.__transition(ant))

            fitness_values = [self.__calculate_fitness(ant) for ant in population]

            best_index  = fitness_values.index(min(fitness_values))
            best_ant    = population[best_index]
            best_fitness = fitness_values[best_index]
            best_counts = population.count(best_ant)

            self.__update_pheromone(best_ant, best_fitness, best_counts)

            if best_fitness_overall is None or best_fitness < best_fitness_overall:
                best_fitness_overall = best_fitness
                best_solution        = best_ant
                print(f"\rA solution found in iteration: {iteration} with fitness = {best_fitness_overall}")

        return best_solution, best_fitness_overall


# ------------------------------------------------------------------ #
#  Problem Data                                                       #
# ------------------------------------------------------------------ #
flow_matrix = np.array([
    [0,  11,  3,  5,  1],
    [11,  0,  7,  6,  5],
    [3,   7,  0,  2,  8],
    [5,   6,  2,  0,  1],
    [1,   5,  8,  1,  0]
])

dist_matrix = np.array([
    [0,   3,  5,  3,  2],   # Loc 0
    [3,   0, 25,  1,  8],   # Loc 1
    [5,  25,  0, 13,  9],   # Loc 2
    [3,   1, 13,  0,  7],   # Loc 3
    [2,   8,  9,  7,  0]    # Loc 4
])

# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #
problem = Quadratic(
    dim              = 5,
    flow_matrix      = flow_matrix,
    dist_matrix      = dist_matrix,
    pop_size         = 30,
    no_of_iterations = 2000,
    Q                = 3,
    rho              = 0.26,
    beta             = 0.8,
    alpha            = 1.2
)

best_solution, best_cost = problem.solution()
assignment = [int(x) for x in best_solution]

print("\n" + "=" * 45)
print("  ✅ Best Assignment Found")
print("=" * 45)
print(f"  Assignment : {assignment}")
print(f"  Total Cost : {best_cost}")
print("=" * 45)
```

---

## 📤 Sample Output

```
A solution found in iteration: 0 with fitness = 470
A solution found in iteration: 1 with fitness = 442

=============================================
  ✅ Best Assignment Found
=============================================
  Assignment : [1, 4, 3, 2, 0]
  Total Cost : 442
=============================================
```

---

## 🛠️ Requirements

```bash
pip install numpy
```

> Run with **Python 3.x** — only NumPy required.

---

## 📚 Concepts Used

- `Ant Colony Optimization (ACO)`
- `Quadratic Assignment Problem (QAP)`
- `Pheromone Evaporation & Reinforcement`
- `Probabilistic Transition Rule`
- `Heuristic Cost Function`
