# 🧬 Genetic Algorithm — Travelling Salesman Problem (TSP)

A Python implementation of the **Genetic Algorithm** applied to the classic **Travelling Salesman Problem**, finding the shortest route between Egyptian cities.

---

## 📌 Problem

Find the **shortest possible route** that visits each city exactly once and returns to the starting city.

**Cities used:**
| City | Index |
|------|-------|
| Benha | 0 (Start/End) |
| Shebin Al-Kom | 1 |
| Al-Zagazig | 2 |
| Al-Mansoura | 3 |

**Distance Matrix (km):**

|  | Benha | Shebin | Zagazig | Mansoura |
|--|-------|--------|---------|----------|
| **Benha** | 0 | 25 | 42 | 79 |
| **Shebin** | 25 | 0 | 59 | 91 |
| **Zagazig** | 42 | 59 | 0 | 63 |
| **Mansoura** | 79 | 91 | 63 | 0 |

---

## ⚙️ Algorithm Parameters

| Parameter | Value |
|-----------|-------|
| Population Size | 50 |
| Crossover Probability | 0.82 |
| Mutation Probability | 0.30 |
| Generations | 30,000 |

---

## 🔁 How It Works

1. **Initialization** — Generate random valid routes as the initial population
2. **Fitness Evaluation** — Shorter distance = higher fitness (`fitness = -distance`)
3. **Selection** — Tournament selection picks the better of two random individuals
4. **Crossover** — Single-point crossover with a repair function to fix duplicate cities
5. **Mutation** — Randomly swap a city with a repair step to maintain validity
6. **Repeat** — Until all generations are complete

---

## 💻 Code

```python
import random

class TSP:
    def __init__(self, population_size, crossover_probability, mutation_probability, num_of_generations):
        self.cities = ["Benha", "Shebin Al-Kom", "Al-Zagazig", "Al-Mansoura"]
        self.distances = [
            [0, 25, 42, 79],
            [25, 0, 59, 91],
            [42, 59, 0, 63],
            [79, 91, 63, 0]
        ]
        self.__population_size = population_size
        self.__crossover_probability = crossover_probability
        self.__mutation_probability = mutation_probability
        self.__num_of_generations = num_of_generations

        self.__population = []
        other_cities = [1, 2, 3]
        for _ in range(self.__population_size):
            path = [0] + random.sample(other_cities, 3) + [0]
            self.__population.append(path)

    def __calculate_fitness(self):
        fitness_values = []
        for path in self.__population:
            total_distance = sum(self.distances[path[i]][path[i+1]] for i in range(len(path)-1))
            fitness_values.append(-total_distance)
        return fitness_values

    def __selection(self, fitness_vals):
        new_population = []
        for _ in range(self.__population_size):
            idx1, idx2 = random.sample(range(self.__population_size), 2)
            winner = self.__population[idx1] if fitness_vals[idx1] >= fitness_vals[idx2] else self.__population[idx2]
            new_population.append(winner.copy())
        return new_population

    def __repair(self, path):
        middle = path[1:-1]
        seen, repaired = set(), []
        for city in middle:
            if city not in seen and city != 0:
                seen.add(city)
                repaired.append(city)
        missing = [c for c in [1, 2, 3] if c not in seen]
        random.shuffle(missing)
        repaired.extend(missing)
        return [0] + repaired[:3] + [0]

    def __crossover(self, parent1, parent2, probability):
        if random.random() >= probability:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, 3)
        o1 = self.__repair(parent1[:point] + parent2[point:])
        o2 = self.__repair(parent2[:point] + parent1[point:])
        return o1, o2

    def __mutation(self, individual, probability):
        ind = individual.copy()
        if random.random() < probability:
            ind[random.randint(1, 3)] = random.randint(1, 3)
            ind = self.__repair(ind)
        return ind

    def __crossover_mutation(self):
        new_population = []
        for i in range(0, self.__population_size, 2):
            p1 = self.__population[i]
            p2 = self.__population[(i + 1) % self.__population_size]
            c1, c2 = self.__crossover(p1, p2, self.__crossover_probability)
            new_population.extend([self.__mutation(c1, self.__mutation_probability),
                                    self.__mutation(c2, self.__mutation_probability)])
        self.__population = new_population[:self.__population_size]

    def solution(self):
        best_fitness, best_path = float('-inf'), None
        for gen in range(self.__num_of_generations):
            fitness_values = self.__calculate_fitness()
            max_idx = fitness_values.index(max(fitness_values))
            if fitness_values[max_idx] > best_fitness:
                best_fitness = fitness_values[max_idx]
                best_path = self.__population[max_idx]
                print(f"\rGen {gen:5d} | Best distance: {-best_fitness:4.0f} km", end="")
            self.__population = self.__selection(fitness_values)
            self.__crossover_mutation()
        print()
        return best_path, -best_fitness

    def display_path(self, path, distance):
        print("\nBest Route Found:")
        print(" → ".join(self.cities[i] for i in path))
        print(f"Total Distance: {distance} KM")


# Run
problem = TSP(
    population_size=50,
    crossover_probability=0.82,
    mutation_probability=0.30,
    num_of_generations=30000
)
best_path, best_distance = problem.solution()
problem.display_path(best_path, best_distance)
```

---

## 📤 Sample Output

```
Gen     0 | Best distance:  221 km

Best Route Found:
Benha → Al-Zagazig → Al-Mansoura → Shebin Al-Kom → Benha
Total Distance: 221 KM
```

---

## 🛠️ Requirements

```bash
pip install random  # built-in, no install needed
```

> Just run with **Python 3.x** — no external libraries required.

---

## 📚 Concepts Used

- `Genetic Algorithm (GA)`
- `Tournament Selection`
- `Single-Point Crossover`
- `Mutation with Repair`
- `Travelling Salesman Problem (TSP)`
