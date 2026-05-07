# 🤖 AI Planning Techniques for Robotics

> **Course:** Planning Techniques for Robotics — 2025/2026
> **Author:** Youssef Emad
> **University:** Benha University — Computer Science, AI Track

A collection of **3 bio-inspired optimization algorithms** implemented in Python, applied to real-world planning and resource allocation problems.

---

## 📁 Repository Structure

```
Planning-Techniques/
│
├── 🧬 Genetic_Algorithm/
│   ├── Genetic_Algorithm_TSP.ipynb
│   └── README.md
│
├── 🐜 Ant_Colony_Optimization/
│   ├── ACO_QAP.ipynb
│   └── README.md
│
├── 🐝 Artificial_Bee_Colony/
│   ├── ABC_Bandwidth.ipynb
│   └── README.md
│
└── README.md  ← you are here
```

---

## 🔬 Algorithms Overview

| # | Algorithm | Problem | Objective |
|---|-----------|---------|-----------|
| 1 | 🧬 Genetic Algorithm (GA) | Travelling Salesman Problem (TSP) | Minimize total travel distance |
| 2 | 🐜 Ant Colony Optimization (ACO) | Quadratic Assignment Problem (QAP) | Minimize total assignment cost |
| 3 | 🐝 Artificial Bee Colony (ABC) | Network Bandwidth Allocation | Maximize weighted user satisfaction |

---

## 🧬 1. Genetic Algorithm — TSP

**Problem:** Find the shortest route visiting 4 Egyptian cities exactly once and returning to start.

**Cities:** Benha → Shebin Al-Kom → Al-Zagazig → Al-Mansoura → Benha

| Parameter | Value |
|-----------|-------|
| Population Size | 50 |
| Crossover Probability | 0.82 |
| Mutation Probability | 0.30 |
| Generations | 30,000 |

**Key Steps:** Initialization → Fitness → Tournament Selection → Single-Point Crossover → Mutation → Repair

**Result:**
```
Benha → Al-Zagazig → Al-Mansoura → Shebin Al-Kom → Benha
Total Distance: 221 KM
```

📂 [`Genetic_Algorithm/`](./Genetic_Algorithm/)

---

## 🐜 2. Ant Colony Optimization — QAP

**Problem:** Assign 5 facilities to 5 locations to minimize total interaction cost `∑ Distance × Flow`.

| Parameter | Value |
|-----------|-------|
| Ants | 30 |
| Iterations | 2000 |
| Alpha (α) | 1.2 |
| Beta (β) | 0.8 |
| Rho (ρ) | 0.26 |
| Q | 3 |

**Key Steps:** Ant Initialization → Probabilistic Transition → Fitness Evaluation → Pheromone Update

**Result:**
```
Assignment : [1, 4, 3, 2, 0]
Total Cost : 442
```

📂 [`Ant_Colony_Optimization/`](./Ant_Colony_Optimization/)

---

## 🐝 3. Artificial Bee Colony — Bandwidth Allocation

**Problem:** Allocate 750 Gbps across 6 clients to maximize weighted log-utility satisfaction `Σ priority_i × ln(BW_i)`.

| Parameter | Value |
|-----------|-------|
| Bees | 30 |
| Iterations | 2000 |
| Limit | 25 |
| Penalty | 10,000 |
| Total BW Pool | 750 Gbps |

**Key Steps:** Employed Phase → Onlooker Phase → Elitism → Scout Phase

**Result:**
```
Total BW Used : 749.99 Gbps
Objective f(x): 81.6458
Constraint satisfied: True
```

📂 [`Artificial_Bee_Colony/`](./Artificial_Bee_Colony/)

---

## 📊 Comparison

| Algorithm | Type | Problem Type | Key Mechanism |
|-----------|------|-------------|---------------|
| GA | Evolutionary | Combinatorial | Crossover + Mutation |
| ACO | Swarm Intelligence | Combinatorial | Pheromone Trails |
| ABC | Swarm Intelligence | Continuous | Employed / Onlooker / Scout |

---

## 🛠️ Requirements

```bash
pip install numpy
```

> All algorithms run with **Python 3.x** — only NumPy required (GA uses built-in `random` only).

---

## 📚 Topics Covered

- `Genetic Algorithm (GA)` · `Ant Colony Optimization (ACO)` · `Artificial Bee Colony (ABC)`
- `Travelling Salesman Problem (TSP)` · `Quadratic Assignment Problem (QAP)`
- `Network Resource Allocation` · `Bio-inspired Optimization`
- `Swarm Intelligence` · `Evolutionary Computation`
