# Smoke Screen Decoy Dispensing Strategy Optimization System – Code Documentation

## Project Overview
This project builds a smoke-screen decoy-dispensing-strategy optimizer based on **Differential Evolution (DE)**.  
It addresses the core problem of the 2025 “Higher-Education Press Cup” National College Mathematical-Modeling Contest (Problem A):  
maximize the effective smoke-screen shielding duration on the real target by optimizing the UAV flight parameters (speed, direction) and the decoy parameters (release time, burst delay), thereby confusing incoming missiles.

The system supports strategy optimization for multi-UAV, multi-decoy, multi-missile scenarios.  
Core computations are accelerated with **Numba**, guaranteeing high efficiency even for large-scale simulations.

The architecture is designed for high reusability—by simply editing a few parameters you can solve Questions 1–5.  
Through algorithmic optimization of the elementary unit (single-missile / single-UAV / single-decoy) and heavy use of **NumPy + Numba**, the elementary unit is solved in **0.0001 s**, unleashing the full power of the population-based intelligence algorithm.

---

## Functional Architecture
The workflow is “parameter modeling → simulation → optimization → result output”.  
Key modules:

| Module | Purpose |
|--------|---------|
| **Parameter Modeling** | Define UAV & missile initial positions, real-target area, generate smoke-validity test points. |
| **Core Simulation** | Compute UAV trajectory, decoy release / burst position, missile trajectory, decide whether smoke shields the line-of-sight. |
| **Optimization** | Differential-Evolution optimizer with elite retention & local search. |
| **Result Output** | Save best strategy, fitness curve, support hot-start from historic best. |

---

## Environment
### Basic
- Python 3.7+
- OS: Windows / macOS / Linux (Linux recommended for parallel speed-up)

### Dependencies
```bash
pip install numpy matplotlib seaborn numba
```
- **numpy** – numerical arrays  
- **matplotlib / seaborn** – fitness-curve plotting  
- **numba** – JIT compilation of hotspots (crucial 5–10× speed-up)

---

## Core Code Explanation
### 1. Directory
```
.
├── cumcm_a.py        # main script (all classes & functions)
├── exp.txt           # auto-created best-solution file
├── README.pdf        #
└── README.md         #
```

### 2. Key Functions / Classes
#### Geometry & Simulation (Numba-accelerated)
| Name | Purpose |
|------|---------|
| `intersect(p1, p2, center, radius_sq)` | Does segment p1-p2 intersect sphere (center, radius²)? (smoke occlusion test) |
| `test(p1, p2, test_points, radius_sq)` | Is missile LOS p1-p2 blocked by smoke at every test point? |
| `pos_numba(...)` | Simulate one UAV’s whole decoy process, return effective shielding time intervals. |

#### Optimization Core – Class `Pop`
Encapsulates initialization, fitness evaluation, DE operators, elite & local search.

- `__init__` – sets UAV/missile data, search bounds (speed 70–140 m/s, angle 0–2π, …), population size, etc.  
- `evaluate_strategy` – total shielding duration (fitness) of a strategy.  
- `differential_evolution` – multi-strategy mutation (rand/1, best/1), crossover, selection, diversity control, elite.  
- `fine_local_search` – refined tweak of best individual for extra fitness gain.

#### Save / Load
- `save_best_solution` – append best strategy & fitness to `exp.txt`.  
- `load_best_solution` – hot-start from historic best (avoids redundant runs).

---

## Quick-Start
### 1. Default Run (Question 5 demo)
```bash
python cumcm_a.py
```
Flow:
1. Load historic best from `exp.txt` (if any).  
2. Init population & parameters.  
3. Run 2500 DE generations, live-plot fitness curve.  
4. Local-search the best.  
5. Save final strategy to `exp.txt`.

### 2. Switch to Other Questions
Edit the `main` block in `cumcm_a.py`.

#### Question 1 – FY1 drops 1 decoy to interfere M1 (fixed parameters)
```python
if __name__ == "__main__":
    ...
    missile_num = 1
    drone_num   = 1
    smack_num   = 1
    pop_size    = 50
    generations = 1000
    # fix FY1 params as required
    fixed_vel, fixed_angle, fixed_drop_time, fixed_delay = 120, 0, 1.5, 3.6
    pop = Pop(...)
    pop.population[:, 0, 0] = fixed_vel
    ...
    fitness = pop.evaluate_strategy(pop.population[0])
    print(f"Q1 effective shielding = {fitness:.2f} s")
```

#### Question 2 – FY1 drops 1 decoy, **maximize** shielding duration
```python
...
pop = Pop(1, 1, 1, pop_size=80)
best_ind, history = pop.differential_evolution(1500, plot_fitness=True)
print("Q2 best strategy:")
print(f"speed  : {best_ind[0][0]:.2f} m/s")
print(f"angle  : {best_ind[0][1]:.2f} rad")
print(f"drop T : {best_ind[0][2]:.2f} s")
print(f"delay  : {best_ind[0][3]:.2f} s")
print(f"max shielding: {history[-1]:.2f} s")
```

---

## Tuning Hints
| Parameter | Advice |
|-----------|--------|
| `pop_size` | simple (1 UAV, 1 decoy) 50–80; complex (5 UAV, 3 decoy each) 100–150. |
| `generations` | simple 1000–1500; complex 2000–3000. |
| `F` (mutation) | start 0.5; raise to 0.7 if diversity low; lower to 0.4 if convergence slow. |
| `CR` (crossover) | start 0.6; raise to 0.8 for complex landscapes. |

---

## Reading Results
### 1. `exp.txt` format
```
=== START: missile_num=3, drone_num=5, smack_num=3 ===
best_fitness: 125.87
best_individual:
135.2,1.57,20.1,5.2,45.3,6.1,70.5,4.8  # UAV1: speed,angle,drop1,delay1,drop2,delay2,drop3,delay3
120.3,0.89,18.5,4.9,50.2,5.5,75.1,3.2
...
=== END: missile_num=3, drone_num=5, smack_num=3 ===
```

### 2. Fitness Curve
A live window pops up while running: x-axis = generation, y-axis = best shielding duration. Curve flattening ⇒ convergence.

---

## Performance Notes
1. **Numba JIT**: core functions (`intersect`, `pos_numba`, …) decorated with `@jit(nopython=True)` give 5–10× speed-up.  
2. **Parallel eval**: `fit_numba_flat` uses `prange` to multi-thread fitness evaluation, exploiting multi-core CPUs.  
3. **Diversity injection**: when population diversity drops, new random individuals are injected to escape local optima.

---

## FAQ
1. **Slow?**  
   – Ensure `numba` installed and `nopython=True` set.  
   – Reduce `pop_size` or `generations`.  
   – Run on multi-core machine (Numba auto-parallel).

2. **Fitness curve flat?**  
   – Increase `F` (e.g. 0.5→0.7) for more diversity.  
   – Check parameter bounds (e.g. drop time within 0–200 s).

3. **Cannot load historic best?**  
   – Check `exp.txt` format: `START` / `END` markers must match current `(missile, drone, smack)` counts.
