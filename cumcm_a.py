import numpy as np
import time
import os
from numba import jit, float64, boolean, int32, prange
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


@jit(float64(float64[:], float64[:], float64[:], float64), nopython=True, fastmath=True)
def intersect(p1, p2, center, radius_sq):
    v0 = p2[0] - p1[0]
    v1 = p2[1] - p1[1]
    v2 = p2[2] - p1[2]

    w0 = p1[0] - center[0]
    w1 = p1[1] - center[1]
    w2 = p1[2] - center[2]

    a = v0**2 + v1**2 + v2**2

    if a == 0:
        return (w0**2 + w1**2 + w2**2) <= radius_sq

    b = v0 * w0 + v1 * w1 + v2 * w2
    t = -b / a

    if t < 0:
        closest0 = p1[0]
        closest1 = p1[1]
        closest2 = p1[2]
    elif t > 1:
        closest0 = p2[0]
        closest1 = p2[1]
        closest2 = p2[2]
    else:
        closest0 = p1[0] + t * v0
        closest1 = p1[1] + t * v1
        closest2 = p1[2] + t * v2

    dx = closest0 - center[0]
    dy = closest1 - center[1]
    dz = closest2 - center[2]
    return (dx*dx + dy*dy + dz*dz) <= radius_sq


@jit(boolean(float64[:], float64[:], float64[:,:], float64), nopython=True, fastmath=True)
def test(p1, p2, test_points, radius_sq):
    for i in range(test_points.shape[0]):
        p3 = test_points[i]
        if not intersect(p2, p3, p1, radius_sq):
            return False
    return True


@jit(float64(float64, float64, boolean, float64[:], float64[:], float64, float64[:,:], float64),
     nopython=True, fastmath=True)
def refine_edge(prev_time, curr_time, prev_state, drone_params, missile_params, total_time, test_points, radius_sq):
    x0, y0, z0_drone, vx, vy, g = drone_params
    mx0, my0, mz0 = missile_params

    iterations = 20
    left, right = prev_time, curr_time

    for _ in range(iterations):
        mid_time = (left + right) / 2

        t = total_time + mid_time
        z0 = z0_drone - 3 * mid_time
        
        r0 = np.sqrt(mx0**2 + my0**2 + mz0**2)
        if r0 == 0:
            x1, y1, z1 = 0.0, 0.0, 0.0
        else:
            ux, uy, uz = -mx0 / r0, -my0 / r0, -mz0 / r0
            distance = 300.0 * t
            x1 = mx0 + ux * distance
            y1 = my0 + uy * distance
            z1 = mz0 + uz * distance

        p1 = np.array([x0, y0, z0], dtype=np.float64)
        p2 = np.array([x1, y1, z1], dtype=np.float64)
        current_state = test(p1, p2, test_points, radius_sq)

        if current_state == prev_state:
            left = mid_time
        else:
            right = mid_time

    return (left + right) / 2


@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64[:,:], float64),
     nopython=True, fastmath=True)
def pos_numba(drone_pos, drone_vel, drone_angle, drop_time, detonate_delay, missile_pos, test_points, radius_sq):
    empty_result = np.empty((0,), dtype=np.float64)

    time_step = 0.1
    total_simulation_time = 20.0
    total_steps = int(total_simulation_time / time_step)

    g = 9.8
    x0, y0, z0 = drone_pos

    if not np.isnan(drone_angle):
        vx = drone_vel * np.cos(drone_angle)
        vy = drone_vel * np.sin(drone_angle)
    else:
        vx = -120
        vy = 0

    total_time = drop_time + detonate_delay
    x0 += vx * total_time
    y0 += vy * total_time
    z0_drone = z0 - 0.5 * g * (detonate_delay ** 2)

    mx0, my0, mz0 = missile_pos
    r0 = np.sqrt(mx0**2 + my0**2 + mz0**2)
    if r0 == 0:
        return empty_result

    ux, uy, uz = -mx0 / r0, -my0 / r0, -mz0 / r0

    distance_total = 300.0 * total_time
    x1_init = mx0 + ux * distance_total
    y1_init = my0 + uy * distance_total
    z1_init = mz0 + uz * distance_total

    distance_step = 300.0 * time_step
    dx_step = ux * distance_step
    dy_step = uy * distance_step
    dz_step = uz * distance_step

    drone_params = np.array([x0, y0, z0_drone, vx, vy, g], dtype=np.float64)
    missile_params = np.array([mx0, my0, mz0], dtype=np.float64)

    max_transitions = total_steps
    transition_points = np.empty(max_transitions, dtype=np.float64)
    transition_states = np.empty(max_transitions, dtype=np.bool_)
    transition_count = 0

    z0_current = z0_drone - 3 * 0
    x1_current = x1_init
    y1_current = y1_init
    z1_current = z1_init

    p1_init = np.array([x0, y0, z0_current], dtype=np.float64)
    p2_init = np.array([x1_current, y1_current, z1_current], dtype=np.float64)
    prev_state = test(p1_init, p2_init, test_points, radius_sq)

    for i in range(1, total_steps):
        current_time = i * time_step

        z0_current = z0_drone - 3 * current_time
        x1_current += dx_step
        y1_current += dy_step
        z1_current += dz_step

        p1 = np.array([x0, y0, z0_current], dtype=np.float64)
        p2 = np.array([x1_current, y1_current, z1_current], dtype=np.float64)
        current_state = test(p1, p2, test_points, radius_sq)

        if current_state != prev_state and transition_count < max_transitions:
            refined_time = refine_edge(
                (i-1)*time_step,
                current_time,
                prev_state,
                drone_params,
                missile_params,
                total_time,
                test_points,
                radius_sq
            )
            transition_points[transition_count] = refined_time
            transition_states[transition_count] = current_state
            transition_count += 1
            prev_state = current_state

    first = -1.0
    last = -1.0
    for i in range(transition_count):
        t = transition_points[i]
        state = transition_states[i]
        if state:
            if first < 0:
                first = t
        else:
            last = t

    if transition_count == 0 and prev_state:
        first = 0.0
        last = total_simulation_time
    elif transition_count > 0 and transition_states[transition_count - 1]:
        last = total_simulation_time

    if first < 0:
        return empty_result
    return np.array([total_time + first, total_time + last], dtype=np.float64)


def get_test_points():
    center = (0, 200, 0)
    radius = 7
    height = 10
    num_theta = 12
    num_radial = 12
    num_z_side = 16

    test_points = []

    z_bottom = center[2]
    radii_bottom = np.linspace(0, radius, num_radial)
    thetas = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    for r in radii_bottom:
        for theta in thetas:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            test_points.append((x, y, z_bottom))

    z_top = center[2] + height
    radii_top = np.linspace(0, radius, num_radial)
    for r in radii_top:
        for theta in thetas:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            test_points.append((x, y, z_top))

    z_side = np.linspace(center[2] + 1e-6, z_top - 1e-6, num_z_side)
    for z in z_side:
        for theta in thetas:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            test_points.append((x, y, z))

    return np.array(test_points, dtype=np.float64)


@jit(nopython=True, fastmath=True)
def get_smack_intervals(drone_params_list, drone_vels, drone_angles, drop_times, detonate_delays, missile_positions, test_points, radius_sq):
    num_drones = len(drone_params_list)
    num_missiles = missile_positions.shape[0]
    num_smacks_per_drone = drop_times.shape[0] if drop_times.ndim == 1 else drop_times.shape[1]

    all_intervals = []
    for drone_idx in range(num_drones):
        drone_pos = drone_params_list[drone_idx]
        vel = drone_vels[drone_idx]
        angle = drone_angles[drone_idx]

        for smack_idx in range(num_smacks_per_drone):
            drop_time = drop_times[drone_idx, smack_idx] if drop_times.ndim == 2 else drop_times[smack_idx]
            delay = detonate_delays[drone_idx, smack_idx] if detonate_delays.ndim == 2 else detonate_delays[smack_idx]

            for missile_idx in range(num_missiles):
                missile_pos = missile_positions[missile_idx]
                interval = pos_numba(drone_pos, vel, angle, drop_time, delay, missile_pos, test_points, radius_sq)
                if interval.size > 0:
                    all_intervals.append((interval[0], interval[1], missile_idx))

    return np.array(all_intervals, dtype=np.float64)


@jit(float64(float64[:], int32, int32, int32, float64[:,:], float64[:,:], float64[:,:], float64),
     nopython=True, fastmath=True, parallel=True)
def fit_numba_flat(individual_flat, missile_num, drone_num, smack_num, drone_params, missile_params, test_points, radius_sq):
    individual_flat = np.ascontiguousarray(individual_flat)
    param_count = 2 + 2 * smack_num
    individual = individual_flat.reshape(drone_num, param_count)

    total_score = 0.0

    for i in prange(missile_num):
        max_intervals = drone_num * smack_num
        all_start = np.empty(max_intervals, dtype=np.float64)
        all_end = np.empty(max_intervals, dtype=np.float64)
        interval_count = 0
        

        for j in range(drone_num):
            drop_times = np.empty(smack_num, dtype=np.float64)
            for k in range(smack_num):
                drop_times[k] = individual[j, 2 + k*2]

            min_gap = 1.0
            for a in range(smack_num):
                for b in range(a + 1, smack_num):
                    gap = abs(drop_times[a] - drop_times[b])
                    if gap < min_gap:
                        return 0.0

            for k in range(smack_num):
                drone_pos = drone_params[j]
                drone_vel = individual[j, 0]
                drone_angle = individual[j, 1]
                drop_time = individual[j, 2 + k*2]
                detonate_delay = individual[j, 3 + k*2]
                missile_pos = missile_params[i]

                interval = pos_numba(drone_pos, drone_vel, drone_angle, drop_time, detonate_delay, missile_pos, test_points, radius_sq)

                if interval.size > 0:
                    all_start[interval_count] = interval[0]
                    all_end[interval_count] = interval[1]
                    interval_count += 1
            

        if interval_count == 0:
            continue

        sorted_intervals = np.empty((interval_count, 2), dtype=np.float64)
        for idx in range(interval_count):
            sorted_intervals[idx, 0] = all_start[idx]
            sorted_intervals[idx, 1] = all_end[idx]

        sorted_indices = np.argsort(sorted_intervals[:, 0])
        sorted_intervals = sorted_intervals[sorted_indices]

        merged_start = np.empty(interval_count, dtype=np.float64)
        merged_end = np.empty(interval_count, dtype=np.float64)
        merged_count = 0

        if interval_count > 0:
            merged_start[0] = sorted_intervals[0, 0]
            merged_end[0] = sorted_intervals[0, 1]
            merged_count = 1

            for idx in range(1, interval_count):
                current_start = sorted_intervals[idx, 0]
                current_end = sorted_intervals[idx, 1]
                last_end = merged_end[merged_count - 1]

                if current_start <= last_end:
                    merged_end[merged_count - 1] = max(last_end, current_end)
                else:
                    merged_start[merged_count] = current_start
                    merged_end[merged_count] = current_end
                    merged_count += 1

        duration = 0.0
        for m in range(merged_count):
            duration += merged_end[m] - merged_start[m]

        total_score += duration

    return total_score


class Pop:
    def __init__(self, missile_num, drone_num, smack_num, size, initial_solution=None):
        self.size = size
        self.missile_num = missile_num
        self.drone_num = drone_num
        self.smack_num = smack_num
        self.missile_params = np.array([[20000,0,2000],[19000,600,2100],[18000,-600,1900]], dtype=np.float64)
        self.drone_params = np.array([[17800,0,1800],[12000,1400,1400],[6000,-3000,700],[11000,2000,1800],[13000,-2000,1300]], dtype=np.float64)

        self.drone_vel_bound = [70,140]
        self.drone_angle_bound = [0,2*np.pi]
        self.smack_drop_bound = [0,200]
        self.smack_delay_bound = [0,20]
        self.initial_solution = initial_solution
        self.population = self.init_population()
        self.fitness = np.zeros(self.size)
        self.test_points = get_test_points()
        self.radius_sq = 100.0

        self.success_history = np.zeros(50)
        self.success_count = 0
        self.history_idx = 0

        self.archive = []
        self.archive_fitness = []
        self.archive_threshold = 0.01
        
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.best_fitness_history = []
        self.diversity_threshold = 0.05
        
        self.local_search_step_sizes = [0.1, 0.05, 0.01]

        if self.initial_solution is not None:
            fitness_value = self.evaluate_strategy(self.initial_solution)
            self.archive_solution(self.initial_solution, fitness_value)

    def evaluate_strategy(self, strategy):
        individual_flat = strategy.flatten()
        return fit_numba_flat(
            individual_flat,
            self.missile_num,
            self.drone_num,
            self.smack_num,
            self.drone_params,
            self.missile_params,
            self.test_points,
            self.radius_sq
        )

    def init_population(self):
        param_count = 2 + 2 * self.smack_num
        population = np.empty((self.size, self.drone_num, param_count), dtype=np.float64)
        
        if self.initial_solution is not None:
            if self.initial_solution.shape == (self.drone_num, param_count):
                population[0] = self.initial_solution.copy()
            else:
                raise ValueError(f"初始解形状不匹配，期望 ({self.drone_num}, {param_count})，实际 {self.initial_solution.shape}")
            start_idx = 1
        else:
            start_idx = 0
        
        for i in range(start_idx, self.size):
            for j in range(self.drone_num):
                population[i,j,0] = np.random.uniform(self.drone_vel_bound[0], self.drone_vel_bound[1])
                population[i,j,1] = np.random.uniform(self.drone_angle_bound[0], self.drone_angle_bound[1])
                for k in range(self.smack_num):
                    population[i,j,2+k*2] = np.random.triangular(
                        self.smack_drop_bound[0],
                        (self.smack_drop_bound[0]+self.smack_drop_bound[1])/2,
                        self.smack_drop_bound[1]
                    )
                    population[i,j,3+k*2] = np.random.triangular(
                        self.smack_delay_bound[0],
                        (self.smack_delay_bound[0]+self.smack_delay_bound[1])/3,
                        self.smack_delay_bound[1]
                    )
        return population

    def update_fitness(self):
        param_count = 2 + 2 * self.smack_num
        for i in prange(self.size):
            individual_flat = self.population[i].flatten()
            self.fitness[i] = fit_numba_flat(
                individual_flat,
                self.missile_num,
                self.drone_num,
                self.smack_num,
                self.drone_params,
                self.missile_params,
                self.test_points,
                self.radius_sq
            )
    
    def calculate_diversity(self):
        flat_pop = self.population.reshape(self.size, -1)
        mean_individual = np.mean(flat_pop, axis=0)
        distances = np.linalg.norm(flat_pop - mean_individual, axis=1)
        return np.mean(distances) / np.linalg.norm(mean_individual) if np.any(mean_individual) else 0

    def archive_solution(self, individual, fitness_value):
        if not self.archive:
            self.archive.append(individual)
            self.archive_fitness.append(fitness_value)
            return

        distances = [np.linalg.norm(individual - archived) for archived in self.archive]
        if all(d > self.archive_threshold for d in distances):
            self.archive.append(individual)
            self.archive_fitness.append(fitness_value)

    def local_search(self, individual, bounds, F):
        param_count = 2 + 2 * self.smack_num
        perturbed_individual = individual.copy()
        for j in range(self.drone_num):
            for p in range(param_count):
                perturbation = np.random.uniform(-F, F) * (bounds[j, p, 1] - bounds[j, p, 0])
                perturbed_individual[j, p] = np.clip(individual[j, p] + perturbation, bounds[j, p, 0], bounds[j, p, 1])
        return perturbed_individual
    
    def fine_local_search(self, individual, bounds, max_iter=100):
        best_ind = individual.copy()
        best_fitness = self.evaluate_strategy(best_ind)
        
        for step_size in self.local_search_step_sizes:
            improved = True
            iter_count = 0
            
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                
                for j in range(self.drone_num):
                    for p in range(bounds.shape[1]):
                        temp_ind = best_ind.copy()
                        temp_ind[j, p] = np.clip(temp_ind[j, p] + step_size * (bounds[j, p, 1] - bounds[j, p, 0]), 
                                               bounds[j, p, 0], bounds[j, p, 1])
                        temp_fitness = self.evaluate_strategy(temp_ind)
                        
                        if temp_fitness > best_fitness:
                            best_ind = temp_ind.copy()
                            best_fitness = temp_fitness
                            improved = True
                            continue
                            
                        temp_ind = best_ind.copy()
                        temp_ind[j, p] = np.clip(temp_ind[j, p] - step_size * (bounds[j, p, 1] - bounds[j, p, 0]), 
                                               bounds[j, p, 0], bounds[j, p, 1])
                        temp_fitness = self.evaluate_strategy(temp_ind)
                        
                        if temp_fitness > best_fitness:
                            best_ind = temp_ind.copy()
                            best_fitness = temp_fitness
                            improved = True
                            
        return best_ind, best_fitness

    def differential_evolution(self, generations=1000, F_init=0.5, CR_init=0.6, plot_fitness=False):
        self.update_fitness()
        best_idx = np.argmax(self.fitness)
        best_fitness = self.fitness[best_idx]
        best_individual = self.population[best_idx].copy()
        self.best_fitness_history.append(best_fitness)

        param_count = 2 + 2 * self.smack_num
        bounds = np.empty((self.drone_num, param_count, 2), dtype=np.float64)
        for j in range(self.drone_num):
            bounds[j, 0] = self.drone_vel_bound
            bounds[j, 1] = self.drone_angle_bound
            for k in range(self.smack_num):
                bounds[j, 2 + k*2] = self.smack_drop_bound
                bounds[j, 3 + k*2] = self.smack_delay_bound

        fitness_history = [best_fitness]

        if plot_fitness:
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(10, 6))
            line, = ax.plot([], [], 'b-', label='Best Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Fitness Evolution During Optimization')
            ax.legend()
            ax.grid(True)
            plt.ion()
            fig.show()

        elite_count = 5
        elites = np.zeros((elite_count, self.drone_num, param_count), dtype=np.float64)
        elite_fitness = np.zeros(elite_count)
        elites[0] = best_individual
        elite_fitness[0] = best_fitness

        F = F_init
        CR = CR_init
        self.stagnation_counter = 0

        for gen in range(generations):
            current_diversity = self.calculate_diversity()
            if gen > 0:
                success_rate = np.mean(self.success_history)
                if success_rate < 0.2:
                    F = min(F * 1.1, 0.95)
                    CR = max(CR * 0.9, 0.1)
                elif success_rate > 0.5:
                    F = max(F * 0.9, 0.4)
                    CR = min(CR * 1.1, 0.95)
            
            if gen > 0 and best_fitness == self.best_fitness_history[-1]:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.stagnation_threshold or current_diversity < self.diversity_threshold:
                    replace_ratio = 0.3
                    replace_count = int(self.size * replace_ratio)
                    for i in range(replace_count):
                        rand_idx = np.random.randint(self.size)
                        self.population[rand_idx] = self.init_population()[0]
                    
                    for i in range(replace_count):
                        rand_idx = np.random.randint(self.size)
                        individual_flat = self.population[rand_idx].flatten()
                        self.fitness[rand_idx] = fit_numba_flat(
                            individual_flat,
                            self.missile_num,
                            self.drone_num,
                            self.smack_num,
                            self.drone_params,
                            self.missile_params,
                            self.test_points,
                            self.radius_sq
                        )
                    
                    current_best_idx = np.argmax(self.fitness)
                    if self.fitness[current_best_idx] > best_fitness:
                        best_fitness = self.fitness[current_best_idx]
                        best_individual = self.population[current_best_idx].copy()
                    
                    self.stagnation_counter = 0
                    print(f"Generation {gen}: 检测到局部最优，引入新个体增加多样性")
            else:
                self.stagnation_counter = 0

            mutation_strategy = np.random.choice(['rand/1', 'best/1', 'rand/2', 'best/2', 'archive'])

            for i in range(self.size):
                idxs = np.setdiff1d(np.arange(self.size), i)

                if mutation_strategy in ['rand/1', 'best/1']:
                    a, b = np.random.choice(idxs, 2, replace=False)
                    if mutation_strategy == 'rand/1':
                        mutant = self.population[a] + F * (self.population[b] - self.population[np.random.choice(idxs)])
                    else:
                        mutant = best_individual + F * (self.population[a] - self.population[b])
                elif mutation_strategy in ['rand/2', 'best/2']:
                    a, b, c, d = np.random.choice(idxs, 4, replace=False)
                    if mutation_strategy == 'rand/2':
                        mutant = self.population[a] + F * (self.population[b] - self.population[c] + self.population[d] - self.population[np.random.choice(idxs)])
                    else:
                        mutant = best_individual + F * (self.population[a] - self.population[b] + self.population[c] - self.population[d])
                elif mutation_strategy == 'archive':
                    if self.archive:
                        archived_idx = np.random.randint(len(self.archive))
                        mutant = self.population[np.random.choice(idxs)] + F * (self.archive[archived_idx] - self.population[np.random.choice(idxs)])
                    else:
                        mutant = self.population[np.random.choice(idxs)]
                else:
                    mutant = self.population[np.random.choice(idxs)]

                trial = self.population[i].copy()
                cross_idx = np.random.randint(param_count)
                for j in range(self.drone_num):
                    for p in range(param_count):
                        if p == cross_idx or np.random.rand() < CR:
                            trial[j, p] = np.clip(mutant[j, p], bounds[j, p, 0], bounds[j, p, 1])

                if np.random.rand() < 0.1:
                    trial = self.local_search(trial, bounds, F)

                trial_flat = trial.flatten()
                trial_fitness = fit_numba_flat(
                    trial_flat,
                    self.missile_num,
                    self.drone_num,
                    self.smack_num,
                    self.drone_params,
                    self.missile_params,
                    self.test_points,
                    self.radius_sq
                )

                improved = trial_fitness > self.fitness[i]
                self.success_history[self.history_idx] = 1.0 if improved else 0.0
                self.history_idx = (self.history_idx + 1) % len(self.success_history)

                if improved:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.archive_solution(trial, trial_fitness)

                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

                        if trial_fitness > np.min(elite_fitness):
                            min_idx = np.argmin(elite_fitness)
                            elites[min_idx] = trial.copy()
                            elite_fitness[min_idx] = trial_fitness

            if gen % 30 == 0 and gen > 0:
                for i in range(elite_count // 2):
                    elite_idx = np.random.choice(elite_count)
                    rand_idx = np.random.randint(self.size)
                    mutated_elite = elites[elite_idx] + 0.1 * F * (np.random.rand(*elites[elite_idx].shape) - 0.5)
                    for j in range(self.drone_num):
                        for p in range(param_count):
                            mutated_elite[j, p] = np.clip(mutated_elite[j, p], bounds[j, p, 0], bounds[j, p, 1])
                    self.population[rand_idx] = mutated_elite
                    self.fitness[rand_idx] = fit_numba_flat(
                        self.population[rand_idx].flatten(),
                        self.missile_num,
                        self.drone_num,
                        self.smack_num,
                        self.drone_params,
                        self.missile_params,
                        self.test_points,
                        self.radius_sq
                    )

            
            if gen % 100 == 0:
                fitness_std = np.std(self.fitness)
                if fitness_std < 0.1 * best_fitness:
                    F = min(F * 1.5, 0.95)
                    print(f"Increasing diversity, F adjusted to: {F:.2f}")

            fitness_history.append(best_fitness)
            self.best_fitness_history.append(best_fitness)

            if plot_fitness and (gen + 1) % 1 == 0:
                line.set_data(range(len(fitness_history)), fitness_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.001)

            if (gen + 1) % 10 == 0:
                print(f"Generation: {gen + 1}, Best Fitness: {best_fitness:.6f}, F: {F:.2f}, CR: {CR:.2f}, Archive Size: {len(self.archive)}, Diversity: {current_diversity:.6f}")

        if plot_fitness:
            plt.ioff()
            plt.show()
        
        print("开始对最优解进行局部搜索...")
        refined_ind, refined_fitness = self.fine_local_search(best_individual, bounds)
        
        if refined_fitness > best_fitness:
            improvement = (refined_fitness - best_fitness) / best_fitness * 100
            print(f"局部搜索优化成功，适应度提升 {improvement:.2f}%")
            best_individual = refined_ind
            best_fitness = refined_fitness
            fitness_history.append(best_fitness)
        else:
            print("局部搜索未找到更优解")

        return best_individual, fitness_history


def save_best_solution(filename, missile_num, drone_num, smack_num, best_individual, best_fitness):
    existing_content = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_content = f.readlines()
    
    start_marker = f"=== START: missile_num={missile_num}, drone_num={drone_num}, smack_num={smack_num} ==="
    end_marker = f"=== END: missile_num={missile_num}, drone_num={drone_num}, smack_num={smack_num} ==="
    
    new_content = []
    i = 0
    found = False
    
    while i < len(existing_content):
        if existing_content[i].strip() == start_marker:
            found = True
            new_content.append(f"{start_marker}\n")
            new_content.append(f"best_fitness: {best_fitness}\n")
            new_content.append("best_individual:\n")
            for drone in best_individual:
                new_content.append(",".join(map(str, drone)) + "\n")
            new_content.append(f"{end_marker}\n")
            
            while i < len(existing_content) and existing_content[i].strip() != end_marker:
                i += 1
            if i < len(existing_content):
                i += 1
        else:
            new_content.append(existing_content[i])
            i += 1
    
    if not found:
        new_content.append(f"{start_marker}\n")
        new_content.append(f"best_fitness: {best_fitness}\n")
        new_content.append("best_individual:\n")
        for drone in best_individual:
            new_content.append(",".join(map(str, drone)) + "\n")
        new_content.append(f"{end_marker}\n")
    
    with open(filename, 'w') as f:
        f.writelines(new_content)


def load_best_solution(filename, missile_num, drone_num, smack_num):
    if not os.path.exists(filename):
        return None, None
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    start_marker = f"=== START: missile_num={missile_num}, drone_num={drone_num}, smack_num={smack_num} ==="
    end_marker = f"=== END: missile_num={missile_num}, drone_num={drone_num}, smack_num={smack_num} ==="
    
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line == start_marker:
            start_idx = i + 1
        elif line == end_marker and start_idx != -1:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        return None, None
    
    params = {}
    data_start = start_idx
    for i in range(start_idx, end_idx):
        line = lines[i]
        if line.startswith("best_fitness:"):
            key, value = line.split(':', 1)
            params[key.strip()] = value.strip()
        elif line == "best_individual:":
            data_start = i + 1
            break
    
    param_count = 2 + 2 * smack_num
    best_individual = np.empty((drone_num, param_count), dtype=np.float64)
    
    for i in range(drone_num):
        if data_start + i >= end_idx:
            return None, None 
        best_individual[i] = np.fromstring(lines[data_start + i], dtype=np.float64, sep=',')
    
    best_fitness = float(params.get('best_fitness', -1))
    return best_individual, best_fitness


if __name__ == "__main__":
    start_time = time.time()
    missile_num = 3
    drone_num = 5
    smack_num = 3
    pop_size = 100
    generations = 2500
    exp_file = "exp.txt"

    best_ind, best_fitness = load_best_solution(exp_file, missile_num, drone_num, smack_num)
    
    if best_ind is not None and best_fitness is not None:
        print(f"成功加载已有最优解，适应度: {best_fitness}")
        print("将在已有最优解基础上进行优化...")
    else:
        print("未找到匹配的最优解，将进行新的优化...")

    pop = Pop(missile_num, drone_num, smack_num, pop_size, initial_solution=best_ind)
    best_ind, history = pop.differential_evolution(generations, plot_fitness=True)
    best_fitness = history[-1] if history else 0
    
    print("优化完成，最佳个体:", best_ind.tolist())
    print(f"最佳适应度: {best_fitness}")
    
    save_best_solution(exp_file, missile_num, drone_num, smack_num, best_ind, best_fitness)
    print(f"最优解已保存到 {exp_file}")

    end_time = time.time()
    print(f"总时间: {end_time - start_time:.2f} 秒")
