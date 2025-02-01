import numpy as np
from scipy.stats import lognorm, randint
from hypergeometric import multivariate_hypergeometric
from tqdm import tqdm

# Constants
N_OF_SPECIES = round(25 * 10**3)
TOTAL_SIZE = round(5.4 * 10**9)
SIZE_PER_SPECIES = int(TOTAL_SIZE / N_OF_SPECIES)
GR_MEAN = 0.33407776069792355  # Per hour
GR_VAR = 0.0003118415980511718  # Per hour
SIGMA = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))
MU = np.log(GR_MEAN) - (SIGMA**2) / 2
EXPERIMENT_LENGTH = 51810  # 5 minute increments
DT = 5 / 600  # In Hour
VOLUME = 90  # mL
PUMP_OUT_RATE = 80  # mL / Hour

def initialize_system():
    grs = lognorm.rvs(s=SIGMA, scale=np.exp(MU), size=N_OF_SPECIES)
    sizes = randint.rvs(SIZE_PER_SPECIES - 100, SIZE_PER_SPECIES + 101, size=N_OF_SPECIES)
    return grs, sizes

def run_simulation():
    # Initialize
    grs, sizes = initialize_system()
    steps_b_measurements = 10
    pump_is_on = False
    
    # Storage arrays
    total_sizes_array = np.zeros((round(EXPERIMENT_LENGTH / steps_b_measurements), N_OF_SPECIES), dtype=int)
    total_pump_array = np.zeros(round(EXPERIMENT_LENGTH / steps_b_measurements), dtype=bool)
    total_ratio = np.zeros(round(EXPERIMENT_LENGTH / steps_b_measurements), dtype=float)
    total_sizes_array[0, :] = sizes
    
    # Buffer arrays for intermediate steps
    last_sizes_array = np.zeros((steps_b_measurements, N_OF_SPECIES), dtype=int)
    last_sizes_array[0, :] = sizes
    last_pump_array = np.zeros(steps_b_measurements, dtype=bool)
    last_ratio = np.zeros(steps_b_measurements, dtype=float)

    # Main simulation loop with progress bar
    for i in tqdm(range(1, EXPERIMENT_LENGTH), desc="Running simulation"):
        curr_step = i % 10
        prev_step = curr_step - 1
        curr_sizes = last_sizes_array[prev_step, :]
        
        # Growth step
        next_sizes = np.round(np.exp(grs * DT) * curr_sizes).astype(np.int64)
        
        # Pump step
        if pump_is_on:
            n_to_remove = round(np.sum(next_sizes) * (1 - np.exp(-PUMP_OUT_RATE * DT / VOLUME)))
            to_remove = multivariate_hypergeometric(next_sizes, n_to_remove)
            next_sizes = next_sizes - to_remove
            
        # Store intermediate results
        last_sizes_array[curr_step, :] = next_sizes
        last_pump_array[curr_step] = pump_is_on
        last_ratio[curr_step] = np.sum(next_sizes) / np.sum(last_sizes_array[prev_step, :])
        
        # Store results every 10 steps
        if i % 10 == 9:
            total_sizes_array[i // 10, :] = next_sizes
            total_pump_array[i // 10] = pump_is_on
            total_ratio[i // 10] = np.sum(next_sizes) / np.sum(total_sizes_array[i // 10 - 1, :])
        
        # Update pump state
        if np.sum(next_sizes) > TOTAL_SIZE * 1.05:
            pump_is_on = True
        elif np.sum(next_sizes) < TOTAL_SIZE:
            pump_is_on = False
            
        # Removed progress print since we now have tqdm
    
    return {
        'sizes': total_sizes_array,
        'growth_rates': grs,
        'pump_states': total_pump_array,
        'ratios': total_ratio
    }

def save_results(results):
    np.savetxt("total_sizes_array.csv", results['sizes'], delimiter=",")
    np.savetxt("grs.csv", results['growth_rates'], delimiter=",")
    np.savetxt("total_pump_array.csv", results['pump_states'], delimiter=",")
    
    # Calculate and save additional metrics
    averegre_gr_over_time = np.dot(results['sizes'], results['growth_rates']) / np.sum(results['sizes'], axis=1)
    np.savetxt("averegre_gr_over_time.csv", averegre_gr_over_time, delimiter=",")
    
    sizes_sum = np.sum(results['sizes'], axis=1)
    np.savetxt("sizes_sum.csv", sizes_sum, delimiter=",")
    
    sizes_sum = sizes_sum[:, np.newaxis]
    freq_total = results['sizes'] / sizes_sum
    freq_samples = freq_total[::500, :]
    np.savetxt("freq_samples.csv", freq_samples, delimiter=",")

# Run simulation
if __name__ == "__main__":
    results = run_simulation()
    save_results(results)