import numpy as np
from numba import njit
import time
from multivar_hypergeom import MultivariateHypergeometric

@njit
def draw_sample(population_counts, sample_size):
    k = len(population_counts)
    counts = np.zeros(k, dtype=np.int64)
    remaining = population_counts.copy()
    total_remaining = np.sum(remaining)
    
    for _ in range(sample_size):
        r = np.random.randint(0, total_remaining)
        cum_sum = 0
        for i in range(k):
            cum_sum += remaining[i]
            if r < cum_sum:
                counts[i] += 1
                remaining[i] -= 1
                total_remaining -= 1
                break
    return counts

@njit(parallel=True)
def multivariate_hypergeometric(population, sample_size, num_samples):
    samples = np.zeros((num_samples, len(population)), dtype=np.int64)
    for i in range(num_samples):
        samples[i] = draw_sample(population, sample_size)
    return samples

if __name__ == "__main__":
    # Microbial community parameters
    N_SPECIES = round(25 * 10**3)      # 25,000 microbial species
    TOTAL_SIZE = round(5.4 * 10**9)    # 5.4 billion total cells
    population = np.full(N_SPECIES, TOTAL_SIZE // N_SPECIES, dtype=np.int64)
    sample_size = round(0.007 * TOTAL_SIZE)  # 0.7% of total population
    
    # Warm-up JIT compilation
    multivariate_hypergeometric(population[:10], 10, 1)
    
    # Performance test
    print(f"Testing with:\n- {N_SPECIES:,} species\n- {TOTAL_SIZE:,} total cells")
    print(f"- Sampling {sample_size:,} cells ({sample_size/TOTAL_SIZE:.1%})")
    
    start_time = time.perf_counter()
    samples = multivariate_hypergeometric(population, sample_size, num_samples=1)
    elapsed = time.perf_counter() - start_time
    
    # Validation checks
    print("\nResults:")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Sample sum check: {samples.sum() == sample_size}")
    print(f"Max per species: {samples.max()} (population: {population[0]})")
    print("First 5 species counts:", samples[0, :5])
