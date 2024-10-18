import numpy as np
from scipy.stats import lognorm, randint
from hypergeometric import multivariate_hypergeometric

N_OF_SPECIES = round(25 * 10**3)
TOTAL_SIZE = round(5.4 * 10**9)
SIZE_PER_SPECIES = int(TOTAL_SIZE / N_OF_SPECIES)
GR_MEAN = 0.33407776069792355  # Per hour
GR_VAR = 0.0003118415980511718  # Per hour
SIGMA = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))
MU = np.log(GR_MEAN) - (SIGMA**2) / 2
EXPERIMENT_LENGTH = 51810  # 5 minute increments
DT = 5 / 600  # In Hour
steps_b_measurements = 10
grs = lognorm.rvs(s=SIGMA, scale=np.exp(MU), size=N_OF_SPECIES)
sizes = randint.rvs(SIZE_PER_SPECIES - 100, SIZE_PER_SPECIES + 101, size=N_OF_SPECIES)
pump_is_on = False
VOLUME = 90  # mL
PUMP_OUT_RATE = 80  # mL / Hour

total_sizes_array = np.zeros(
    (round(EXPERIMENT_LENGTH / steps_b_measurements), N_OF_SPECIES), dtype=int
)
total_pump_array = np.zeros(round(EXPERIMENT_LENGTH / steps_b_measurements), dtype=bool)
total_ratio = np.zeros(round(EXPERIMENT_LENGTH / steps_b_measurements), dtype=float)
total_sizes_array[0, :] = sizes
last_sizes_array = np.zeros((steps_b_measurements, N_OF_SPECIES), dtype=int)
last_sizes_array[0, :] = sizes
last_pump_array = np.zeros(steps_b_measurements, dtype=bool)
last_ratio = np.zeros(steps_b_measurements, dtype=float)

for i in range(
    1, EXPERIMENT_LENGTH
):  # Python uses 0-based indexing, so we adjust the range
    curr_step = i % 10
    prev_step = curr_step - 1
    curr_sizes = last_sizes_array[prev_step, :]
    pump_is_on = last_pump_array[prev_step]

    next_sizes = np.round(np.exp(grs * DT) * curr_sizes).astype(np.int64)

    if i % 10 == 0:
        if np.sum(next_sizes) > TOTAL_SIZE * 1.05:
            pump_is_on = True
        elif np.sum(next_sizes) < TOTAL_SIZE:
            pump_is_on = False

    if pump_is_on:
        n_to_remove = round((np.sum(next_sizes) / VOLUME) * PUMP_OUT_RATE * DT)
        to_remove = multivariate_hypergeometric(next_sizes, n_to_remove)
        next_sizes = next_sizes - to_remove

    last_sizes_array[curr_step, :] = next_sizes
    last_pump_array[curr_step] = pump_is_on
    last_ratio[curr_step] = np.sum(next_sizes) / np.sum(last_sizes_array[prev_step, :])

    if i % 10 == 9:
        total_sizes_array[i // 10, :] = next_sizes
        total_pump_array[i // 10] = pump_is_on
        total_ratio[i // 10] = np.sum(next_sizes) / np.sum(
            total_sizes_array[i // 10 - 1, :]
        )
        print(f"{np.sum(next_sizes)}-----{i}")

np.savetxt("total_sizes_array.csv", total_sizes_array, delimiter=",")
np.savetxt("grs.csv", grs, delimiter=",")
np.savetxt("total_pump_array.csv", total_pump_array, delimiter=",")
averegre_gr_over_time = np.dot(total_sizes_array, grs) / np.sum(
    total_sizes_array, axis=1
)
np.savetxt("averegre_gr_over_time.csv", averegre_gr_over_time, delimiter=",")
sizes_sum = np.sum(total_sizes_array, axis=1)
np.savetxt("sizes_sum.csv", sizes_sum, delimiter=",")
sizes_sum = sizes_sum[:, np.newaxis]
freq_total = total_sizes_array / sizes_sum
freq_samples = freq_total[::500, :]
np.savetxt("freq_samples.csv", freq_samples, delimiter=",")
