import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_OF_SPECIES = round(25 * 10**3)
TOTAL_SIZE = round(5.4 * 10**9)
SIZE_PER_SPECIES = int(TOTAL_SIZE / N_OF_SPECIES)
GR_MEAN = 0.33407776069792355  # Per hour
GR_VAR = 0.0003118415980511718  # Per hour
SIGMA = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))
MU = np.log(GR_MEAN) - (SIGMA**2) / 2
EXPERIMENT_LENGTH = 51810  # 5 minute increments
DT = 5 / 60  # In Hour
DT_OF_SAMPLE = 5 / 600  # In hour
VOLUME = 90  # mL
PUMP_OUT_RATE = 80  # mL / Hour

SIMULATION = False
USE_PUMP_DATA = False


def load_data(simulation=True):
    if simulation:
        sizes = np.genfromtxt("data/simulation_data/sizes_sum.csv", delimiter=",")
        pump_rate_array = (
            np.genfromtxt("data/simulation_data/total_pump_array.csv", delimiter=",")
            * PUMP_OUT_RATE
        )
        frequencies = np.genfromtxt(
            "data/simulation_data/freq_samples.csv", delimiter=","
        )
        grs = np.genfromtxt("data/simulation_data/grs.csv", delimiter=",")
        average_gr_over_time = np.genfromtxt(
            "data/simulation_data/averegre_gr_over_time.csv", delimiter=","
        )
        timestamps = np.arange(0, 500 * frequencies.shape[0], 500)
    else:
        experiment_data = pd.read_csv("data/experiment_data/experiment_data.csv")
        sizes = (
            np.array(experiment_data["OD_Converted"]) * 90 * 30 * (10**6)
        ).astype(np.int64)[11:]
        pump_rate_array = (
            np.array(experiment_data["Pump_Rate[mL/h]"]) * 2 * 30 * (10**6)
        ).astype(np.int64)[11:]
        frequencies_table = pd.read_csv('data/experiment_data/frequency_data.csv')
        timestamps_in_hours = list(frequencies_table.columns)
        timestamps = [int(float(i)* 12) for i in timestamps_in_hours]
        frequencies = frequencies_table.values.T
        grs = None
        average_gr_over_time = None
    return sizes, pump_rate_array, frequencies, grs, average_gr_over_time, timestamps


def get_indices(sizes, pump_rate_array, use_pump_data=True):
    if use_pump_data:
        nonzero_indices = np.nonzero(pump_rate_array)[0]
        zero_indices = np.where(pump_rate_array == 0)[0]

        nonzero_indices = np.split(
            nonzero_indices, np.where(np.diff(nonzero_indices) != 1)[0] + 1
        )
        zero_indices = np.split(
            zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1
        )
    else:
        # Calculate differences between consecutive elements
        diffs = np.diff(sizes)

        # Find indices of increasing and decreasing segments
        zero_indices = np.where(diffs > 0)[0] + 1
        nonzero_indices = np.where(diffs <= 0)[0] + 1

        # Include the start index for each segment
        # zero_indices = np.r_[0, zero_indices]
        # nonzero_indices = np.r_[0, nonzero_indices]

        # Split the indices into continuous runs
        zero_indices = np.split(
            zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1
        )
        nonzero_indices = np.split(
            nonzero_indices, np.where(np.diff(nonzero_indices) != 1)[0] + 1
        )

    # Convert numpy arrays to lists
    zero_indices = [run.tolist() for run in zero_indices if len(run) >= 1]
    nonzero_indices = [run.tolist() for run in nonzero_indices if len(run) >= 1]

    for i in range(len(zero_indices)):
        arr = zero_indices[i]
        new_value = max(arr[0] - 1, 0)
        zero_indices[i] = np.insert(arr, 0, new_value)

    if use_pump_data:
        zero_indices[0] = zero_indices[0][1:]

    return zero_indices, nonzero_indices


def calculate_kfir_growth_rate(zero_indices, nonzero_indices, sizes):
    average_gr_list = []
    for growth_period in zero_indices:
        start = growth_period[0]
        stop = growth_period[-1]
        time_elapsed = DT * (stop - start)
        gr = np.log(sizes[stop] / sizes[start]) / time_elapsed
        average_gr_list.append(gr)
        if gr < 0:
            print(0)

    average_gr_over_time = np.zeros(sizes.shape[0])
    for i in range(len(nonzero_indices)):
        gr = average_gr_list[i]
        average_gr_over_time[zero_indices[i]] = gr
        average_gr_over_time[nonzero_indices[i]] = gr

    if average_gr_over_time[-1] == 0:
        average_gr_over_time = average_gr_over_time[:-1]

    return average_gr_over_time


def calculate_ruti_ajusted_growth_rate(zero_indices, nonzero_indices, sizes):
    average_gr_list = []
    for i in range(len(nonzero_indices)):
        growth_period = zero_indices[i]
        pump_period = nonzero_indices[i]
        start = growth_period[0]
        middle = growth_period[-1]
        stop = pump_period[-1]
        time_elapsed = DT * (middle - start)
        gr = (
            np.log(sizes[stop] / sizes[start])
            - np.log(1 - (sizes[middle] - sizes[stop]) / sizes[middle])
        ) / time_elapsed
        average_gr_list.append(gr)

    average_gr_over_time = np.zeros(sizes.shape[0])
    for i in range(len(nonzero_indices)):
        gr = average_gr_list[i]
        average_gr_over_time[zero_indices[i]] = gr
        average_gr_over_time[nonzero_indices[i]] = gr

    if average_gr_over_time[-1] == 0:
        average_gr_over_time = average_gr_over_time[:-1]

    return average_gr_over_time


def calculate_ruti_growth_rate(zero_indices, nonzero_indices, sizes, pump_rate_array):
    average_gr_list = []
    for i in range(len(nonzero_indices)):
        growth_period = zero_indices[i]
        pump_period = nonzero_indices[i]
        start = growth_period[0]
        middle = growth_period[-1]
        stop = pump_period[-1]
        time_elapsed = DT * (stop - start)
        pumped_out_vol = np.sum((pump_rate_array[middle + 1 : stop + 1]) * DT)
        gr = (
            (pumped_out_vol / VOLUME) + np.log(sizes[stop] / sizes[start])
        ) / time_elapsed
        average_gr_list.append(gr)

    average_gr_over_time = np.zeros(sizes.shape[0])
    for i in range(len(nonzero_indices)):
        gr = average_gr_list[i]
        average_gr_over_time[zero_indices[i]] = gr
        average_gr_over_time[nonzero_indices[i]] = gr

    if average_gr_over_time[-1] == 0:
        average_gr_over_time = average_gr_over_time[:-1]

    return average_gr_over_time


def calculate_relative_frequency(frequencies, timestamps, average_gr_over_time):
    n_of_timestamps, n_of_grs = frequencies.shape
    strain_grs_table = np.zeros((n_of_timestamps - 1, n_of_grs))
    for i in range(n_of_timestamps - 1):
        prev_ts = timestamps[i]
        curr_ts = timestamps[i + 1]
        time_elapsed = DT * (curr_ts - prev_ts)

        prev_relative_freq = frequencies[i, :]
        curr_relative_freq = frequencies[i + 1, :]
        log_relative_freq = np.log(curr_relative_freq / prev_relative_freq)
        average_gr = np.mean(average_gr_over_time[prev_ts:curr_ts])
        strain_grs = log_relative_freq / time_elapsed + average_gr
        strain_grs_table[i, :] = strain_grs

    return strain_grs_table


sizes, pump_rate_array, frequencies, grs, average_gr_over_time, timestamps = load_data(
    SIMULATION
)

zero_indices, nonzero_indices = get_indices(sizes, pump_rate_array, USE_PUMP_DATA)

# Kfir Ajusted method
kfir_average_gr_over_time = calculate_kfir_growth_rate(
    zero_indices, nonzero_indices, sizes
)

# Ruti Ajusted method
ruti_ajusted_average_gr_over_time = calculate_ruti_ajusted_growth_rate(
    zero_indices, nonzero_indices, sizes
)

# Ruti method
ruti_average_gr_over_time = calculate_ruti_growth_rate(
    zero_indices, nonzero_indices, sizes, pump_rate_array
)

#plt.plot(kfir_average_gr_over_time)
#plt.plot(ruti_ajusted_average_gr_over_time)
#plt.plot(ruti_average_gr_over_time)
#plt.show()

# Relative frequency calculation
kfir_strain_grs_table = calculate_relative_frequency(
    frequencies, timestamps, ruti_ajusted_average_gr_over_time
)
ruti_strain_grs_table = calculate_relative_frequency(
    frequencies, timestamps, ruti_average_gr_over_time
)

n_of_timestamps, n_of_grs = frequencies.shape
ruti_strain_grs_table = np.zeros((n_of_timestamps - 1, n_of_grs))

frequency_inicator = frequencies > 4 * 10 ** (-5)
frequency_inicator = frequency_inicator[:-1, :]
# plt.plot(average_gr_over_time)
plt.scatter(ruti_strain_grs_table[-1, :], grs)
plt.plot([0, 1], [0, 1])
plt.show()
plt.plot(average_gr_over_time)
plt.plot(kfir_average_gr_over_time)
plt.plot(ruti_average_gr_over_time)
plt.show()

print(0)
