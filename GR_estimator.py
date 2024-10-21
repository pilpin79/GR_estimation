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
PUMP_SEPERATION = False

if SIMULATION:
    sizes = np.genfromtxt(
        "data/simulation_data/sizes_sum.csv",
        delimiter=",",
    )
    pump_rate_array = (
        np.genfromtxt("data/simulation_data/total_pump_array.csv", delimiter=",")
        * PUMP_OUT_RATE
    )
    frequencies = np.genfromtxt("data/simulation_data/freq_samples.csv", delimiter=",")
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

nonzero_indices = np.nonzero(pump_rate_array)[0]
zero_indices = np.where(pump_rate_array == 0)[0]

nonzero_indices = np.split(
    nonzero_indices, np.where(np.diff(nonzero_indices) != 1)[0] + 1
)
zero_indices = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)

# Kfir method
kfir_average_gr_list = []
for growth_period in zero_indices:
    start = max(growth_period[0] - 1, 0)
    stop = growth_period[-1]
    time_elapsed = DT * (stop - start)
    kfir_gr = np.log(sizes[stop] / sizes[start]) / time_elapsed
    kfir_average_gr_list.append(kfir_gr)
    if kfir_gr < 0:
        print(0)

kfir_average_gr_over_time = np.zeros(sizes.shape[0]-1)
for i in range(len(nonzero_indices)):
    kfir_gr = kfir_average_gr_list[i]
    kfir_average_gr_over_time[zero_indices[i]] = kfir_gr
    kfir_average_gr_over_time[nonzero_indices[i]] = kfir_gr

# Ruti method
ruti_average_gr_list = []
for i in range(len(nonzero_indices)):
    growth_period = zero_indices[i]
    pump_period = nonzero_indices[i]
    start = max(growth_period[0] - 1, 0)
    middle = growth_period[-1]
    stop = pump_period[-1]
    time_elapsed = DT * (stop - start)
    pumped_out_vol = np.sum((pump_rate_array[middle + 1 : stop + 1]) * DT)
    ruti_gr = (
        (pumped_out_vol / VOLUME) + np.log(sizes[stop] / sizes[start])
    ) / time_elapsed
    ruti_average_gr_list.append(ruti_gr)

ruti_average_gr_over_time = np.zeros(sizes.shape[0]-1)
for i in range(len(nonzero_indices)):
    ruti_gr = ruti_average_gr_list[i]
    ruti_average_gr_over_time[zero_indices[i]] = ruti_gr
    ruti_average_gr_over_time[nonzero_indices[i]] = ruti_gr

plt.plot(kfir_average_gr_over_time)
#plt.plot(ruti_average_gr_over_time)
plt.show()

n_of_timestamps, n_of_grs = frequencies.shape
kfir_strain_grs_table = np.zeros((n_of_timestamps - 1, n_of_grs))
ruti_strain_grs_table = np.zeros((n_of_timestamps - 1, n_of_grs))

# relative frequency calculation
for i in range(n_of_timestamps - 1):
    prev_ts = timestamps[i]
    curr_ts = timestamps[i + 1]
    time_elapsed = DT * (curr_ts - prev_ts)

    prev_relative_freq = frequencies[i, :]
    curr_relative_freq = frequencies[i + 1, :]
    log_relative_freq = np.log(curr_relative_freq / prev_relative_freq)
    kfir_average_gr = np.mean(kfir_average_gr_over_time[prev_ts:curr_ts])
    ruti_average_gr = np.mean(ruti_average_gr_over_time[prev_ts:curr_ts])
    kfir_strain_grs = log_relative_freq / time_elapsed + kfir_average_gr
    ruti_strain_grs = log_relative_freq / time_elapsed + ruti_average_gr
    kfir_strain_grs_table[i, :] = kfir_strain_grs
    ruti_strain_grs_table[i, :] = ruti_strain_grs

frequency_inicator = frequencies > 4 * 10 ** (-5)
frequency_inicator = frequency_inicator[:-1,:]
# plt.plot(average_gr_over_time)
plt.scatter(ruti_strain_grs_table[-1,:], grs)
plt.plot([0,1],[0,1])
plt.show()
plt.plot(average_gr_over_time)
plt.plot(kfir_average_gr_over_time)
plt.plot(ruti_average_gr_over_time)
plt.show()

print(0)
