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

SIMULATION = True

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
    averege_gr_over_time = np.genfromtxt(
        "data/simulation_data/averegre_gr_over_time.csv", delimiter=","
    )
else:
    experiment_data = pd.read_csv("csvs_for_drive/experiment_data.csv")
    sizes_total_array = (
        np.array(experiment_data["OD_Converted"]) * 90 * 30 * (10**6)
    ).astype(np.int64)
    pump_rate_array = (
        np.array(experiment_data["Pump_Rate[mL/h]"]) * 2 * 30 * (10**6)
    ).astype(np.int64)

nonzero_indices = np.nonzero(pump_rate_array)[0]
zero_indices = np.where(pump_rate_array == 0)[0]

nonzero_indices = np.split(
    nonzero_indices, np.where(np.diff(nonzero_indices) != 1)[0] + 1
)
zero_indices = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)

# Kfir method
kfir_averege_gr_list = []
for growth_period in zero_indices:
    if growth_period[0] == 0:
        kfir_averege_gr_list.append(0)
        continue
    start = growth_period[0] - 1
    stop = growth_period[-1]
    time_elapsed = DT * (stop - start)
    kfir_gr = np.log(sizes[stop] / sizes[start]) / time_elapsed
    kfir_averege_gr_list.append(kfir_gr)

kfir_averege_gr_over_time = np.zeros(sizes.shape[0])
for i in range(len(nonzero_indices)):
    kfir_gr = kfir_averege_gr_list[i]
    kfir_averege_gr_over_time[zero_indices[i]] = kfir_gr
    kfir_averege_gr_over_time[nonzero_indices[i]] = kfir_gr

# Ruti method
ruti_averege_gr_list = [0]
for i in range(1, len(nonzero_indices)):
    growth_period = zero_indices[i]
    pump_period = nonzero_indices[i]
    start = growth_period[0] - 1
    middle = growth_period[-1]
    stop = pump_period[-1]
    time_elapsed = DT * (stop - start)
    pumped_out_vol = np.sum(pump_rate_array[middle + 1 : stop + 1] * DT)
    ruti_gr = (
        (pumped_out_vol / VOLUME) + np.log(sizes[stop] / sizes[start])
    ) / time_elapsed
    ruti_averege_gr_list.append(ruti_gr)

ruti_averege_gr_over_time = np.zeros(sizes.shape[0])
for i in range(len(nonzero_indices)):
    ruti_gr = ruti_averege_gr_list[i]
    ruti_averege_gr_over_time[zero_indices[i]] = ruti_gr
    ruti_averege_gr_over_time[nonzero_indices[i]] = ruti_gr

plt.plot(averege_gr_over_time)
plt.plot(kfir_averege_gr_over_time)
plt.plot(ruti_averege_gr_over_time)
plt.show()
print(0)
