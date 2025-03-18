import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
N_SPECIES = round(25 * 10**3)         # Number of microbial species
TOTAL_SIZE = round(5.4 * 10**9)       # Total population size (cells)
CELLS_PER_SPECIES = TOTAL_SIZE // N_SPECIES  # Initial cells per species
GR_MEAN = 0.33407776069792355         # Mean growth rate (hr⁻¹)
GR_VAR = 0.0003118415980511718        # Growth rate variance (hr⁻¹)²
DT = 5/60                             # Time step (5 minutes -> 0.0833 hrs)
VOLUME = 90                           # Turbidostat volume (mL)
PUMP_RATE = 80                        # Dilution pump rate (mL/h) 
EXPERIMENT_STEPS = 5000              # Total 5-minute intervals (~179.9 days) 51810
MEASURE_INTERVAL = int(EXPERIMENT_STEPS / 11)  # Measurement frequency (5-min steps)

# Derived parameters (unitless distribution parameters)
sigma = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))  # Lognormal shape parameter
mu = np.log(GR_MEAN) - (sigma**2)/2  # Lognormal scale parameter

# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================

# Load data
exp_data = pd.read_csv(Path("data/simulation_data/experiment_data.csv"))
true_gr = pd.read_csv(Path("data/simulation_data/true_mean_growth_rates.csv"))

# Process timestamps

exp_data['Timestamp'] = pd.to_datetime(exp_data['Timestamp'], format='%d-%m-%Y %H:%M:%S')
true_gr['Timestamp'] = pd.to_datetime(true_gr['Timestamp'], format='%d-%m-%Y %H:%M:%S')

# Extract OD values as NumPy array
od_vals = exp_data['OD_Converted'].values  # [12]

# Calculate increasing subsequences
inc_mask = od_vals[1:] > od_vals[:-1]
inc_breaks = np.where(~inc_mask)[0]
inc_bounds = np.column_stack([np.r_[0, inc_breaks+1], np.r_[inc_breaks+1, len(od_vals)]])
growth_phases = [exp_data.iloc[start:end] for start, end in inc_bounds if (end-start) >= 2]

# Calculate decreasing subsequences
dec_mask = od_vals[1:] < od_vals[:-1]
dec_breaks = np.where(~dec_mask)[0]
dec_bounds = np.column_stack([np.r_[0, dec_breaks+1], np.r_[dec_breaks+1, len(od_vals)]])
pump_phases = [exp_data.iloc[start:end] for start, end in dec_bounds if (end-start) >= 2]

growth_pump_cycles = []
curr_growth_phase = 0
curr_pump_phase = 0

while curr_growth_phase < len(growth_phases):
    growth_phase = growth_phases[curr_growth_phase]
    pump_phase = pump_phases[curr_pump_phase]
    if growth_phase.index[-1] == pump_phase.index[0]:
        growth_pump_cycles.append((growth_phase, pump_phase))
        curr_growth_phase += 1
    else:
        curr_pump_phase += 1

kfir_average_gr_over_time = np.zeros(len(true_gr))
for growth_entry, pump_entry in growth_pump_cycles:
    log_od_difference = np.log(growth_entry['OD_Converted'].values[-1]) - np.log(growth_entry['OD_Converted'].values[0])
    od_time_difference = (growth_entry['Timestamp'].values[-1] - growth_entry['Timestamp'].values[0]) / np.timedelta64(1, 'h')
    kfir_average_gr_over_time[growth_entry.index[0]: pump_entry.index[-1]] = log_od_difference / od_time_difference
 
ruti_adjusted_average_gr_over_time = np.zeros(len(true_gr))
for growth_entry, pump_entry in growth_pump_cycles:
    log_dilution_fraction = np.log(1 - (growth_entry['OD_Converted'].values[-1] - pump_entry['OD_Converted'].values[-1])/(growth_entry['OD_Converted'].values[-1]))
    log_od_difference = (np.log(pump_entry['OD_Converted'].values[-1]) - np.log(growth_entry['OD_Converted'].values[0]))
    od_growth_time_difference = (growth_entry['Timestamp'].values[-1] - growth_entry['Timestamp'].values[0]) / np.timedelta64(1, 'h')
    ruti_adjusted_average_gr_over_time[growth_entry.index[0]: pump_entry.index[-1]] = (log_od_difference - log_dilution_fraction) / od_growth_time_difference

plt.plot(true_gr['Timestamp'], true_gr['True_Mean_GR'], label='True Growth Rate')
plt.plot(exp_data['Timestamp'], kfir_average_gr_over_time, label='KFIR Estimated Growth Rate', linestyle='--')
plt.plot(exp_data['Timestamp'], ruti_adjusted_average_gr_over_time, label='RUTI ADJUSTED Estimated Growth Rate', linestyle=':')
plt.show()
print(0)

