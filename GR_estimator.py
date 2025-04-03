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

use_simulation = False
# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================

# Load data
if use_simulation:
    exp_data = pd.read_csv(Path("data/simulation_data/experiment_data.csv"))
else:
    exp_data = pd.read_csv(Path("data/experiment_data/experiment_data.csv"))

if use_simulation: true_gr = pd.read_csv(Path("data/simulation_data/true_mean_growth_rates.csv")) 

# Process timestamps

exp_data['Timestamp'] = pd.to_datetime(exp_data['Timestamp'], format='%d-%m-%Y %H:%M:%S')
if use_simulation: true_gr['Timestamp'] = pd.to_datetime(true_gr['Timestamp'], format='%d-%m-%Y %H:%M:%S')

# Extract OD values as NumPy array
od_vals = exp_data['OD_Converted'].values  # [12]

def find_consecutive_sections(arr):
    diffs = np.diff(arr)
    mask = diffs >= 0  # True for >=0 elements, False for negatives
    change_points = np.where(np.diff(mask, prepend=mask[0]-1, append=mask[-1]-1))[0]
    
    pos_groups = []
    neg_groups = []
    
    for start, end in zip(change_points[:-1], change_points[1:]):
        section = np.arange(start, end).tolist()
        (pos_groups if mask[start] else neg_groups).append(section)
    
    return pos_groups, neg_groups

growth_phases, pump_phases = find_consecutive_sections(od_vals)

for entry in growth_phases:
    entry.append(entry[-1]+1)

for entry in pump_phases:
    entry.append(entry[-1]+1)

growth_pump_cycles_indeces = []
curr_growth_phase = 0
curr_pump_phase = 0

while curr_growth_phase < len(growth_phases) and curr_pump_phase < len(pump_phases):
    growth_phase = growth_phases[curr_growth_phase]
    pump_phase = pump_phases[curr_pump_phase]
    if growth_phase[-1] == pump_phase[0]:
        growth_pump_cycles_indeces.append((growth_phase, pump_phase))
        curr_growth_phase += 1
    else:
        curr_pump_phase += 1

growth_pump_cycles = []
for growth_inds, pump_inds in growth_pump_cycles_indeces:
    growth_df = exp_data.loc[growth_inds]
    pump_df = exp_data.loc[pump_inds]
    growth_pump_cycles.append((growth_df, pump_df))

kfir_average_gr_over_time = np.zeros(len(exp_data))
for growth_entry, pump_entry in growth_pump_cycles:
    log_od_difference = np.log(growth_entry['OD_Converted'].values[-1]) - np.log(growth_entry['OD_Converted'].values[0])
    od_time_difference = (growth_entry['Timestamp'].values[-1] - growth_entry['Timestamp'].values[0]) / np.timedelta64(1, 'h')
    kfir_average_gr_over_time[growth_entry.index[0]: pump_entry.index[-1]] = log_od_difference / od_time_difference

ruti_average_gr_over_time = np.zeros(len(exp_data))
for growth_entry, pump_entry in growth_pump_cycles:
# Assuming your dataframe is named df
    temp_df = pump_entry.copy()

    # Shift Pump_Rate and set last entry to 0
    temp_df['Pump_Rate[mL/h]'] = temp_df['Pump_Rate[mL/h]'].shift(-1)
    temp_df.loc[temp_df.index[-1], 'Pump_Rate[mL/h]'] = 0

    # Convert to datetime and sort
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'])
    temp_df = temp_df.sort_values('Timestamp')

    # Calculate time deltas in hours
    temp_df['Delta_Hours'] = (temp_df['Timestamp'].shift(-1) - temp_df['Timestamp']).dt.total_seconds() / 3600

    # Calculate volume contributions and sum
    temp_df['Contribution'] = temp_df['Pump_Rate[mL/h]'] * temp_df['Delta_Hours']
    total_volume = temp_df['Contribution'].sum()

    # Calculate pump volume ratio
    pump_volume_ratio = total_volume / VOLUME

    log_od_difference = np.log(pump_entry['OD_Converted'].values[-1]) - np.log(growth_entry['OD_Converted'].values[0])

    od_time_difference = (pump_entry['Timestamp'].values[-1] - growth_entry['Timestamp'].values[0]) / np.timedelta64(1, 'h')

    ruti_average_gr_over_time[growth_entry.index[0]: pump_entry.index[-1]] = (log_od_difference + pump_volume_ratio) / od_time_difference

ruti_adjusted_average_gr_over_time = np.zeros(len(exp_data))
for growth_entry, pump_entry in growth_pump_cycles:
    log_dilution_fraction = np.log(1 - (growth_entry['OD_Converted'].values[-1] - pump_entry['OD_Converted'].values[-1])/(growth_entry['OD_Converted'].values[-1]))
    log_od_difference = (np.log(pump_entry['OD_Converted'].values[-1]) - np.log(growth_entry['OD_Converted'].values[0]))
    od_growth_time_difference = (growth_entry['Timestamp'].values[-1] - growth_entry['Timestamp'].values[0]) / np.timedelta64(1, 'h')
    ruti_adjusted_average_gr_over_time[growth_entry.index[0]: pump_entry.index[-1]] = (log_od_difference - log_dilution_fraction) / od_growth_time_difference

if use_simulation: plt.plot(true_gr['Timestamp'], true_gr['True_Mean_GR'], label='True Growth Rate')
plt.plot(exp_data['Timestamp'], kfir_average_gr_over_time, label='KFIR Estimated Growth Rate', linestyle='--')
plt.plot(exp_data['Timestamp'], ruti_adjusted_average_gr_over_time, label='RUTI ADJUSTED Estimated Growth Rate', linestyle=':')
plt.plot(exp_data['Timestamp'], ruti_average_gr_over_time, label='RUTI Estimated Growth Rate', linestyle='-.')
plt.show()
print(0)

