import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
N_OF_SPECIES = round(25 * 10**3)
TOTAL_SIZE = round(5.4 * 10**9)
SIZE_PER_SPECIES = int(TOTAL_SIZE / N_OF_SPECIES)
GR_MEAN = 0.33407776069792355
GR_VAR = 0.0003118415980511718
SIGMA = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))
MU = np.log(GR_MEAN) - (SIGMA**2) / 2
EXPERIMENT_LENGTH = 51810
DT = 5 / 60
DT_OF_SAMPLE = 5 / 600
VOLUME = 90
PUMP_OUT_RATE = 80

# Configuration flags
SIMULATION = False
USE_PUMP_DATA = False

def load_growth_data():
    """Load growth rate data from CSV files."""
    grs = np.genfromtxt("data/simulation_data/grs.csv", delimiter=",")
    avg_gr = np.genfromtxt("data/simulation_data/averegre_gr_over_time.csv", delimiter=",")
    return grs, avg_gr

def load_data(simulation=True):
    """Load experimental or simulation data."""
    if simulation:
        sizes = np.genfromtxt("data/simulation_data/sizes_sum.csv", delimiter=",")
        pump_rates = np.genfromtxt("data/simulation_data/total_pump_array.csv", delimiter=",") * PUMP_OUT_RATE
        frequencies = np.genfromtxt("data/simulation_data/freq_samples.csv", delimiter=",")
        timestamps = np.arange(0, 500 * frequencies.shape[0], 500)
    else:
        exp_data = pd.read_csv("data/experiment_data/experiment_data.csv")
        sizes = (exp_data["OD_Converted"].values * 90 * 30 * 1e6).astype(int)[11:]
        pump_rates = (exp_data["Pump_Rate[mL/h]"].values * 2 * 30 * 1e6).astype(int)[11:]
        freq_table = pd.read_csv("data/experiment_data/frequency_data.csv")
        timestamps = [int(float(i) * 12) for i in freq_table.columns]
        frequencies = freq_table.values.T
    return sizes, pump_rates, frequencies, timestamps

def get_indices(sizes, pump_rates, use_pump=True):
    """Identify growth and pump periods from data."""
    if use_pump:
        nonzero = np.nonzero(pump_rates)[0]
        zero = np.where(pump_rates == 0)[0]
    else:
        diffs = np.diff(sizes)
        zero = np.where(diffs > 0)[0] + 1
        nonzero = np.where(diffs <= 0)[0] + 1

    nonzero = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0] + 1)
    zero = np.split(zero, np.where(np.diff(zero) != 1)[0] + 1)

    zero = [run.tolist() for run in zero if run.size >= 1]
    nonzero = [run.tolist() for run in nonzero if run.size >= 1]

    for i in range(len(zero)):
        arr = zero[i]
        zero[i] = np.insert(arr, 0, max(arr[0]-1, 0))

    if use_pump and zero:
        zero[0] = zero[0][1:]

    return zero, nonzero

def calculate_kfir_growth(zero_indices, nonzero_indices, sizes):
    """Calculate growth rates using Kfir's method."""
    growth_rates = []
    for period in zero_indices:
        start, stop = period[0], period[-1]
        dt = DT * (stop - start)
        gr = np.log(sizes[stop]/sizes[start]) / dt
        growth_rates.append(gr)
        if gr < 0: print("Negative growth rate detected")

    gr_over_time = np.zeros_like(sizes)
    for i, gr in enumerate(growth_rates):
        gr_over_time[zero_indices[i]] = gr
        gr_over_time[nonzero_indices[i]] = gr
    
    return gr_over_time[:-1] if gr_over_time[-1] == 0 else gr_over_time

def calculate_ruti_growth(zero_indices, nonzero_indices, sizes, pump_rates):
    """Calculate growth rates using Ruti's physical dilution method."""
    growth_rates = []
    for i, (growth_period, pump_period) in enumerate(zip(zero_indices, nonzero_indices)):
        start = growth_period[0]
        middle = growth_period[-1]
        stop = pump_period[-1]
        dt_total = DT * (stop - start)
        
        pumped_vol = np.sum(pump_rates[middle+1:stop+1] * DT)
        gr = (np.log(sizes[stop]/sizes[start]) + pumped_vol/VOLUME) / dt_total
        growth_rates.append(gr)

    gr_over_time = np.zeros_like(sizes)
    for i, gr in enumerate(growth_rates):
        gr_over_time[zero_indices[i]] = gr
        gr_over_time[nonzero_indices[i]] = gr
    
    return gr_over_time[:-1] if gr_over_time[-1] == 0 else gr_over_time

def calculate_ruti_adjusted_growth(zero_indices, nonzero_indices, sizes):
    """Calculate growth rates using Ruti's adjusted dilution method."""
    growth_rates = []
    for i, (growth_period, pump_period) in enumerate(zip(zero_indices, nonzero_indices)):
        start = growth_period[0]
        middle = growth_period[-1]
        stop = pump_period[-1]
        dt = DT * (middle - start)
        
        dilution_term = np.log(1 - (sizes[middle] - sizes[stop])/sizes[middle])
        gr = (np.log(sizes[stop]/sizes[start]) - dilution_term) / dt
        growth_rates.append(gr)

    gr_over_time = np.zeros_like(sizes)
    for i, gr in enumerate(growth_rates):
        gr_over_time[zero_indices[i]] = gr
        gr_over_time[nonzero_indices[i]] = gr
    
    return gr_over_time[:-1] if gr_over_time[-1] == 0 else gr_over_time

def plot_comparison(sizes, pump_rates):
    """Generate comparison plots."""
    # Pump rate vs OD plot
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    time_points = np.arange(800, 900)
    ax1.plot(5*time_points, pump_rates[800:900], 'b-', label='Pump Rate')
    ax1.set_ylabel('Pump Rate (mL/h)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(5*time_points, sizes[800:900], 'r-', label='OD')
    ax2.set_ylabel('Optical Density', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Pump Rate vs Optical Density Comparison (Time 800-900)')
    plt.show()

    # Growth rate comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(kfir_gr, label="Kfir's Method", alpha=0.7)
    plt.plot(ruti_gr, label="Ruti's Physical Dilution", alpha=0.7)
    plt.plot(ruti_adjusted_gr, label="Ruti's Adjusted Dilution", alpha=0.7)
    plt.title('Growth Rate Calculation Method Comparison')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Growth Rate (1/hour)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load and process data
    sizes, pump_rates, frequencies, timestamps = load_data(SIMULATION)
    zero_idx, nonzero_idx = get_indices(sizes, pump_rates, USE_PUMP_DATA)

    # Calculate growth rates using all methods
    kfir_gr = calculate_kfir_growth(zero_idx, nonzero_idx, sizes)
    ruti_gr = calculate_ruti_growth(zero_idx, nonzero_idx, sizes, pump_rates)
    ruti_adjusted_gr = calculate_ruti_adjusted_growth(zero_idx, nonzero_idx, sizes)

    # Generate plots
    plot_comparison(sizes, pump_rates)