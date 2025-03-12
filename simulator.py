import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import lognorm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from pathlib import Path
import time
from scipy.stats import multivariate_hypergeom

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
def initialize():
    """Initialize species with lognormal growth rates"""
    gr = np.sort(lognorm.rvs(s=sigma, scale=np.exp(mu), size=N_SPECIES))
    pop = np.full(N_SPECIES, CELLS_PER_SPECIES, dtype=np.int64)
    return gr, pop

def simulate():
    """Main simulation with comprehensive data collection"""
    rng = np.random.default_rng()
    gr, pop = initialize()
    measurements = []
    true_mean_gr_data = []
    pump_active = False
    start_time = datetime.strptime("01-01-2023 00:00:00", "%d-%m-%Y %H:%M:%S")
    
    freq_data_array = np.zeros((N_SPECIES, 12))
    timestamps = []
    for step in tqdm(range(EXPERIMENT_STEPS), 
                   desc="Simulating experiment", 
                   unit="step",
                   ncols=80):
        # Growth phase
        pop = np.round(pop * np.exp(gr * DT)).astype(np.int64)
        total = pop.sum()
        
        # Pump control logic
        if total > 1.05 * TOTAL_SIZE:
            pump_active = True
        elif total < TOTAL_SIZE:
            pump_active = False
        
        # Dilution using hypergeometric sampling
        if pump_active and total > 0:
            frac = 1 - np.exp(-PUMP_RATE * DT / VOLUME)
            remove_total = int(total * frac)
       
            dist = multivariate_hypergeom(pop, remove_total)
            removed = dist.rvs()
            pop = pop - removed
            if np.any(pop < 0):
                print(0)
            
        
        # Calculate metrics EVERY TIME STEP (5 minutes)
        timestamp = start_time + timedelta(minutes=5*step)
        current_od = pop.sum() / TOTAL_SIZE
        
        # Calculate weighted average of true growth rates
        current_gr = np.sum(pop * gr) / pop.sum() if pop.sum() > 0 else 0
        
        # Record core measurements
        measurements.append({
            "Timestamp": timestamp.strftime("%d-%m-%Y %H:%M:%S"),
            "OD_Converted": current_od,
            "Pump_Rate[mL/h]": PUMP_RATE if pump_active else 0
        })
        
        # Record true mean growth rate
        true_mean_gr_data.append({
            "Timestamp": timestamp.strftime("%d-%m-%Y %H:%M:%S"),
            "True_Mean_GR": current_gr
        })
        
        # Record frequency data
        if step % MEASURE_INTERVAL == 0:
            freq_data_array[:, step // MEASURE_INTERVAL] = pop / pop.sum()
            timestamps.append(timestamp.strftime("%d-%m-%Y %H:%M:%S"))
        # Append to main DataFrame using pd.concat 

    frequency_data = pd.DataFrame(freq_data_array.T, columns=[f'Strain_{i}' for i in range(N_SPECIES)])
    frequency_data['Timestamp'] = timestamps

    return measurements, gr, frequency_data, true_mean_gr_data

def save_results(measurements, growth_rates, frequency_data, mean_gr_data):
    """Save all datasets with compatible structure"""
    data_dir = Path("data/experiment_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(measurements).to_csv(data_dir / "experiment_data.csv", index=False)
    pd.DataFrame({"Original_GRs": growth_rates}).to_csv(data_dir / "true_growth_rates.csv", index=False)
    pd.DataFrame(mean_gr_data).to_csv(data_dir / "true_mean_growth_rates.csv", index=False)
    pd.DataFrame(frequency_data).to_csv(data_dir / "strain_frequencies.csv", index=False)

# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================
def plot_figures(experiment_data, mean_gr_data, freq_data, top_n=10):
    data_dir = Path("data/experiment_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert data to DataFrames
    df_exp = pd.DataFrame(experiment_data)
    df_exp['Timestamp'] = pd.to_datetime(df_exp['Timestamp'], format="%d-%m-%Y %H:%M:%S")
    
    df_gr = pd.DataFrame(mean_gr_data)
    df_gr['Timestamp'] = pd.to_datetime(df_gr['Timestamp'], format="%d-%m-%Y %H:%M:%S")
    
    df_freq = pd.DataFrame(freq_data)
    df_freq['Timestamp'] = pd.to_datetime(df_freq['Timestamp'], format="%d-%m-%Y %H:%M:%S")
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)
    
    # 1. OD over time
    axs[0].plot(df_exp['Timestamp'], df_exp['OD_Converted'], 'b-', linewidth=2)
    axs[0].set_title('Optical Density Over Time', fontsize=14)
    axs[0].set_ylabel('OD (Arbitrary Units)', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].xaxis.set_major_locator(MonthLocator())
    axs[0].xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
    # 2. Growth Rate over time
    axs[1].plot(df_gr['Timestamp'], df_gr['True_Mean_GR'], 'g-', linewidth=2)
    axs[1].set_title('Average Growth Rate Over Time', fontsize=14)
    axs[1].set_ylabel('Growth Rate (hr⁻¹)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].xaxis.set_major_locator(MonthLocator())
    axs[1].xaxis.set_major_formatter(DateFormatter('%b %Y'))
    
    # 3. Area chart of top N strains + others
    # Get strain columns (exclude Timestamp)
    strain_cols = [col for col in df_freq.columns if col.startswith('Strain_')]

    # Ensure all strain columns are numeric
    for col in strain_cols:
        df_freq[col] = pd.to_numeric(df_freq[col], errors='coerce')

    # Identify top N strains based on final frequencies
    final_freqs = df_freq.iloc[-1][strain_cols]
    top_n_strains = final_freqs.sort_values(ascending=False).head(top_n).index.tolist()
    other_strains = [col for col in strain_cols if col not in top_n_strains]

    # Create a new dataframe with top strains and "Others"
    plot_data = df_freq[['Timestamp'] + top_n_strains].copy()
    plot_data['Others'] = df_freq[other_strains].sum(axis=1)
    
    # Set Timestamp as index
    plot_data = plot_data.set_index('Timestamp')
    
    # Create more readable strain labels
    readable_labels = {}
    for i, strain in enumerate(top_n_strains):
        strain_id = strain.split('_')[1]
        readable_labels[strain] = f"Strain {strain_id} (#{i+1})"
    readable_labels['Others'] = 'Others'
    
    # Create stacked area chart with distinct colors
    plot_columns = top_n_strains + ['Others']
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_columns)))
    
    axs[2].stackplot(plot_data.index, 
                     [plot_data[col] for col in plot_columns],
                     labels=[readable_labels[col] for col in plot_columns],
                     colors=colors,
                     alpha=0.8)
    
    axs[2].set_title(f'Relative Abundance of Top {top_n} Strains Over Time', fontsize=14)
    axs[2].set_ylabel('Relative Abundance', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].xaxis.set_major_locator(MonthLocator())
    axs[2].xaxis.set_major_formatter(DateFormatter('%b %Y'))
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Save figure
    plt.savefig(data_dir / "simulation_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {data_dir}/simulation_overview.png")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Run simulation
    experiment_data, original_grs, freq_data, mean_gr_data = simulate()
    
    # Save results
    save_results(experiment_data, original_grs, freq_data, mean_gr_data)
    
    # Loading
    data_dir = Path("data/experiment_data")
    experiment_data = pd.read_csv(data_dir / "experiment_data.csv")
    growth_rates_df = pd.read_csv(data_dir / "true_growth_rates.csv")
    mean_gr_data = pd.read_csv(data_dir / "true_mean_growth_rates.csv")
    freq_data = pd.read_csv(data_dir / "strain_frequencies.csv")

    # Plot figures (can specify number of top strains, default is 10)
    plot_figures(experiment_data, mean_gr_data, freq_data, top_n=10)
    
    print("\nSimulation complete. Output files:")
    print(f"• {Path('data/experiment_data')}/experiment_data.csv")
    print(f"• {Path('data/experiment_data')}/true_growth_rates.csv")
    print(f"• {Path('data/experiment_data')}/true_mean_growth_rates.csv")
    print(f"• {Path('data/experiment_data')}/strain_frequencies.csv")
    print(f"• {Path('data/experiment_data')}/simulation_overview.png")
