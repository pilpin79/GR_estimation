"""
Yeast Turbidostat Growth Rate Analyzer
Author: [Your Name]
Date: [Date]
Modified for DD-MM-YYYY timestamp format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================
# Configuration
# =============================================
USE_MOCK_DATA = True  # Switch between real/mock data
DATA_DIR = Path("data/simulation_data")  # Update this path as needed
PLOT_STYLE = "seaborn-v0_8"  # Modern matplotlib style
V_total = 90  # Turbidostat volume in mL

# =============================================
# Data Loading Function
# =============================================
def load_turbidostat_data(use_mock: bool) -> tuple:
    """Load dataset based on simulation flag"""
    file_map = {
        'experiment': DATA_DIR / "experiment_data.csv",
        'growth_rates': DATA_DIR / "growth_rates.csv",
        'mean_rates': DATA_DIR / "true_mean_growth_rates.csv",
        'frequencies': DATA_DIR / "strain_frequencies.csv"
    }
    
    if use_mock:
        print("💻 Loading simulated data...")
        return (
            pd.read_csv(file_map['experiment']),
            pd.read_csv(file_map['growth_rates']),
            pd.read_csv(file_map['mean_rates']),
            pd.read_csv(file_map['frequencies'])
        )
    else:
        print("🔬 Loading experimental data...")
        return (
            pd.read_csv(file_map['experiment']),
            None, None, None  # Placeholders for real data
        )

# =============================================
# Main Processing Pipeline
# =============================================
def main():
    plt.style.use(PLOT_STYLE)
    
    # Load data
    df, gr_truth, mean_gr_truth, freq_truth = load_turbidostat_data(USE_MOCK_DATA)
    
    # Preprocess timestamps (modified format)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['delta_t'] = df['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600

    # Detect pump cycles
    pump_changes = df['Pump_Rate[mL/h]'].ne(df['Pump_Rate[mL/h]'].shift()).cumsum()
    pump_states = df.groupby(pump_changes).agg(
        start=('Timestamp', 'first'),
        end=('Timestamp', 'last'),
        rate=('Pump_Rate[mL/h]', 'first')
    ).reset_index(drop=True)

    # Identify growth/dilution cycles
    cycles = []
    for i in range(len(pump_states)-1):
        if pump_states.loc[i, 'rate'] == 0 and pump_states.loc[i+1, 'rate'] > 0:
            cycles.append({
                'growth_start': pump_states.loc[i, 'start'],
                'growth_end': pump_states.loc[i+1, 'start'],
                'cycle_end': pump_states.loc[i+1, 'end']
            })

    # Calculate growth rates for each cycle
    results = []
    for idx, cycle in enumerate(cycles):
        # Extract phase data
        growth_phase = df[(df.Timestamp >= cycle['growth_start']) & 
                         (df.Timestamp <= cycle['growth_end'])]
        full_cycle = df[(df.Timestamp >= cycle['growth_start']) & 
                       (df.Timestamp <= cycle['cycle_end'])]
        
        if growth_phase.empty or full_cycle.empty:
            continue
            
        # Get key parameters
        N_start = growth_phase['OD_Converted'].iloc[0]
        dt_grow = (cycle['growth_end'] - cycle['growth_start']).total_seconds()/3600
        dt_total = (cycle['cycle_end'] - cycle['growth_start']).total_seconds()/3600
        
        # Kfir's Method (Growth phase only)
        N_end = growth_phase['OD_Converted'].iloc[-1]
        lambda_k = np.log(N_end/N_start)/dt_grow
        
        # Ruti's Physical Method (Full cycle)
        pumped = (full_cycle['Pump_Rate[mL/h]'] * full_cycle.delta_t).sum()
        N_cycle_end = full_cycle['OD_Converted'].iloc[-1]
        lambda_p = (np.log(N_cycle_end/N_start) + pumped/V_total)/dt_total
        
        # Ruti's Adjusted Method (OD-based dilution)
        N_max = N_end
        N_stop = full_cycle['OD_Converted'].iloc[-1]
        phi = (N_max - N_stop)/N_max
        lambda_a = (np.log(N_stop/N_start) - np.log(1 - phi))/dt_grow
        
        results.append({
            'Cycle': idx+1,
            'Kfir': lambda_k,
            'Physical': lambda_p,
            'Adjusted': lambda_a
        })

    results_df = pd.DataFrame(results)

    # =============================================
    # Visualization
    # =============================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot estimation methods
    methods = {
        'Kfir': ('o-', '#2ca02c'),
        'Physical': ('s--', '#1f77b4'),
        'Adjusted': ('^:', '#ff7f0e')
    }
    for method, (style, color) in methods.items():
        ax.plot(results_df['Cycle'], results_df[method], style,
                color=color, markersize=8, linewidth=2, label=method)

    # Add ground truth if available
    if USE_MOCK_DATA and mean_gr_truth is not None:
        true_gr = mean_gr_truth['True_Mean_GR'].iloc[0]
        ax.axhline(true_gr, color='k', linestyle='--', linewidth=1.5, 
                   label='True Mean')
        ax.fill_between(results_df['Cycle'], true_gr*0.95, true_gr*1.05,
                        color='gray', alpha=0.1)

    # Formatting
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Growth Rate (hr⁻¹)', fontsize=12)
    title = 'Growth Rate Estimation - ' + ('Simulation' if USE_MOCK_DATA else 'Experiment')
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(results_df['Cycle'].astype(int))
    plt.tight_layout()
    plt.show()

    # =============================================
    # Reporting
    # =============================================
    print("\n▶ Growth Rate Estimation Results")
    print(results_df.round(4).to_string(index=False))

    if USE_MOCK_DATA:
        print("\n🔍 Simulation Validation")
        print("True Individual Growth Rates:")
        print(gr_truth.describe().round(4))
        
        if freq_truth is not None:
            print("\nFinal Strain Frequencies:")
            print(freq_truth.tail(3).round(4))

# =============================================
# Execution
# =============================================
if __name__ == "__main__":
    main()
