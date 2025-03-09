import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import lognorm
from tqdm import tqdm
from hypergeometric import multivariate_hypergeometric

# Original parameters preserved
N_SPECIES = round(25 * 10**3)
TOTAL_SIZE = round(5.4 * 10**9)
CELLS_PER_SPECIES = TOTAL_SIZE // N_SPECIES
GR_MEAN = 0.33407776069792355
GR_VAR = 0.0003118415980511718
DT = 5/60
VOLUME = 90
PUMP_RATE = 80
EXPERIMENT_STEPS = 51810
MEASURE_INTERVAL = int(EXPERIMENT_STEPS / 11)

# Derived parameters
sigma = np.sqrt(np.log(1 + (GR_VAR / GR_MEAN**2)))
mu = np.log(GR_MEAN) - (sigma**2)/2

def initialize():
    """Initialize species with lognormal growth rates"""
    gr = lognorm.rvs(s=sigma, scale=np.exp(mu), size=N_SPECIES)
    pop = np.full(N_SPECIES, CELLS_PER_SPECIES, dtype=np.int64)
    return gr, pop

def simulate():
    """Main simulation with frequency tracking"""
    gr, pop = initialize()
    measurements = []
    frequency_data = []
    pump_active = False
    start_time = datetime.strptime("01-01-2023 00:00:00", "%d-%m-%Y %H:%M:%S")
    
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
            remove_total = min(int(total * frac), total)
            
            if remove_total > 0:
                removed = multivariate_hypergeometric(pop, remove_total)
                pop = np.maximum(pop - removed, 0)
        
        # Record measurements
        if step % MEASURE_INTERVAL == 0:
            current_total = pop.sum()
            timestamp = start_time + timedelta(minutes=5*step)
            
            # Store experiment measurements
            measurements.append({
                "Timestamp": timestamp.strftime("%d-%m-%Y %H:%M:%S"),
                "OD_Converted": current_total / TOTAL_SIZE,
                "Pump_Rate[mL/h]": PUMP_RATE if pump_active else 0
            })
            
            # Store frequency measurements
            if current_total > 0:
                frequencies = pop / current_total
            else:
                frequencies = np.zeros_like(pop)
            
            freq_entry = {"Timestamp": timestamp.strftime("%d-%m-%Y %H:%M:%S")}
            freq_entry.update({f"Strain_{i}": freq for i, freq in enumerate(frequencies)})
            frequency_data.append(freq_entry)
    
    return measurements, gr, frequency_data

def save_results(measurements, growth_rates, frequency_data):
    """Save all data to CSV files"""
    # Save experiment metrics
    pd.DataFrame(measurements).to_csv("experiment_metrics.csv", index=False)
    
    # Save growth rates
    pd.DataFrame({"Growth_Rate": growth_rates}).to_csv("growth_rates.csv", index=False)
    
    # Save frequencies
    freq_df = pd.DataFrame(frequency_data)
    freq_df.to_csv("strain_frequencies.csv", index=False)

if __name__ == "__main__":
    experiment_metrics, growth_rates, frequency_data = simulate()
    save_results(experiment_metrics, growth_rates, frequency_data)