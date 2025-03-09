import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('data/experiment_data/experiment_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
df = df.sort_values('Timestamp').reset_index(drop=True)
df['delta_t'] = df['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600

# Configuration
V_total = 90  # Turbidostat volume in mL

# Auto-detect pump cycles (same as before)
pump_changes = df['Pump_Rate[mL/h]'].ne(df['Pump_Rate[mL/h]'].shift()).cumsum()
pump_states = df.groupby(pump_changes).agg(
    start=('Timestamp', 'first'),
    end=('Timestamp', 'last'),
    rate=('Pump_Rate[mL/h]', 'first')
).reset_index(drop=True)

cycles = []
for i in range(len(pump_states)-1):
    if pump_states.loc[i, 'rate'] == 0 and pump_states.loc[i+1, 'rate'] > 0:
        cycles.append({
            'growth_start': pump_states.loc[i, 'start'],
            'growth_end': pump_states.loc[i+1, 'start'],
            'cycle_end': pump_states.loc[i+1, 'end']
        })

# Calculate growth rates (same as before)
results = []
for idx, cycle in enumerate(cycles):
    growth_phase = df[(df.Timestamp >= cycle['growth_start']) & 
                     (df.Timestamp <= cycle['growth_end'])]
    full_cycle = df[(df.Timestamp >= cycle['growth_start']) & 
                   (df.Timestamp <= cycle['cycle_end'])]
    
    if growth_phase.empty or full_cycle.empty:
        continue
        
    N_start = growth_phase['OD_Converted'].iloc[0]
    dt_grow = (cycle['growth_end'] - cycle['growth_start']).total_seconds()/3600
    dt_total = (cycle['cycle_end'] - cycle['growth_start']).total_seconds()/3600
    
    # Kfir's Method
    N_end = growth_phase['OD_Converted'].iloc[-1]
    lambda_k = np.log(N_end/N_start)/dt_grow
    
    # Ruti's Physical Method
    pumped = (full_cycle['Pump_Rate[mL/h]'] * full_cycle.delta_t).sum()
    N_cycle_end = full_cycle['OD_Converted'].iloc[-1]
    lambda_p = (np.log(N_cycle_end/N_start) + pumped/V_total)/dt_total
    
    # Ruti's Adjusted Method
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

# Create line plot visualization
plt.figure(figsize=(10, 6))

# Plot each method with different markers and line styles
plt.plot(results_df['Cycle'], results_df['Kfir'], 'o-', color='#2ca02c', 
         markersize=8, linewidth=2, label='Kfir')
plt.plot(results_df['Cycle'], results_df['Physical'], 's--', color='#1f77b4', 
         markersize=8, linewidth=2, label='Physical')
plt.plot(results_df['Cycle'], results_df['Adjusted'], '^:', color='#ff7f0e', 
         markersize=8, linewidth=2, label='Adjusted')

plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('Growth Rate (hr⁻¹)', fontsize=12)
plt.title('Average Growth Rate Trends', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Set x-axis to integer cycle numbers
plt.xticks(results_df['Cycle'].astype(int))

plt.tight_layout()
plt.show()

# Print results (same as before)
print("\nAverage Growth Rate Results:")
print(results_df.round(4).to_string(index=False))