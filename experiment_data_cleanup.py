import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook

#  od data

wb = load_workbook(filename="data/unformatted_experiment_data/Manager 2154_Un_1_Reich_A_30c_Turbido_0.7.Control.xlsx")

sheet = wb["Data1"]

data = sheet.values
columns = next(data)  # Get header row
experiment_data = pd.DataFrame(data, columns=columns)
experiment_data = experiment_data[['Timestamp', 'ODCX1.PV []','FD1.PV [mL/h]']]
experiment_data = experiment_data.rename(columns={
    'ODCX1.PV []': 'OD_Converted',
    'FD1.PV [mL/h]': 'Pump_Rate[mL/h]'
})

referance = pd.read_csv(
    "data/simulation_data/strain_frequencies.csv"
)
original_data = pd.read_csv(
    "data/unformatted_experiment_data/A_freq_df_minutes.csv"
)

# Full transformation pipeline with time delta calculation
# Full transformation pipeline with time delta replacing timestamps
final_df = (
    original_data.melt(
        id_vars=['Unnamed: 0', 'lineage'],
        var_name='timestamp',
        value_name='measurement'
    )
    .assign(
        # Convert timestamp to numeric and calculate timedelta
        timestamp=lambda x: pd.to_numeric(x['timestamp']) - pd.to_numeric(x['timestamp']).iloc[0],  # Delta in minutes
    )
    .assign(
        # Convert the numeric timestamp delta into timedelta format
        timestamp=lambda x: pd.to_timedelta(x['timestamp'], unit='m')
    )
    .pivot(
        index='timestamp',  # Replace with new timedelta-based timestamp
        columns='lineage',
        values='measurement'
    )
    .reset_index()
)

experiment_data.to_csv('data/experiment_data/experiment_data.csv')
final_df.to_csv('data/experiment_data/strain_frequencies.csv')
print(0)
