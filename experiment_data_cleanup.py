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

# Use melt to pivot columns into rows
reshaped_df = original_data.melt(
    id_vars=['Unnamed: 0', 'lineage'],
    var_name='timestamp',
    value_name='measurement'
)

# Pivot to get lineages as columns
final_df = reshaped_df.pivot(
    index='timestamp',
    columns='lineage',
    values='measurement'
).reset_index()

frequencies = original_data.values
frequencies = frequencies[:, 1:].astype(str)
frequencies = np.char.replace(frequencies, "#NAME?", "0")
frequencies = frequencies.astype(float)
frequencies = np.nan_to_num(frequencies, nan=0, posinf=0, neginf=0)
timestamps = original_data.columns.to_list()[1:]
timestamps = [float(i) for i in timestamps]
transformed_data = pd.DataFrame(frequencies, columns=timestamps)
transformed_data.to_csv("frequency_data.csv", index=False)
print(0)
