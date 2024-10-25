import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# frequency cleanup

original_data = pd.read_csv('data/experiment_data/growth_rate_per_strain__replica_A.csv')
frequencies = original_data.values
frequencies = frequencies[:,1:].astype(str)
frequencies = np.char.replace(frequencies, '#NAME?', '0')
frequencies = frequencies.astype(float)
frequencies = np.nan_to_num(frequencies, nan=0, posinf=0, neginf=0)
timestamps = original_data.columns.to_list()[1:]
timestamps = [float(i) for i in timestamps]
transformed_data = pd.DataFrame(frequencies, columns=timestamps)
transformed_data.to_csv('frequency_data.csv', index=False)
print(0)