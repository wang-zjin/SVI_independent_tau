# R2 time series

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ====== STEP 1: Load and preprocess your market data ======
path = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
os.chdir(path)

data_path = os.path.join(path, "SVI", "v1", "svi_Tau-Ind_Mon-Uni_iv_and_r2_results.csv")
df = pd.read_csv(data_path)

# Convert 'Date' column to actual datetime format
df['Date'] = pd.to_datetime(df['Date'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

# Group by Date and tau, calculate the average R2 for each day and tau
average_r2_by_day_tau = df.groupby(['Date', 'tau'])['R2'].mean()

# Identify and remove outliers using IQR
Q1 = average_r2_by_day_tau.quantile(0.25)
Q3 = average_r2_by_day_tau.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data by removing outliers
filtered_r2_by_day_tau = average_r2_by_day_tau[
    (average_r2_by_day_tau >= lower_bound) & (average_r2_by_day_tau <= upper_bound)
]

# ====== STEP 2: Calculate average R2 for each date based on the filtered data ======
average_r2_by_day = filtered_r2_by_day_tau.groupby('Date').mean()

# ====== STEP 3: Plot the time series of average R2 by day ======
plt.figure(figsize=(14, 7))
plt.plot(average_r2_by_day.index, average_r2_by_day.values, marker='o', linestyle='-', color='b')
plt.title('Time Series of Average $R^2$ by Day (Tau-independent SVI)')
plt.xlabel('Date')
plt.ylabel('Average $R^2$')
plt.ylim(-0.05, 1.05)
plt.yticks(np.linspace(0, 1, 6))
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

path_to_save = os.path.join(path, "SVI", "v1", "fitness", "R2_TimeSeries")
os.makedirs(path_to_save, exist_ok=True)
plt.savefig(os.path.join(path_to_save, "r2_ts_by_day_Tau-Ind_Mon-Uni.png"))

plt.show()

# Print results
print("Average R2 for each date (filtered):")
print(average_r2_by_day)
print(f"Average R2 for all dates: {average_r2_by_day.mean()}")
print(f"The standard deviation for all dates: {average_r2_by_day.std()}")