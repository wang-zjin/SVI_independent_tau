# Visualize R2 as a function of tau and date

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ====== STEP 1: Load and preprocess your market data ======
path = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
os.chdir(path)

data_path = os.path.join(path, "SVI", "v1", "svi_Tau-Ind_Mon-Uni_iv_and_r2_results.csv")
df = pd.read_csv(data_path)

# Convert 'Date' column to actual datetime format
df['Date'] = pd.to_datetime(df['Date'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

# Group by Date and tau, calculate the average R2 for each day
average_r2_by_day_tau = df.groupby(['Date', 'tau'])['R2'].mean().reset_index()

# Identify and remove outliers using IQR method
Q1 = average_r2_by_day_tau['R2'].quantile(0.25)
Q3 = average_r2_by_day_tau['R2'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
filtered_r2_by_day_tau = average_r2_by_day_tau[
    (average_r2_by_day_tau['R2'] >= lower_bound) & (average_r2_by_day_tau['R2'] <= upper_bound)
]

print(f"The number of outliers: {sum( (average_r2_by_day_tau['R2'] <= lower_bound) | (average_r2_by_day_tau['R2'] >= upper_bound) )}")

# Pivot the DataFrame to create a matrix for the heatmap
heatmap_data = filtered_r2_by_day_tau.pivot_table(index='tau', columns='Date', values='R2')

# Reverse the y-axis to show lowest tau at the bottom and highest tau at the top
heatmap_data = heatmap_data.sort_index(ascending=False)

# ====== STEP 2: Plot the 2-D heatmap ======
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='viridis', cbar=True, vmin=0, vmax=1)
plt.title('Tau-independent SVI $R^2$ by Tau and Date (Outliers Removed)')
plt.xlabel('Date')
plt.ylabel('Tau (in days)')

# Set y-axis ticks to show only a few labels, including the first and last
y_tick_positions = np.linspace(0, len(heatmap_data.index) - 1, 8, dtype=int)
y_tick_labels = heatmap_data.index[y_tick_positions]
plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)

plt.tight_layout()

# Set x-axis ticks to show fewer dates for better readability
"""plt.xticks(ticks=np.linspace(0, len(heatmap_data.columns) - 1, 10, dtype=int),
           labels=pd.date_range(start=heatmap_data.columns.min(), 
                                end=heatmap_data.columns.max(), periods=10).strftime('%Y-%m-%d'),
           rotation=45)"""

# Ensure only years that exist in heatmap_data.columns are used
unique_years = [year for year in pd.date_range(start=heatmap_data.columns.min(), 
                                               end=heatmap_data.columns.max(), 
                                               freq='YS').year 
                if str(year) + '-01-01' in heatmap_data.columns]

x_tick_positions = [heatmap_data.columns.get_loc(str(year) + '-01-01') for year in unique_years]

plt.xticks(ticks=x_tick_positions, labels=unique_years, rotation=45)

path_to_save = os.path.join(path, "SVI", "v1", "fitness")
os.makedirs(path_to_save, exist_ok=True)
plt.savefig(os.path.join(path_to_save, "r2_heatmap_by_day_Tau-Ind_Mon-Uni.png"))

plt.show()


