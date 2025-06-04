# Plot IV curves in the same figure

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Directory with the interpolated IV files
interpolated_iv_path = "IV/IV_surface_SVI/Tau-independent/unique/moneyness_step_0d01/"
files = sorted(glob.glob(os.path.join(interpolated_iv_path, "interpolated_*_allR2.csv")))
plot_to_save_path = "IV/IV_curve_SVI_average_plot/Tau-independent/unique/moneyness_step_0d01/"
os.makedirs(plot_to_save_path, exist_ok=True)

plt.figure(figsize=(10, 6))

# Lists to store all IV curves and corresponding moneyness values
all_iv_curves = []
common_moneyness = None

for file in files:
    df = pd.read_csv(file)
    # Filter the DataFrame to only include rows where TTM == 27
    df_tau = df[df["TTM"] == 27]
    if df_tau.empty:
        continue
    
    # There should be one row per file with TTM == 27
    row = df_tau.iloc[0]
    
    # Extract IV values by dropping the "Date" and "TTM" columns
    iv_values = row.drop(labels=["Date", "TTM"])
    
    # Convert the remaining column names (assumed moneyness) to floats
    try:
        moneyness = np.array([float(col) for col in iv_values.index])
    except ValueError:
        moneyness = iv_values.index.astype(float)
    
    iv = iv_values.values.astype(float)
    
    # Sort by moneyness in case columns are not in order
    sort_idx = np.argsort(moneyness)
    moneyness = moneyness[sort_idx]
    iv = iv[sort_idx]
    
    # Plot individual IV curve in gray
    date_label = row["Date"]
    plt.plot(moneyness, iv, label=date_label, alpha=0.1, color="gray")
    
    # Store the IV curve for averaging if it doesn't contain NaNs, othersie print the date
    if np.isnan(iv).any():
        print(f"NaN values found in IV curve for {date_label}")
    else:
        all_iv_curves.append(iv)
    if common_moneyness is None:
        common_moneyness = moneyness

# Compute and plot the average IV curve if any curves were found
if all_iv_curves:
    avg_iv = np.nanmean(all_iv_curves, axis=0)
    plt.plot(common_moneyness, avg_iv, color="black", linewidth=2, label="Average IV")

plt.xlabel("Moneyness")
plt.ylabel("Implied Volatility")
plt.title("IV Curves for Tau = 27 with Average IV")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(plot_to_save_path, "IV_curve_SVI_overall_average_Tau-independent_unique_moneyness_step_0d01_TTM_27.png"))
plt.show()