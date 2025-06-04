# Interpolate IV curves for unobserved TTM values by linear interpolation

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Set work directory
path = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
os.chdir(path)

# Directory where the original IV curve files are stored
iv_path = "IV/IV_SVI/Tau-independent/unique/moneyness_step_0d01/"
files = sorted([f for f in os.listdir(iv_path) if f.endswith(".csv")])

# New directory to save interpolated IV surface
interpolated_iv_path = "IV/IV_surface_SVI/Tau-independent/unique/moneyness_step_0d01/"
os.makedirs(interpolated_iv_path, exist_ok=True)

def process_file(file):
    """Function to process each file in parallel."""
    date = file.split("_")[1]  # Extract date from filename
    file_path = os.path.join(iv_path, file)
    
    # Read the IV file
    df_iv = pd.read_csv(file_path)

    if "TTM" in df_iv.columns:
        # Sort by TTM
        df_iv = df_iv.sort_values(by="TTM")

        # Generate a complete TTM range
        ttm_min, ttm_max = df_iv["TTM"].min(), df_iv["TTM"].max()
        full_ttm = pd.DataFrame({"Date": date, "TTM": np.arange(ttm_min, ttm_max + 1)})

        # Merge to get all TTM values
        df_iv_full = full_ttm.merge(df_iv, on=["Date", "TTM"], how="left")

        # Interpolate only numeric columns (exclude "Date")
        numeric_cols = df_iv_full.select_dtypes(include=[np.number]).columns
        df_iv_full[numeric_cols] = df_iv_full[numeric_cols].interpolate(method="linear")

        # Fill any remaining NaNs (if present at edges)
        df_iv_full.ffill(inplace=True)
        df_iv_full.bfill(inplace=True)

        # Reorder columns
        column_order = ["Date", "TTM"] + [col for col in df_iv_full.columns if col not in ["Date", "TTM"]]
        df_iv_full = df_iv_full[column_order]

        # Save the new IV file
        output_file = os.path.join(interpolated_iv_path, f"interpolated_{date}_allR2.csv")
        df_iv_full.to_csv(output_file, index=False)
        print(f"Saved interpolated IV file for {date} to {output_file}")

Parallel(n_jobs=-2)(
    delayed(process_file)(file) for file in files
)

print("Interpolation completed for all dates.")