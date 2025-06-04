# IV surface
"""
Generate IV curves using the estimated SVI parameters
"""
import os
import pandas as pd
import numpy as np
import re
from joblib import Parallel, delayed

# Set the working directory
path = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
os.chdir(path)

# Read in the SVI estimation results
df1 = pd.read_csv('SVI/v1/svi_Tau-Ind_Mon-Uni_iv_and_r2_results.csv')
df2 = pd.read_csv('SVI/v1/svi_Tau-Ind_Mon-Uni_paras.csv')

# Process paras DataFrame (align indices and rename columns)
paras = df2.loc[df1.index].copy()
paras.columns.values[0] = 'Date'
paras.columns = paras.columns.str.strip()

# Define the SVI model (time-to-maturity independent version)
def svi_model_ind(theta, k):
    base_params = np.array(theta[:5])
    a, b, rho, m, sigma = base_params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

# This function processes a single date.
def process_date(date_val, paras, ttms, k_new):
    # Extract the date string (e.g., "2020-09-23") from the file name
    Date = re.search(r'(\d{4}-\d{2}-\d{2})', date_val).group(0)
    print(f"Estimate IV for date {Date}.")
    all_IVs = pd.DataFrame()
    
    for ttm in ttms:
        # Extract the theta parameters for this date and time-to-maturity
        theta = paras.loc[(paras['Date'] == date_val) & (paras['tau'] == ttm)]
        if theta.empty:
            continue
        theta = theta.drop(theta.columns[0:2], axis=1)  # Drop the 'Date' and 'tau' columns
        theta = np.squeeze(theta.values)
        
        # Calculate IV using the SVI model
        IV = np.sqrt(svi_model_ind(theta, k_new))
        # Convert IV values into a dictionary, then into a DataFrame
        IV_dict = {str(k): [iv] for k, iv in zip(k_new, IV)}
        IV_dict['Date'] = Date
        IV_dict['TTM'] = ttm
        IV_df = pd.DataFrame(IV_dict)
        
        # Skip if NaN values are found
        if IV_df.isna().any().any():
            print(f"NaN values found in IV for {date_val} at TTM {ttm}.")
            continue
        
        # Ensure 'Date' and 'TTM' are the first two columns
        cols = ['Date', 'TTM'] + [col for col in IV_df if col not in ('Date', 'TTM')]
        IV_df = IV_df[cols]
        
        # Append the results for this ttm to the DataFrame for the current date
        all_IVs = pd.concat([all_IVs, IV_df], ignore_index=True)
    
    # Sort and write the IV surface for this date to CSV
    all_IVs = all_IVs.sort_values(by=['Date', 'TTM'])
    output_path = f'IV/IV_SVI/Tau-independent/unique/moneyness_step_0d01/interpolated_{Date}_allR2.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_IVs.to_csv(output_path, index=False)
    return all_IVs

# Define the grid of moneyness and the range of ttms
k_new = np.linspace(-1, 1, 201)
ttms = list(range(3, 121))

# Get the unique date strings from paras (these are the "files" to process)
unique_dates = paras['Date'].unique()
print("Files to process:", unique_dates)

# Process each date in parallel using Joblib
all_IVs_list = Parallel(n_jobs=-2)(
    delayed(process_date)(date_val, paras, ttms, k_new) for date_val in unique_dates
)

# Optionally, concatenate all the individual DataFrames into one master DataFrame and write to CSV
all_IVs_concat = pd.concat(all_IVs_list, ignore_index=True)
output_path_all = 'IV/IV_SVI/Tau-independent/unique/moneyness_step_0d01/interpolated_all_dates_allR2.csv'
os.makedirs(os.path.dirname(output_path_all), exist_ok=True)
all_IVs_concat.to_csv(output_path_all, index=False)