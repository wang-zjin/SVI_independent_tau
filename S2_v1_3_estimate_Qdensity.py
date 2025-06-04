# Q density

import os
import pandas as pd
import numpy as np
from src.quandl_data import get_btc_prices_2015_2022
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def estimate_Q(log_ret, IV, dIVdr, d2IVdr2, rf, tau, r_obj, out_dir):                            
    try:
        # Call the R function from 'src/Q_from_IV.R'
        moneyness, spd, logret, spd_logret, volas, cdf_m, cdf_ret, sigmas1, sigmas2 = r_obj.estimate_Q_from_IV(
            robjects.FloatVector(log_ret),
            robjects.FloatVector(IV[0]), 
            robjects.FloatVector(dIVdr), 
            robjects.FloatVector(d2IVdr2), 
            robjects.FloatVector(rf),
            robjects.FloatVector([tau]),
            robjects.StrVector([out_dir])
        )
        moneyness = np.array(moneyness) - 1
        spd_df = pd.DataFrame({
            'm': moneyness,
            'spdy': spd,
            'ret': logret,
            'spd_ret': spd_logret,
            'volatility': volas,
            'cdf_m': cdf_m,
            'cdf_ret': cdf_ret,
            'sigma_prime': sigmas1,
            'sigma_double_prime': sigmas2
        })
        return spd_df
    except Exception as e:
        print('Exception in estimate_Q:', e)
        return None

def process_file(file, ttm, out_path, interest_rate_data, iv_dir):
    tau = ttm / 365.0
    out_dir = os.path.join(out_path, f'tau_{ttm}')
    os.makedirs(out_dir, exist_ok=True)
    
    file_path = os.path.join(iv_dir, file)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Failed to read file {file}: {e}"
    
    if 'TTM' not in df.columns:
        return f"TTM column not found in {file}."
    if ttm not in df['TTM'].unique():
        return f"TTM value {ttm} not in file {file}."
    
    df = df[df["TTM"] == ttm]
    
    try:
        date = file.split("_")[1]
    except IndexError:
        return f"Filename {file} does not contain expected date info."
    
    # Assume the first two columns are 'Date' and 'TTM'
    try:
        logret = np.log(df.columns[2:].astype(float) + 1 + 1e-10).to_numpy()
        #logret = np.log(df.columns[2:].astype(float) + 1).to_numpy()
    except Exception as e:
        return f"Error converting column names to floats for log returns in file {file}: {e}"
    
    IV_filtered = df[df['Date'] == date]
    if IV_filtered.empty:
        return f"No data for date {date} in file {file}."
    IV = IV_filtered.drop(columns=['Date', 'TTM']).to_numpy()
    
    try:
        dIVdr = (IV[0][2:] - IV[0][:-2]) / (logret[2:] - logret[:-2])
        dIVdr = [float('nan')] + list(dIVdr) + [float('nan')]
        dIVdr[0] = (IV[0][1] - IV[0][0]) / (logret[1] - logret[0])
        dIVdr[-1] = (IV[0][-1] - IV[0][-2]) / (logret[-1] - logret[-2])
    except Exception as e:
        return f"Error computing first derivative for file {file}: {e}"
    
    try:
        d2IVdr2 = (IV[0][2:] - 2 * IV[0][1:-1] + IV[0][:-2]) / ((logret[2:] - logret[:-2]) ** 2)
        d2IVdr2 = [float('nan')] + list(d2IVdr2) + [float('nan')]
        d2IVdr2[0] = (IV[0][2] - 2 * IV[0][1] + IV[0][0]) / ((logret[1] - logret[0]) ** 2)
        d2IVdr2[-1] = (IV[0][-1] - 2 * IV[0][-2] + IV[0][-3]) / ((logret[-1] - logret[-2]) ** 2)
    except Exception as e:
        return f"Error computing second derivative for file {file}: {e}"
    
    rf = interest_rate_data.interest_rate[interest_rate_data['date'] == date].to_numpy()
    if rf.size == 0:
        return f"No risk-free rate data for date {date} in file {file}."
    
    # Reinitialize R environment in this worker
    try:
        from rpy2 import robjects
        r_obj = robjects.r
        r_obj.source('src/Q_from_IV.R')
    except Exception as e:
        return f"Error initializing R in file {file}: {e}"
    
    spd_btc = estimate_Q(logret.tolist(), IV, dIVdr, d2IVdr2, rf, tau, r_obj, out_dir)
    if spd_btc is None:
        return f"Failed to compute Q for date {date} in file {file}."
    
    try:
        output_file = os.path.join(out_dir, f"btc_Q_{date}.csv")
        spd_btc.to_csv(output_file, index=False, float_format="%.4f")
    except Exception as e:
        return f"Error saving CSV for date {date} in file {file}: {e}"
    
    return f"Successfully processed file {file} for date {date}"

def run_parallel_processing():
    base_path = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
    os.chdir(base_path)
    
    # Load interest rate data once
    interest_rate_data = pd.read_csv(os.path.join('Data', 'IR_daily.csv'))
    interest_rate_data = interest_rate_data.rename(columns={'index': 'date', 'DTB3': 'interest_rate'})
    interest_rate_data['interest_rate'] = interest_rate_data['interest_rate'] / 100
    
    iv_dir = os.path.join(base_path, 'IV', 'IV_surface_SVI', 'Tau-independent', 'unique', 'moneyness_step_0d01')
    out_path = os.path.join(base_path, 'Q_from_pure_SVI', 'Tau-independent', 'unique', 'moneyness_step_0d01')
    
    files = sorted([f for f in os.listdir(iv_dir) if f.endswith(".csv")])
    ttm_list = range(3, 121)
    
    for ttm in ttm_list:
        print(f"Processing TTM = {ttm} ...")
        results = Parallel(n_jobs=-2)(
            delayed(process_file)(file, ttm, out_path, interest_rate_data, iv_dir)
            for file in files
        )
        for result in results:
            print(result)

# Call the function from a notebook cell
run_parallel_processing()