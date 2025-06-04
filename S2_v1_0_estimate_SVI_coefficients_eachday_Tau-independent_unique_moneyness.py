# Tau-independent SVI model fit

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed

# Define the SVI model (time-to-maturity independent version)
def svi_model_ind(theta, k):
    a, b, rho, m, sigma = theta[:5]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def objective_function_grid(theta, k_new, iv_obs_grid):
    iv_model_grid = np.empty_like(iv_obs_grid)
    penalty = 0
    epsilon = 1e-10
    for j in range(len(k_new)):
        iv_model_val = svi_model_ind(theta, k_new[j])
        if iv_model_val < -epsilon:
            penalty += 100000
            iv_model_grid[j] = 0
        else:
            iv_model_grid[j] = np.sqrt(max(iv_model_val, 0))
    mse = np.mean((iv_model_grid - iv_obs_grid) ** 2)
    result = np.sqrt(mse) + penalty
    if np.isnan(result):
        print("NaN value detected. Returning a large value.")
        return 1e10
    return result

# SVI constraints
def constraint1(theta):
    return theta[1]

def constraint2(theta):
    return 1 - abs(theta[2])

def constraint3(theta):
    return theta[0] + theta[1] * theta[4] * np.sqrt(1 - theta[2] ** 2)

def constraint4(theta):
    return theta[4]

# Function to process a single CSV file
def process_csv_file(filepath):
    results_list = []
    thetas_list = []
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return results_list, thetas_list

    # Skip files with insufficient columns
    if len(df.columns) <= 2:
        print(f"Skipping {filepath} due to having {len(df.columns)} columns.")
        return results_list, thetas_list

    df = df.transpose()

    # Filter out non-numeric indices and convert to integers
    try:
        filtered_indices = [int(i) for i in df.index if i.isnumeric()]
    except Exception as e:
        print(f"Error processing indices in {filepath}: {e}")
        return results_list, thetas_list

    row_names_array = np.array(filtered_indices)
    iv_real = df.iloc[1:, :] / 100
    iv = iv_real.to_numpy()
    
    constraints = [{'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2},
                   {'type': 'ineq', 'fun': constraint3},
                   {'type': 'ineq', 'fun': constraint4}]
    
    # Prepare data for optimization
    ttm = row_names_array
    k = df.iloc[0, :].to_numpy()
    
    # Process each time-to-maturity (ttm)
    for i_tau, tau_of_interest in enumerate(ttm):
        iv_index = ~np.isnan(iv[i_tau, :])
        iv_new = iv[i_tau, iv_index]    # Grid of IV
        k_new = k[iv_index]             # Grid of moneyness
        
        # Optimization parameters
        theta_guess = 0.05 * np.random.rand(5)
        max_iterations = 4
        bounds = [(-4, 4), (-50, 18), (-2, 2), (-2, 2), (-0.5, 1)]
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        iteration_count = 0
        best_loss = np.inf
        best_thetas = None
        
        while iteration_count < max_iterations:
            iteration_count += 1
            optimized_thetas = []
            losses = []
            for _ in range(10):
                theta_guess[(theta_guess < lower_bounds) | (theta_guess > upper_bounds)] = 0
                res = minimize(objective_function_grid, theta_guess, args=(k_new, iv_new.ravel()),
                               constraints=constraints, method='SLSQP', bounds=bounds)
                optimized_thetas.append(res.x)
                losses.append(res.fun)
                theta_guess = res.x + 0.02 * np.random.rand(5)
            
            best_idx = np.argmin(losses)
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx]
                best_thetas = optimized_thetas[best_idx]
            
            # Compute R^2 for convergence check
            y_obs = iv_new.ravel()
            y_pred = np.array([np.sqrt(svi_model_ind(best_thetas, x)) for x in k_new])
            SS_res = np.sum((y_obs - y_pred) ** 2)
            SS_tot = np.sum((y_obs - y_obs.mean()) ** 2)
            R2 = 1 - (SS_res / SS_tot)
            if R2 >= 0.97:
                break

        # Generate the SVI-implied IV surface for a predefined grid
        k_full = np.linspace(-1, 1, 201)
        iv_svi_of_interest = np.sqrt(np.array([svi_model_ind(best_thetas, k_val) for k_val in k_full]))
        iv_svi_dict = {f"{k_val:.4f}": iv_val for k_val, iv_val in zip(k_full, iv_svi_of_interest)}
        
        result_entry = {'Date': os.path.basename(filepath), 'R2': R2, 'tau': tau_of_interest}
        result_entry.update(iv_svi_dict)
        results_list.append(result_entry)
        thetas_list.append([os.path.basename(filepath), tau_of_interest] + list(best_thetas))
    
    return results_list, thetas_list

# Main execution: parallel processing over CSV files

if __name__ == '__main__':
    iv_folder = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/IV/IV_raw/unique/moneyness"
    csv_files = [os.path.join(iv_folder, f) for f in os.listdir(iv_folder)
                 if f.endswith('.csv') and f.startswith('IV')]

    all_results = []
    all_thetas = []

    """with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_csv_file, filepath) for filepath in csv_files]
        for future in futures:
            try:
                res_list, theta_list = future.result()
                all_results.extend(res_list)
                all_thetas.extend(theta_list)
            except Exception as e:
                print("Error in worker:", e)"""

    
    results = Parallel(n_jobs=-2)(
        delayed(process_csv_file)(filepath)
        for filepath in csv_files
    )
    for res_list, theta_list in results:
        print(res_list, theta_list)
        all_results.extend(res_list)
        all_thetas.extend(theta_list)

    # Check if any results were obtained before proceeding
    if not all_results:
        print("No results obtained.")
    else:
        results_df = pd.DataFrame(all_results).sort_values(by="Date")
        thetas_df = pd.DataFrame(all_thetas, columns=['filename', 'tau', 'a', 'b', 'rho', 'm', 'sigma']).sort_values(by="filename")
        out_dir = '/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/SVI/v1/'
        os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(os.path.join(out_dir, 'svi_Tau-Ind_Mon-Uni_iv_and_r2_results.csv'), index=False)
        thetas_df.to_csv(os.path.join(out_dir, 'svi_Tau-Ind_Mon-Uni_paras.csv'), index=False)