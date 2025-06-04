# Analyze Q density and IV curves, 3-by-2 plot

"""
Parallel version
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from datetime import datetime
from joblib import Parallel, delayed

# -------------------------------
# Helper functions for the new task
# -------------------------------

def get_Q_density_for_date_tau(date_str, tau, Q_data_dir, grid_full):
    """
    Retrieve the Q density curve for a given date and tau by scanning the appropriate directory.
    
    Parameters:
      date_str   : The date string (format: "YYYY-MM-DD").
      tau        : The time-to-maturity (tau) for which to get the Q density.
      Q_data_dir : Base directory where Q density CSV files are stored.
      grid_full  : Grid over which to interpolate the Q density.
      
    Returns:
      Q_interp   : The interpolated (and normalized) Q density vector or None if not found.
    """
    tau_dir = os.path.join(Q_data_dir, f"tau_{tau}")
    if not os.path.exists(tau_dir):
        print(f"Directory {tau_dir} does not exist.")
        return None
    for file in os.listdir(tau_dir):
        if date_str in file and file.endswith(".csv"):
            file_path = os.path.join(tau_dir, file)
            try:
                Q_data = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None
            if 'm' not in Q_data.columns or 'spdy' not in Q_data.columns:
                print(f"Columns missing in {file_path}")
                return None
            # Check that the moneyness range is as expected.
            if np.isclose(Q_data['m'].max(), 1) and np.isclose(Q_data['m'].min(), -1):
                try:
                    interp_func = PchipInterpolator(Q_data['m'], Q_data['spdy'])
                    Q_interp = interp_func(grid_full)
                except Exception as e:
                    print(f"Interpolation error in {file}: {e}")
                    return None
                norm_factor = np.trapezoid(Q_interp, grid_full)
                if norm_factor > 0:
                    Q_interp /= norm_factor
                return Q_interp
    return None

def get_iv_curve_for_date_ttm(date_str, ttm, iv_surface_dir, tol=1e-6):
    """
    Given a date string and a TTM value, search for an IV surface CSV file
    in iv_surface_dir that has the date in its filename.
    Then, extract the estimated IV curve for that TTM.
    Returns (m_grid, estimated_iv) or (None, None) if not found.
    """
    for file in os.listdir(iv_surface_dir):
        if date_str in file and file.endswith(".csv"):
            file_path = os.path.join(iv_surface_dir, file)
            try:
                df_iv = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading IV surface file {file}: {e}")
                continue
            if 'TTM' not in df_iv.columns:
                print(f"Column 'TTM' not found in {file}")
                continue
            df_ttm = df_iv[np.abs(df_iv['TTM'] - ttm) < tol]
            if not df_ttm.empty:
                moneyness_cols = df_ttm.columns[2:]
                try:
                    m_grid = np.array([float(col) for col in moneyness_cols])
                except Exception as e:
                    print(f"Error converting moneyness columns in {file}: {e}")
                    return None, None
                estimated_iv = df_ttm.iloc[0, 2:].values.astype(float)
                return m_grid, estimated_iv
    return None, None

def find_tau_L_H(date_str, df_obs, target_tau=27):
    """
    Find the highest tau <= target_tau (tau_L) and the lowest tau >= target_tau (tau_H)
    that have observed raw IV data on the given date.
    
    Parameters:
      date_str   : The date string (format: "YYYY-MM-DD").
      df_obs     : DataFrame of observed IV data (with a 'date' column).
      target_tau : The target tau (default 27).
      
    Returns:
      (tau_L, tau_H) where each may be None if no candidate is found.
    """
    df_date = df_obs[df_obs['date'] == pd.to_datetime(date_str)]
    if df_date.empty:
        return None, None
    tau_values = df_date['tau'].unique()
    tau_L_candidates = [tau for tau in tau_values if tau <= target_tau]
    tau_H_candidates = [tau for tau in tau_values if tau >= target_tau]
    tau_L = max(tau_L_candidates) if tau_L_candidates else None
    tau_H = min(tau_H_candidates) if tau_H_candidates else None
    return tau_L, tau_H

def plot_tau_comparison(date_str, Q_data_dir, iv_surface_dir, df_obs, grid_full, tol, target_tau=27, save_dir="."):
    """
    For a given date, create a 3×2 figure where each row corresponds to:
      - Row 1: Q density and IV curve for tau = tau_L (highest tau ≤ target_tau with IV observations).
      - Row 2: Q density and IV curve for tau = target_tau.
      - Row 3: Q density and IV curve for tau = tau_H (lowest tau ≥ target_tau with IV observations).
    
    Left column shows the Q density; right column shows the corresponding IV curve with observed IVs.
    
    Parameters:
      date_str      : Date string (format: "YYYY-MM-DD").
      Q_data_dir    : Base directory for Q density CSV files.
      iv_surface_dir: Directory containing IV surface CSV files.
      df_obs        : DataFrame with observed IV data.
      grid_full     : Grid for Q density interpolation.
      tol           : Tolerance used when matching tau.
      target_tau    : The target tau (default 27).
      save_dir      : Directory to save the plot.
    """
    # Determine tau_L and tau_H based on observed IV data.
    tau_L, tau_H = find_tau_L_H(date_str, df_obs, target_tau)
    taus = [tau_L, target_tau, tau_H]
    row_labels = ["Lower", "Target", "Higher"]
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    
    for idx, tau in enumerate(taus):
        label = row_labels[idx]
        # -- Left subplot: Q density curve --
        if tau is not None:
            Q_density = get_Q_density_for_date_tau(date_str, tau, Q_data_dir, grid_full)
        else:
            Q_density = None
        if Q_density is not None:
            axs[idx, 0].plot(grid_full, Q_density, alpha=0.5)
            axs[idx, 0].set_title(f"Q Density ({label} tau = {tau})")
        else:
            axs[idx, 0].text(0.5, 0.5, f"No Q density data for tau = {tau}", 
                             horizontalalignment='center', verticalalignment='center', transform=axs[idx, 0].transAxes, color='red')
        axs[idx, 0].set_xlabel("Moneyness (K/S - 1)")
        axs[idx, 0].set_ylabel("Q Density")
        
        # -- Right subplot: IV curve --
        if tau is not None:
            m_grid, estimated_iv = get_iv_curve_for_date_ttm(date_str, tau, iv_surface_dir, tol)
        else:
            m_grid, estimated_iv = (None, None)
        df_obs_date = df_obs[df_obs['date'] == pd.to_datetime(date_str)]
        # If no raw IV data is observed exactly at the tau of interest, obs_for_tau will be empty.
        obs_for_tau = df_obs_date[np.abs(df_obs_date['tau'] - (tau if tau is not None else target_tau)) < tol]
        if m_grid is not None and estimated_iv is not None:
            axs[idx, 1].plot(m_grid, estimated_iv, label=f"Estimated IV (tau = {tau})", color='blue')
        else:
            axs[idx, 1].text(0.5, 0.5, f"No IV surface data for tau = {tau}", 
                             horizontalalignment='center', verticalalignment='center', transform=axs[idx, 1].transAxes, color='red')
        if not obs_for_tau.empty:
            axs[idx, 1].scatter(obs_for_tau['moneyness'], obs_for_tau['IV'], label="Observed IV", color='red')
        axs[idx, 1].set_title(f"IV Curve ({label} tau = {tau})")
        axs[idx, 1].set_xlabel("Moneyness (K/S - 1)")
        axs[idx, 1].set_ylabel("IV")
        axs[idx, 1].legend()
    
    fig.suptitle(f"Q Density & IV Curves for {date_str}\n(tau_L = {tau_L}, Target tau = {target_tau}, tau_H = {tau_H})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Q_IV_tau_{target_tau}_{date_str}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot for {date_str} at {save_path}")

def process_all_dates_parallel(Q_data_dir, iv_surface_dir, df_obs, grid_full, tol, target_tau=27, save_dir="."):
    """
    Process all available dates (for which Q density files exist for the target_tau) in parallel.
    
    This function extracts dates from the Q density files in the folder corresponding to target_tau
    and then creates the 3×2 comparison plot for each date in parallel.
    """
    tau_dir = os.path.join(Q_data_dir, f"tau_{target_tau}")
    if not os.path.exists(tau_dir):
        print(f"Directory {tau_dir} does not exist.")
        return
    files = sorted(os.listdir(tau_dir))
    dates = []
    for file in files:
        parts = file.split("_")
        if len(parts) >= 3:
            date_part = parts[2].split(".")[0]
            try:
                datetime.strptime(date_part, "%Y-%m-%d")
                dates.append(date_part)
            except Exception:
                continue
    dates = sorted(list(set(dates)))
    
    # Run the plotting function in parallel for each date.
    Parallel(n_jobs=-2)(
        delayed(plot_tau_comparison)(date_str, Q_data_dir, iv_surface_dir, df_obs, grid_full, tol, target_tau, save_dir)
        for date_str in dates
    )

# -------------------------------
# Define parameters and run the new parallel plotting routine
# -------------------------------

if __name__ == "__main__":
    # Set base directories (adjust these paths as needed)
    base_dir = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
    Q_data_dir = os.path.join(base_dir, "Q_from_pure_SVI", "Tau-independent", "unique", "moneyness_step_0d01")
    iv_surface_dir = os.path.join(base_dir, "IV", "IV_surface_SVI", "Tau-independent", "unique", "moneyness_step_0d01")
    obs_data_path = os.path.join(base_dir, "Data", "processed", "20172022_processed_1_3_5_standardized_moneyness.csv")
    save_dir = os.path.join(base_dir, "Q_plots", "Tau-independent", "unique", "moneyness_step_0d01", "preanalysis_3-by-2_tau_comparison")
    
    # Define grid and tolerance.
    grid_full = np.arange(-1, 1.01, 0.01)
    tol = 1e-6
    target_tau = 27
    
    # Load observed IV data.
    df_obs = pd.read_csv(obs_data_path)
    df_obs['date'] = pd.to_datetime(df_obs['date'])
    df_obs = df_obs[(df_obs['moneyness'] >= -1) & (df_obs['moneyness'] <= 1)]
    df_obs = df_obs[(df_obs['tau'] >= 3) & (df_obs['tau'] <= 120)]
    df_obs['IV'] = df_obs['IV'] / 100  # adjust scaling if necessary
    
    # Process all dates in parallel to generate and save the 3x2 plots.
    process_all_dates_parallel(Q_data_dir, iv_surface_dir, df_obs, grid_full, tol, target_tau, save_dir)