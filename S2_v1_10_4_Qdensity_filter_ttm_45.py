# Calculate TTM = 45 separately

"""
Set smoothness value threshold to 0.006
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import scipy.signal as signal

# -------------------------------
# Helper functions
# -------------------------------

def compute_density_moments(ret, density, ttm):
    """
    Computes the first four moments (mean, variance, skewness, kurtosis)
    of a Q density curve given on a grid ret. Moments are scaled by 365/ttm.
    """
    m1 = np.trapezoid(density * ret, ret)
    m2 = np.trapezoid(density * (ret - m1)**2, ret)
    m3 = np.trapezoid(density * (ret - m1)**3, ret)
    m4 = np.trapezoid(density * (ret - m1)**4, ret)
    
    Mean = m1 * 365 / ttm
    Variance = m2 * 365 / ttm
    Skewness = m3 / (m2**1.5 + 1e-12)
    Kurtosis = m4 / (m2**2 + 1e-12) - 3
    return np.array([Mean, Variance, Skewness, Kurtosis])

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

def plot_combined_curves(indices, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                         left_title, right_title, suptitle, save_path, xlim=None, ylim_left=None):
    """
    Generic function to plot a 1x2 subplot:
      - Left: overlay of Q density curves (from Q_array) for the given indices.
      - Right: overlay of the corresponding IV curves with observed IV points.
      
    Parameters:
      indices      : list or array of indices to plot
      Q_array      : 2D numpy array of Q densities (rows: grid, columns: dates)
      date_strs    : list of date strings corresponding to columns in Q_array
      grid_full    : grid over which Q densities are defined
      ttm          : time-to-maturity value
      iv_surface_dir: directory containing IV surface CSV files
      df_obs       : DataFrame with observed IV data (with columns 'date', 'moneyness', 'tau', 'IV')
      tol          : tolerance for matching TTM values
      left_title   : title for the left subplot
      right_title  : title for the right subplot
      suptitle     : overall figure title
      save_path    : full path (including filename) to save the figure
      xlim         : tuple for x-axis limits (optional)
      ylim_left    : tuple for left y-axis limits (optional)
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    for i in indices:
        # Plot Q density on left axis
        axs[0].plot(grid_full, Q_array[:, i], alpha=0.5)
        
        # Get corresponding IV curve for the date
        date_str = date_strs[i]
        m_grid, estimated_iv = get_iv_curve_for_date_ttm(date_str, ttm, iv_surface_dir, tol)
        df_obs_date = df_obs[df_obs['date'] == pd.to_datetime(date_str)]
        obs_for_ttm = df_obs_date[np.abs(df_obs_date['tau'] - ttm) < tol]
        
        if m_grid is not None and estimated_iv is not None:
            axs[1].plot(m_grid, estimated_iv, label=f'Estimated IV (TTM={ttm})', color='blue')
        else:
            axs[1].text(0.5, 0.5, "No IV surface data", horizontalalignment='center',
                        verticalalignment='center', transform=axs[1].transAxes, color='red')
        if not obs_for_ttm.empty:
            axs[1].scatter(obs_for_ttm['moneyness'], obs_for_ttm['IV'], label='Observed IV', color='red')
    
    if xlim is not None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    if ylim_left is not None:
        axs[0].set_ylim(ylim_left)
    
    axs[0].set_title(left_title)
    axs[0].set_xlabel("Moneyness (K/S - 1)")
    axs[0].set_ylabel("Q Density")
    
    axs[1].set_title(right_title)
    axs[1].set_xlabel("Moneyness (K/S - 1)")
    axs[1].set_ylabel("IV")
    
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

def plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol, save_dir, group_label):
    """
    Plot the Q density and its corresponding IV curve for a single date.
    Parameters:
      i         : Column index of Q_array, corresponding to a specific date.
      group_label: Group label, such as "Nonnegative" or "Negative", used in the plot title.
      save_dir  : Directory to save the individual plot image.
    """
    date_str = date_strs[i]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # On the left, plot the Q density for this date.
    axs[0].plot(grid_full, Q_array[:, i])
    axs[0].set_title(f'Q Density for {date_str} ({group_label})\nTTM = {ttm} days')
    axs[0].set_xlabel("Moneyness (K/S - 1)")
    axs[0].set_ylabel("Q Density")
    
    # On the right, plot the IV curve and the observed data.
    m_grid, estimated_iv = get_iv_curve_for_date_ttm(date_str, ttm, iv_surface_dir, tol)
    df_obs_date = df_obs[df_obs['date'] == pd.to_datetime(date_str)]
    obs_for_ttm = df_obs_date[np.abs(df_obs_date['tau'] - ttm) < tol]
    if m_grid is not None and estimated_iv is not None:
        axs[1].plot(m_grid, estimated_iv, label=f'Estimated IV (TTM={ttm})', color='blue')
    else:
        axs[1].text(0.5, 0.5, "No IV surface data", horizontalalignment='center',
                    verticalalignment='center', transform=axs[1].transAxes, color='red')
    if not obs_for_ttm.empty:
        axs[1].scatter(obs_for_ttm['moneyness'], obs_for_ttm['IV'], label='Observed IV', color='red')
    axs[1].set_title(f'IV Curve for {date_str} ({group_label})\nTTM = {ttm} days')
    axs[1].set_xlabel("Moneyness (K/S - 1)")
    axs[1].set_ylabel("IV")
    axs[1].legend()
    
    plt.suptitle(f"Separate Q Density & IV Curve for {date_str} ({group_label})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'Q_IV_{date_str}_{ttm}day.png')
    plt.savefig(save_path)
    plt.close(fig)

# -------------------------------
# Main processing function
# -------------------------------

def process_ttm_combined(ttm, base_dir, Q_plots_dir, Q_data_dir, Q_matrix_dir,
                         iv_surface_dir, obs_data_path):
    """
    For a given TTM:
      - Process Q density CSV files in Q_data_dir/tau_{ttm}.
      - Interpolate, normalize, filter, and compute moments.
      - Generate combined plots (using the helper function) for various groups:
          * Raw densities
          * Nonnegative densities
          * Negative densities
          * Monotonic densities
          * Nonmonotonic densities
          * Final filtered densities based on moment conditions
      - Save final Q matrices.
    """
    print(f"Processing TTM = {ttm} days")
    
    # Define Q density directory.
    Q_data_ttm_dir = os.path.join(Q_data_dir, f"tau_{ttm}")
    
    # Create directories for combined Q-IV plots (all under Combined)
    combined_dir = os.path.join(Q_plots_dir, f"Combined_tau_{ttm}")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Intermediate directories for different groups:
    S0_dir = os.path.join(combined_dir, "S0_raw")
    S1_dir = os.path.join(combined_dir, "S1_nonnegative")
    S1_neg_dir = os.path.join(combined_dir, "S1_negative")
    S2_smooth_dir = os.path.join(combined_dir, "S2_smooth")
    S2_nonsmooth_dir = os.path.join(combined_dir, "S2_nonsmooth")
    S3_unimodal_dir = os.path.join(combined_dir, "S3_unimodal")
    S3_multimodal_dir = os.path.join(combined_dir, "S3_multimodal")
    for d in [S0_dir, S1_dir, S1_neg_dir, S2_smooth_dir, S2_nonsmooth_dir, S3_unimodal_dir, S3_multimodal_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Get date strings from Q density filenames.
    IV_files = sorted(os.listdir(Q_data_ttm_dir))
    Dates_list = [file.split("_")[2].split(".")[0] for file in IV_files]
    Dates_list = sorted(Dates_list)
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in Dates_list]
    date_strs = [d.strftime("%Y-%m-%d") for d in date_objs]
    
    # Define return grids.
    grid_full = np.arange(-1, 1.01, 0.01)
    grid_d15 = np.arange(-0.15, 0.16, 0.01)
    
    n_dates = len(Dates_list)
    Q_array = np.full((len(grid_full), n_dates), np.nan)
    Q_array_d15 = np.full((len(grid_d15), n_dates), np.nan)
    
    # Process each Q density file.
    for i, file in enumerate(IV_files):
        file_path = os.path.join(Q_data_ttm_dir, file)
        Q_data = pd.read_csv(file_path)
        if 'm' not in Q_data.columns or 'spdy' not in Q_data.columns:
            print(f"Columns 'm' and 'spdy' not found in {file}")
            continue
        if np.isclose(Q_data['m'].max(), 1) and np.isclose(Q_data['m'].min(), -1):
            try:
                interp_full = PchipInterpolator(Q_data['m'][1:-1], Q_data['spdy'][1:-1], extrapolate=True)
                Q_interp_full = interp_full(grid_full)
                interp_d15 = PchipInterpolator(Q_data['m'][1:-1], Q_data['spdy'][1:-1])
                Q_interp_d15 = interp_d15(grid_d15)
            except Exception as e:
                print(f"Interpolation error in {file}: {e}")
                continue
            norm_factor = np.trapezoid(Q_interp_full, grid_full)
            if norm_factor > 0:
                Q_interp_full /= norm_factor
                Q_interp_d15 /= norm_factor
            Q_array[:, i] = Q_interp_full
            Q_array_d15[:, i] = Q_interp_d15
        else:
            print(f"Return range mismatch in file: {file}")

    
    
    # -------------------------------
    # Load and preprocess observed IV data.
    df_obs = pd.read_csv(obs_data_path)
    df_obs['date'] = pd.to_datetime(df_obs['date'])
    df_obs = df_obs[(df_obs['moneyness'] >= -1) & (df_obs['moneyness'] <= 1)]
    df_obs = df_obs[(df_obs['tau'] >= 3) & (df_obs['tau'] <= 120)]
    df_obs['IV'] = df_obs['IV'] / 100  # adjust if needed
    tol = 1e-6  # TTM matching tolerance
    
    # -------------------------------
    # Plot raw densities combined with IV curves.
    raw_save_path = os.path.join(S0_dir, f'raw_all_density_{ttm}day.png')
    plot_combined_curves(
        indices=range(Q_array.shape[1]),
        Q_array=Q_array,
        date_strs=date_strs,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Raw Q Densities for TTM = {ttm} days',
        right_title=f"IV Curves (using last date label)",
        suptitle=f"Raw Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=raw_save_path
    )   
    Parallel(n_jobs=-2)(
        delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                                   S0_dir, group_label="Raw")
        for i in range(Q_array.shape[1])
    )
    
    # Identify groups based on density values.
    nonneg_idx = [i for i in range(Q_array.shape[1]) if np.nanmin(Q_array[:, i]) >= -0.1]
    neg_idx = [i for i in range(Q_array.shape[1]) if np.nanmin(Q_array[:, i]) < -0.1]
    
    # Plot nonnegative densities.
    nonneg_save_path = os.path.join(S1_dir, f'all_nonnegative_density_{ttm}day.png')
    plot_combined_curves(
        indices=nonneg_idx,
        Q_array=Q_array,
        date_strs=date_strs,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Nonnegative Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
        right_title=f"IV Curves",
        suptitle=f"Nonnegative Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=nonneg_save_path,
        xlim=(-1,1),
        ylim_left=(0, np.ceil(np.nanmax(Q_array[:, nonneg_idx])))
    )
    Parallel(n_jobs=-2)(
        delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                                   S1_dir, group_label="Nonnegative")
        for i in nonneg_idx
    )
    
    # Plot negative densities if any.
    if neg_idx:
        neg_save_path = os.path.join(S1_neg_dir, f'all_negative_density_{ttm}day.png')
        plot_combined_curves(
            indices=neg_idx,
            Q_array=Q_array,
            date_strs=date_strs,
            grid_full=grid_full,
            ttm=ttm,
            iv_surface_dir=iv_surface_dir,
            df_obs=df_obs,
            tol=tol,
            left_title=f'Negative Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
            right_title="IV Curves",
            suptitle=f"Negative Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
            save_path=neg_save_path,
            xlim=(-1,1),
            ylim_left=(np.floor(np.nanmin(Q_array[:, neg_idx])), np.ceil(np.nanmax(Q_array[:, neg_idx])))
        )
        Parallel(n_jobs=-2)(
            delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                                       S1_neg_dir, group_label="Negative")
            for i in neg_idx
        )
    else:
        print(f"No negative densities found for TTM = {ttm} days.")

    
    smoothness_list = []
    for i in range(Q_array.shape[1]):
        density = Q_array[:, i]
        # Compute second derivative (Laplacian)
        second_derivative = np.gradient(np.gradient(density))
        smoothness = np.var(second_derivative)  # High variance indicates instability
        smoothness_list.append(smoothness)

    # The distribution of smoothness values using boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(smoothness_list)
    plt.title(f'Smoothness Distribution for TTM = {ttm} days, by Tau-Independent SVI')
    plt.xlabel('Smoothness')
    plt.savefig(os.path.join(S2_smooth_dir, f'smoothness_boxplot_{ttm}day.png'))
    plt.close()

    # The threshold for smoothness is set as the upper IQE + 1.5 * IQE
    smoothness_list = np.array(smoothness_list)
    Q1 = np.percentile(smoothness_list, 25)
    Q3 = np.percentile(smoothness_list, 75)
    IQE = Q3 - Q1
    threshold = Q3 + 1.5 * IQE
    print(f"Threshold for smoothness: {threshold}")
    print(f"Number of dates with smoothness above threshold: {np.sum(smoothness_list > threshold)}")
    threshold = 0.006
    print(f"Threshold for smoothness: {threshold}")
    print(f"Number of dates with smoothness above threshold: {np.sum(smoothness_list > threshold)}")
    
    # Enforce smoothness
    smooth_idx = [i for i in nonneg_idx if smoothness_list[i] <= threshold]
    nonsmooth_idx = [i for i in nonneg_idx if i not in smooth_idx]

    print(f"The average smoothness value for the smooth group is {np.mean(smoothness_list[smooth_idx]):.3f}")
    print(f"The average smoothness value for the nonsmooth group is {np.mean(smoothness_list[nonsmooth_idx]):.3f}")

    S2_smooth_dir = os.path.join(combined_dir, "S2_smooth")
    S2_nonsmooth_dir = os.path.join(combined_dir, "S2_nonsmooth")
    for d in [S2_smooth_dir, S2_nonsmooth_dir]:
        os.makedirs(d, exist_ok=True)

    # Plot smooth densities.
    smooth_save_path = os.path.join(S2_smooth_dir, f'all_smooth_density_{ttm}day.png')
    plot_combined_curves(
        indices=smooth_idx,
        Q_array=Q_array,
        date_strs=date_strs,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Smooth Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
        right_title="IV Curves",
        suptitle=f"Smooth Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=smooth_save_path,
        xlim=(-1,1),
        ylim_left=(0, np.ceil(np.nanmax(Q_array[:, smooth_idx])))
    )
    Parallel(n_jobs=-2)(delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                            S2_smooth_dir, group_label=f"Smoothness value {smoothness_list[i]:.4f}") for i in smooth_idx)

    # Plot nonsmooth densities.
    if len(nonsmooth_idx) > 0:
        nonsmooth_save_path = os.path.join(S2_nonsmooth_dir, f'all_nonsmooth_density_{ttm}day.png')
        plot_combined_curves(
            indices=nonsmooth_idx,
            Q_array=Q_array,
            date_strs=date_strs,
            grid_full=grid_full,
            ttm=ttm,
            iv_surface_dir=iv_surface_dir,
            df_obs=df_obs,
            tol=tol,
            left_title=f'Nonsmooth Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
            right_title="IV Curves",
            suptitle=f"Nonsmooth Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
            save_path=nonsmooth_save_path,
            xlim=(-1,1),
            ylim_left=(0, np.ceil(np.nanmax(Q_array[:, nonsmooth_idx])))
        )
        Parallel(n_jobs=-2)(delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                                S2_nonsmooth_dir, group_label=f"Smoothness value {smoothness_list[i]:.4f}") for i in nonsmooth_idx)
    
    peak_list = []
    for i in range(Q_array.shape[1]):
        density = Q_array[:, i]
        # Count peaks
        peaks, _ = signal.find_peaks(density)
        peak_count = len(peaks)
        peak_list.append(peak_count)

    # The distribution of peaks
    plt.figure(figsize=(10, 6))
    plt.hist(peak_list, bins=range(0, 5), edgecolor='black')
    plt.title(f'Peak Distribution for TTM = {ttm} days, by Tau-Independent SVI')
    plt.xlabel('Number of Peaks')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(S3_unimodal_dir, f'peak_distribution_{ttm}day.png'))
    plt.close()

    # The threshold for peaks
    peak_list = np.array(peak_list)
    threshold_peak = 2
    print(f"Threshold for peaks: {threshold_peak}")
    print(f"Number of dates with peaks above threshold: {np.sum(peak_list >= threshold_peak)}")

    
    # Enforce number of peaks
    unimodal_idx = [i for i in smooth_idx if peak_list[i] < threshold_peak]
    multimodal_idx = [i for i in smooth_idx if i not in unimodal_idx]

    S3_unimodal_dir = os.path.join(combined_dir, "S3_unimodal")
    S3_multimodal_dir = os.path.join(combined_dir, "S3_multimodal")
    for d in [S3_unimodal_dir, S3_multimodal_dir]:
        os.makedirs(d, exist_ok=True)

    # Plot unimodal densities.
    unimodal_save_path = os.path.join(S3_unimodal_dir, f'all_unimodal_density_{ttm}day.png')
    plot_combined_curves(
        indices=unimodal_idx,
        Q_array=Q_array,
        date_strs=date_strs,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Unimodal Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
        right_title="IV Curves",
        suptitle=f"Unimodal Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=unimodal_save_path,
        xlim=(-1,1),
        ylim_left=(0, np.ceil(np.nanmax(Q_array[:, unimodal_idx])))
    )
    Parallel(n_jobs=-2)(delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                            S3_unimodal_dir, group_label=f"Unimodal") for i in unimodal_idx)

    # Plot multimodal densities.
    if len(multimodal_idx) > 0:
        multimodal_save_path = os.path.join(S3_multimodal_dir, f'all_multimodal_density_{ttm}day.png')
        plot_combined_curves(
            indices=multimodal_idx,
            Q_array=Q_array,
            date_strs=date_strs,
            grid_full=grid_full,
            ttm=ttm,
            iv_surface_dir=iv_surface_dir,
            df_obs=df_obs,
            tol=tol,
            left_title=f'Multimodal Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
            right_title="IV Curves",
            suptitle=f"Multimodal Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
            save_path=multimodal_save_path,
            xlim=(-1,1),
            ylim_left=(0, np.ceil(np.nanmax(Q_array[:, multimodal_idx])))
        )
        Parallel(n_jobs=-2)(delayed(plot_single_curve)(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                                S3_multimodal_dir, group_label=f"Unimodal") for i in multimodal_idx)
        
    # Final Q matrices
    Q_array_final = Q_array[:, unimodal_idx]
    dates_final = [date_strs[i] for i in unimodal_idx]

    # Save final Q matrices.
    Q_full_df = pd.DataFrame(Q_array_final, index=grid_full, columns=dates_final)
    Q_full_df.index.name = "Return"
    Q_matrix_save_path = os.path.join(Q_matrix_dir, f"Q_matrix_{ttm}day.csv")
    Q_full_df.to_csv(Q_matrix_save_path)

    Q_d15_final = Q_array_d15[:, unimodal_idx]
    Q_d15_df = pd.DataFrame(Q_d15_final, index=grid_d15, columns=dates_final)
    Q_d15_df.index.name = "Return"
    Q_d15_save_path = os.path.join(Q_matrix_dir, f"Q_matrix_{ttm}day_d15.csv")
    Q_d15_df.to_csv(Q_d15_save_path)
        
    print(f"Finished processing combined plots for TTM = {ttm} days.")

# -------------------------------
# Main execution block (parallel)
# -------------------------------


base_dir = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
os.chdir(base_dir)
Q_plots_dir = os.path.join(base_dir, "Q_plots", "Tau-independent", "unique", "moneyness_step_0d01")
Q_data_dir = os.path.join(base_dir, 'Q_from_pure_SVI', 'Tau-independent', 'unique', 'moneyness_step_0d01')
Q_matrix_dir = os.path.join(base_dir, "Q_matrix", "Tau-independent", "unique", "moneyness_step_0d01")
iv_surface_dir = os.path.join(base_dir, "IV", "IV_surface_SVI", "Tau-independent", "unique", "moneyness_step_0d01")
obs_data_path = os.path.join(base_dir, "Data", "processed", "20172022_processed_1_3_5_standardized_moneyness.csv")
os.makedirs(Q_plots_dir, exist_ok=True)
os.makedirs(Q_matrix_dir, exist_ok=True)

ttm_values = [45]
for ttm in ttm_values:
    process_ttm_combined(ttm, base_dir, Q_plots_dir, Q_data_dir, Q_matrix_dir,
                                        iv_surface_dir, obs_data_path)