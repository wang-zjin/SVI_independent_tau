# Plot Q density and IV curve in the same figure

import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed

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
    combined_dir = os.path.join(Q_plots_dir, "Combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Intermediate directories for different groups:
    S0_dir = os.path.join(combined_dir, "S0_raw")
    S1_dir = os.path.join(combined_dir, "S1_nonnegative")
    S1_neg_dir = os.path.join(combined_dir, "S1_negative")
    S2_dir = os.path.join(combined_dir, "S2_monotonic")
    S2_nonmono_dir = os.path.join(combined_dir, "S2_nonmonotonic")
    S3_dir = os.path.join(combined_dir, "S3_moment")
    S3_moment_fail_dir = os.path.join(combined_dir, "S3_moment_fail")
    for d in [S0_dir, S1_dir, S1_neg_dir, S2_dir, S2_nonmono_dir, S3_dir, S3_moment_fail_dir]:
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
                interp_full = PchipInterpolator(Q_data['m'], Q_data['spdy'])
                Q_interp_full = interp_full(grid_full)
                interp_d15 = PchipInterpolator(Q_data['m'], Q_data['spdy'])
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
    for i in range(Q_array.shape[1]):
        plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                          S0_dir, group_label="Raw")
    
    # Identify groups based on density values.
    nonneg_idx = [i for i in range(Q_array.shape[1]) if np.nanmin(Q_array[:, i]) >= 0]
    neg_idx = [i for i in range(Q_array.shape[1]) if np.nanmin(Q_array[:, i]) < 0]
    
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
    for i in nonneg_idx:
        plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                          S1_dir, group_label="Nonnegative")
    
    
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
        for i in neg_idx:
            plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                            S1_neg_dir, group_label="Negative")
    else:
        print(f"No negative densities found for TTM = {ttm} days.")

    
    # Enforce monotonicity.
    mono_idx = []
    nonmono_idx = []
    for i in nonneg_idx:
        dens = Q_array[:, i]
        mode_index = np.argmax(dens)
        left = dens[:mode_index+1]
        right = dens[mode_index:]
        if np.all(np.diff(left) >= 0) and np.all(np.diff(right) <= 0):
            mono_idx.append(i)
        else:
            nonmono_idx.append(i)
    mono_idx = np.array(mono_idx)
    nonmono_idx = np.array(nonmono_idx)
    
    # Plot monotonic densities.
    mono_save_path = os.path.join(S2_dir, f'all_monotonic_density_{ttm}day.png')
    plot_combined_curves(
        indices=mono_idx,
        Q_array=Q_array,
        date_strs=date_strs,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Monotonic Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
        right_title="IV Curves",
        suptitle=f"Monotonic Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=mono_save_path,
        xlim=(-1,1),
        ylim_left=(0, np.ceil(np.nanmax(Q_array[:, mono_idx])))
    )
    for i in mono_idx:
        plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                        S2_dir, group_label="Monotonic")
    
    # Plot nonmonotonic densities.
    if nonmono_idx.size > 0:
        nonmono_save_path = os.path.join(S2_nonmono_dir, f'all_nonmonotonic_density_{ttm}day.png')
        plot_combined_curves(
            indices=nonmono_idx,
            Q_array=Q_array,
            date_strs=date_strs,
            grid_full=grid_full,
            ttm=ttm,
            iv_surface_dir=iv_surface_dir,
            df_obs=df_obs,
            tol=tol,
            left_title=f'Nonmonotonic Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
            right_title="IV Curves",
            suptitle=f"Nonmonotonic Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
            save_path=nonmono_save_path,
            xlim=(-1,1),
            ylim_left=(0, np.ceil(np.nanmax(Q_array[:, nonmono_idx])))
        )
        for i in nonmono_idx:
            plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                            S2_nonmono_dir, group_label="Nonmonotonic")
    
    # Compute moments for monotonic densities.
    moments = [compute_density_moments(grid_full, Q_array[:, i], ttm) for i in mono_idx]
    moments = np.array(moments)
    valid_dates = [date_strs[i] for i in mono_idx]
    moments_df = pd.DataFrame(moments, columns=['Mean', 'Variance', 'Skewness', 'Kurtosis'],
                              index=pd.to_datetime(valid_dates))
    moments_save_path = os.path.join(S2_dir, f'moments_timetable_{ttm}day.csv')
    moments_df.to_csv(moments_save_path)
    
    plt.figure(figsize=(8,6))
    moments_df.boxplot()
    plt.title(f'Box Plot of Moments for TTM = {ttm} days, by Tau-Independent SVI')
    boxplot_save_path = os.path.join(S2_dir, f'moments_boxplot_{ttm}day.png')
    plt.savefig(boxplot_save_path)
    plt.close()
    
    # Further filter based on moment thresholds.
    mean_range = (-10, 10)
    variance_range = (0, 10)
    skewness_threshold = 10
    kurtosis_range = (-5, 30)
    valid_mask = (
        (moments[:, 0] >= mean_range[0]) & (moments[:, 0] <= mean_range[1]) &
        (moments[:, 1] >= variance_range[0]) & (moments[:, 1] <= variance_range[1]) &
        (np.abs(moments[:, 2]) <= skewness_threshold) &
        (moments[:, 3] >= kurtosis_range[0]) & (moments[:, 3] <= kurtosis_range[1])
    )
    final_idx = mono_idx[valid_mask]
    Q_array_final = Q_array[:, final_idx]
    dates_final = [date_strs[i] for i in final_idx]

    # Plot final filtered densities.
    final_save_path = os.path.join(S3_dir, f'all_final_density_{ttm}day.png')
    plot_combined_curves(
        indices=range(Q_array_final.shape[1]),  # indices for Q_array_final correspond to final order
        Q_array=Q_array_final,
        date_strs=dates_final,
        grid_full=grid_full,
        ttm=ttm,
        iv_surface_dir=iv_surface_dir,
        df_obs=df_obs,
        tol=tol,
        left_title=f'Final Filtered {Q_array_final.shape[1]} Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
        right_title="IV Curves",
        suptitle=f"Final Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
        save_path=final_save_path,
        xlim=(-1,1),
        ylim_left=(0, np.ceil(np.nanmax(Q_array_final)))
    )
    for i in range(Q_array_final.shape[1]):
        plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                        S3_dir, group_label="Final")
        

    # Get the indices from the monotonic set that do not satisfy the moment conditions.
    not_final_idx = np.setdiff1d(mono_idx, final_idx)

    # Plot momoent failed densities.
    if not_final_idx.size > 0:
        final_fail_save_path = os.path.join(S3_moment_fail_dir, f'all_final_fail_density_{ttm}day.png')
        plot_combined_curves(
            indices=not_final_idx,  # indices for Q_array_final correspond to final order
            Q_array=Q_array,
            date_strs=date_strs,
            grid_full=grid_full,
            ttm=ttm,
            iv_surface_dir=iv_surface_dir,
            df_obs=df_obs,
            tol=tol,
            left_title=f'Moment fail {Q_array_final.shape[1]} Q Densities for TTM = {ttm} days, by Tau-Independent SVI',
            right_title="IV Curves",
            suptitle=f"Moment fail Q Densities, IV Curves and IV observations (TTM = {ttm} days)",
            save_path=final_fail_save_path,
            xlim=(-1,1),
            ylim_left=(0, np.ceil(np.nanmax(Q_array_final)))
        )
        for i in not_final_idx:
            plot_single_curve(i, Q_array, date_strs, grid_full, ttm, iv_surface_dir, df_obs, tol,
                            S3_moment_fail_dir, group_label="Moment Fail")
        
    
    # Save final Q matrices.
    Q_full_df = pd.DataFrame(Q_array_final, index=grid_full, columns=dates_final)
    Q_full_df.index.name = "Return"
    Q_matrix_save_path = os.path.join(Q_matrix_dir, f"Q_matrix_{ttm}day.csv")
    Q_full_df.to_csv(Q_matrix_save_path)
    
    Q_d15_final = Q_array_d15[:, final_idx]
    Q_d15_df = pd.DataFrame(Q_d15_final, index=grid_d15, columns=dates_final)
    Q_d15_df.index.name = "Return"
    Q_d15_save_path = os.path.join(Q_matrix_dir, f"Q_matrix_{ttm}day_d15.csv")
    Q_d15_df.to_csv(Q_d15_save_path)
    
    print(f"Finished processing combined plots for TTM = {ttm} days.")
    return ttm

# -------------------------------
# Main execution block (parallel)
# -------------------------------

def main_parallel():
    base_dir = "/Users/irtg/Documents/Github/BTC-premia/SVI_independent_tau/"
    os.chdir(base_dir)
    Q_plots_dir = os.path.join(base_dir, "Q_plots", "Tau-independent", "unique", "moneyness_step_0d01")
    Q_data_dir = os.path.join(base_dir, 'Q_from_pure_SVI', 'Tau-independent', 'unique', 'moneyness_step_0d01')
    Q_matrix_dir = os.path.join(base_dir, "Q_matrix", "Tau-independent", "unique", "moneyness_step_0d01")
    iv_surface_dir = os.path.join(base_dir, "IV", "IV_surface_SVI", "Tau-independent", "unique", "moneyness_step_0d01")
    obs_data_path = os.path.join(base_dir, "Data", "processed", "20172022_processed_1_3_5_standardized_moneyness.csv")
    os.makedirs(Q_plots_dir, exist_ok=True)
    os.makedirs(Q_matrix_dir, exist_ok=True)
    
    ttm_values = range(3, 121)
    results = Parallel(n_jobs=-2)(
        delayed(process_ttm_combined)(ttm, base_dir, Q_plots_dir, Q_data_dir, Q_matrix_dir,
                                      iv_surface_dir, obs_data_path) for ttm in ttm_values
    )
    print("Processed TTMs:", results)

# Run the pipeline in parallel.
if __name__ == "__main__":
    main_parallel()