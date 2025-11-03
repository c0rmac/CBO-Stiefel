# filename: visualisation.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List, Dict, Any


def save_cbo_data_to_pickle(save_dir: str,
                            time_points: np.ndarray,
                            all_f_M_dynamics: List[np.ndarray],
                            asymptotic_values: np.ndarray,
                            objective_instance: Any, # Assuming objective_instance is a class instance like AckleyObjective
                            solver_name: str,
                            k: int, n: int) -> str:
    """
    Saves all CBO experiment results data to a pickle file.

    Args:
        time_points: Array of time values (t).
        all_f_M_dynamics: List of f_history arrays for each trial.
        asymptotic_values: Array of final f(M_T) values from all trials.
        objective_instance: The instantiated objective class (for metadata).
        solver_name: Name of the solver used.
        k: Dimension k of the Stiefel manifold V(k,n).
        n: Dimension n of the Stiefel manifold V(k,n).

    Returns:
        str: The filename of the saved pickle file, or an empty string on failure.
    """
    if time_points is None or not all_f_M_dynamics or asymptotic_values is None:
        print("Pickle saving skipped: Data is missing or corrupted.")
        return ""

    # Define a base filename for all output files
    base_filename = f"cbo_results_{solver_name}_V{k}_{n}"
    pickle_filename = f"saves/{save_dir}/{base_filename}.pkl"

    data_to_save: Dict[str, Any] = {
        'time_points': time_points,
        'all_f_M_dynamics': all_f_M_dynamics,
        'asymptotic_values': asymptotic_values,
        'solver_name': solver_name,
        'k': k,
        'n': n
    }

    try:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"✅ Successfully saved results to: {pickle_filename}")
        return pickle_filename
    except Exception as e:
        print(f"❌ Error saving results to pickle file: {e}")
        return ""


def plot_cbo_results(save_dir: str,
                    time_points: np.ndarray,
                     all_f_M_dynamics: List[np.ndarray],
                     asymptotic_values: np.ndarray,
                     objective_instance: Any,
                     solver_name: str,
                     k: int, n: int, use_true_min=False) -> None:
    """
    Generates three separate plots and saves each as a .png file:
    1. The time dynamics (mean and min/max range) of the objective function f(M_t).
    2. The full histogram (probability density) of the final asymptotic f(M_T) values.
    3. A "zoomed-in" histogram focusing on the neighbourhood of the true minimum.

    Args:
        time_points: Array of time values (t).
        all_f_M_dynamics: List of f_history arrays for each trial.
        asymptotic_values: Array of final f(M_T) values from all trials.
        objective_instance: The instantiated objective class.
        solver_name: Name of the solver used.
        k: Dimension k of the Stiefel manifold V(k,n).
        n: Dimension n of the Stiefel manifold V(k,n).
    """

    if time_points is None or not all_f_M_dynamics or asymptotic_values is None:
        print("Plotting skipped: Data is missing or corrupted.")
        return

        # --- Common Calculations ---
    dynamics_array = np.array(all_f_M_dynamics)
    try:
        if use_true_min:
            true_min = objective_instance.get_true_minimum()
        else:
            true_min = None
    except AttributeError:
        true_min = None
    final_time = time_points[-1]
    base_filename = f"{save_dir}/cbo_results_{solver_name}_V{k}_{n}"

    # --- 1. Plot 1: The dynamics of f(M_t) ---
    if dynamics_array.size > 0:
        mean_dynamics = np.mean(dynamics_array, axis=0)

        fig1, ax1 = plt.subplots(figsize=(8, 6))

        ax1.plot(time_points, mean_dynamics, color='mediumblue', linewidth=2, label='Mean $f(M_t)$')
        ax1.fill_between(
            time_points,
            np.min(dynamics_array, axis=0),
            np.max(dynamics_array, axis=0),
            color='blue',
            alpha=0.2,
            label='Min/Max Range'
        )
        if true_min:
            ax1.axhline(
                y=true_min,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'True Min ({true_min:.4g})'
            )
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('$f(M_t)$')
        ax1.set_title(f'CBO Dynamics: {solver_name} on V({k},{n})')
        ax1.set_xlim(0, final_time)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        plt.tight_layout()
        plot_filename_1 = f"saves/{base_filename}_dynamics.png"
        try:
            fig1.savefig(plot_filename_1)
            print(f"Saved dynamics plot to: {plot_filename_1}")
        except Exception as e:
            print(f"Error saving dynamics plot: {e}")
        plt.close(fig1)
    else:
        print("Plotting skipped for dynamics: Dynamics data is empty.")

    # --- 2. Generate Histogram Plots (if data exists) ---
    if asymptotic_values.size == 0:
        print("Skipping histogram plots: No valid asymptotic data exists.")
        return

    # --- 2a. Plot 2: Full Distribution of f(M_T) ---

    # Define scale factor (Delta) for visualization buffer based on *all* data
    upper_percentile_full = np.percentile(asymptotic_values, 99.5)
    lower_percentile_full = np.percentile(asymptotic_values, 0.5)
    observed_range_full = upper_percentile_full - lower_percentile_full
    scale_factor_delta_full = np.maximum(1e-6, observed_range_full * 0.05)

    if true_min:
        min_bin_full = true_min - scale_factor_delta_full
    else:
        min_bin_full = lower_percentile_full

    max_bin_target_full = upper_percentile_full
    max_bin_full = np.maximum(min_bin_full + 2 * scale_factor_delta_full, max_bin_target_full)
    full_bins = np.linspace(min_bin_full, max_bin_full, 30)

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.hist(
        asymptotic_values,
        bins=full_bins,
        color='darkorange',
        edgecolor='k',
        density=True
    )
    if true_min:
        ax2.axvline(
            x=true_min,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'True Min ({true_min:.4g})'
        )
    ax2.legend()
    ax2.set_xlabel(f'Asymptotic value ($f(M_{{{int(final_time)}}})$)')
    ax2.set_ylabel('Probability density')
    ax2.set_title(f'Full Distribution of Final Values: {solver_name} on V({k},{n})')
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_filename_2 = f"saves/{base_filename}_hist_full.png"
    try:
        fig2.savefig(plot_filename_2)
        print(f"Saved full histogram to: {plot_filename_2}")
    except Exception as e:
        print(f"Error saving full histogram: {e}")
    plt.close(fig2)

    # --- 2b. Plot 3: Zoomed Distribution (Nearest Successful Cluster) ---
    if true_min:
        # 1. Sort the data to find gaps
        sorted_values = np.sort(asymptotic_values)

        # 2. Find the differences (gaps) between consecutive sorted points
        diffs = np.diff(sorted_values)

        if diffs.size > 0:
            # 3. Find the index of the largest gap
            i_gap = np.argmax(diffs)

            # 4. The cutoff value is the point *before* the largest gap
            cutoff_value = sorted_values[i_gap]

            # 5. Define the "zoomed data" as all points <= this cutoff
            zoomed_data = asymptotic_values[asymptotic_values <= cutoff_value]
        else:
            # Fallback if all values are identical (no diffs)
            zoomed_data = asymptotic_values

        if zoomed_data.size == 0:
            print("Skipping zoomed histogram: No data in the nearest cluster (this is unusual).")
            return

        # 6. Define bins based on the properties of this "zoomed" subset
        upper_percentile_zoom = np.percentile(zoomed_data, 99.5)
        lower_percentile_zoom = np.percentile(zoomed_data, 0.5)
        observed_range_zoom = upper_percentile_zoom - lower_percentile_zoom

        # Use a smaller floor (1e-7) for potentially tiny ranges
        scale_factor_delta_zoom = np.maximum(1e-7, observed_range_zoom * 0.05)

        # 7. Set bin limits
        min_bin_zoom = true_min - scale_factor_delta_zoom
        max_bin_target_zoom = upper_percentile_zoom
        # Ensure max_bin is always > min_bin and captures the 99.5th percentile
        max_bin_zoom = np.maximum(min_bin_zoom + 2 * scale_factor_delta_zoom, max_bin_target_zoom)

        zoom_bins = np.linspace(min_bin_zoom, max_bin_zoom, 30)

        fig3, ax3 = plt.subplots(figsize=(8, 6))

        # Plot the histogram *only* for the zoomed data
        ax3.hist(
            zoomed_data,
            bins=zoom_bins,
            color='forestgreen',
            edgecolor='k',
            density=True
        )
        ax3.axvline(
            x=true_min,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'True Min ({true_min:.4g})'
        )
        ax3.legend()
        ax3.set_xlabel(f'Asymptotic value ($f(M_{{{int(final_time)}}})$)')
        ax3.set_ylabel('Probability density')
        ax3.set_title(f'Zoomed Distribution (Nearest Cluster): {solver_name} on V({k},{n})')
        # Set x-limits to the bin range to enforce the zoom
        ax3.set_xlim(min_bin_zoom, max_bin_zoom)
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax3.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_filename_3 = f"saves/{base_filename}_hist_zoom.png"
        try:
            fig3.savefig(plot_filename_3)
            print(f"Saved zoomed histogram to: {plot_filename_3}")
        except Exception as e:
            print(f"Error saving zoomed histogram: {e}")
        plt.close(fig3)