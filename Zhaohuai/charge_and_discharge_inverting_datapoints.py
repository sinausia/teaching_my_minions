'''


inverting values
'''

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress RuntimeWarning for better output readability
warnings.filterwarnings("ignore", category=RuntimeWarning)

def exp_func(x, a, b, c):
    """Single exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x)

def perform_fitting_attempt(x_clean, y_clean):
    """
    Helper function to perform the actual curve fitting with multiple initial guesses
    Returns the best fit result or None if all attempts fail
    """
    # Better initial parameter estimation
    y_start = y_clean[0]
    y_end = y_clean[-1]
    x_range = x_clean[-1] - x_clean[0]
    
    # Estimate if it's growth or decay
    if abs(y_end) > abs(y_start):
        # Likely growth
        b_guess = 0.1
    else:
        # Likely decay
        b_guess = -0.1
    
    # Estimate offset as minimum value
    c_guess = np.min(y_clean)
    
    # Estimate amplitude
    a_guess = y_start - c_guess
    if abs(a_guess) < 1e-10:
        a_guess = np.max(y_clean) - c_guess
    
    # Try multiple initial guesses - be more aggressive
    initial_guesses = [
        [a_guess, b_guess, c_guess],
        [y_start, -0.1, 0.0],  # Original guess
        [y_start, 0.1, 0.0],   # Growth instead of decay
        [y_start, -0.001, 0.0],  # Very slow decay
        [y_start, 0.001, 0.0],   # Very slow growth
        [np.max(y_clean), -0.05, np.min(y_clean)],  # Conservative decay
        [y_start - y_end, -0.2, y_end],  # Strong decay to end value
        [1.0, -0.001, np.mean(y_clean)],  # Tiny exponential around mean
        [np.std(y_clean), -0.0001, np.mean(y_clean)],  # Ultra-conservative
    ]
    
    best_result = None
    best_r_squared = -np.inf
    
    for p0 in initial_guesses:
        try:
            # More relaxed bounds to allow tiny b values
            bounds = (
                [-np.inf, -50, -np.inf],  # Lower bounds (allow more extreme decay)
                [np.inf, 50, np.inf]      # Upper bounds (allow more extreme growth)
            )
            
            popt, pcov = curve_fit(
                exp_func, x_clean, y_clean, 
                p0=p0, 
                bounds=bounds,
                maxfev=5000  # Increase max iterations
            )
            
            # Calculate fit quality
            y_fit = exp_func(x_clean, *popt)
            residuals = y_clean - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
            
            if ss_tot == 0:
                r_squared = 1.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Keep the best fit
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_result = (popt, pcov, r_squared)
            
        except Exception as e:
            continue  # Try next initial guess
    
    return best_result

def fit_single_pulse(x_pulse, y_pulse, pulse_number, pulse_type, experiment_id, fit_type="full"):
    """
    Improved fit function with y-inversion when parameter 'a' is negative
    """
    try:
        # Data validation
        if len(x_pulse) < 3:
            raise ValueError("Insufficient data points for fitting")
        
        if len(x_pulse) != len(y_pulse):
            raise ValueError("x and y data have different lengths")
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(x_pulse) & np.isfinite(y_pulse)
        if not np.any(valid_mask):
            raise ValueError("No valid data points")
        
        x_clean = x_pulse[valid_mask]
        y_clean = y_pulse[valid_mask]
        
        if len(x_clean) < 3:
            raise ValueError("Insufficient valid data points after cleaning")
        
        # Handle constant data by forcing a tiny slope
        if np.std(y_clean) < 1e-10:
            # For constant data, create a tiny artificial trend
            y_clean = y_clean + np.linspace(0, 1e-12, len(y_clean))
        
        # Store original y data for potential inversion
        y_original = y_clean.copy()
        data_was_inverted = False
        
        # STEP 1: Initial fit to check if 'a' is negative
        best_result_initial = perform_fitting_attempt(x_clean, y_clean)
        
        # STEP 2: Check if 'a' is negative and invert if necessary
        if best_result_initial is not None:
            a_initial = best_result_initial[0][0]  # Extract 'a' parameter
            
            if a_initial < 0:
                print(f"  Parameter 'a' is negative ({a_initial:.6f}), inverting y values...")
                # Invert y values: y_new = -y_old
                y_clean = -y_clean
                data_was_inverted = True
                
                # STEP 3: Refit with inverted data
                best_result = perform_fitting_attempt(x_clean, y_clean)
                
                # If refitting failed, use original result
                if best_result is None:
                    print("  Refitting with inverted data failed, using original fit")
                    y_clean = y_original
                    data_was_inverted = False
                    best_result = best_result_initial
                else:
                    print(f"  Refitting successful with inverted data")
            else:
                # 'a' is positive, use original fit
                best_result = best_result_initial
        else:
            # Initial fitting failed completely
            best_result = None
        
        # If all attempts failed, force a fit with linear approximation
        if best_result is None:
            try:
                # Last resort: fit a nearly-flat exponential
                mean_y = np.mean(y_clean)
                # Force fit with b very close to zero
                popt = [0.0, 1e-10, mean_y]  # Essentially y = c (constant)
                
                # Create fake covariance matrix
                pcov = np.eye(3) * 1e-6
                
                # Calculate "R-squared" for constant fit
                y_fit = exp_func(x_clean, *popt)
                residuals = y_clean - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
                
                if ss_tot == 0:
                    r_squared = 1.0  # Perfect fit for constant data
                else:
                    r_squared = 1 - (ss_res / ss_tot)
                
                best_result = (popt, pcov, r_squared)
                
            except:
                # Absolute last resort
                popt = [0.0, 0.0, np.mean(y_clean) if len(y_clean) > 0 else 0.0]
                pcov = None
                r_squared = 0.0
                best_result = (popt, pcov, r_squared)
        
        popt, pcov, r_squared = best_result
        a, b, c = popt
        
        # Calculate parameter uncertainties
        if pcov is not None:
            param_errors = np.sqrt(np.diag(pcov))
        else:
            param_errors = [np.nan, np.nan, np.nan]
        
        # Less strict validation - allow any parameter values
        fit_warnings = []
        
        # Add warning about data inversion
        if data_was_inverted:
            fit_warnings.append("Y data was inverted due to negative 'a' parameter")
        
        # Only warn for extreme values, don't fail
        if abs(b) > 10:
            fit_warnings.append("Very extreme decay/growth rate")
        elif abs(b) < 1e-6:
            fit_warnings.append("Nearly constant (very small b parameter)")
        
        if r_squared < 0.1:
            fit_warnings.append("Very poor fit quality")
        elif r_squared < 0.5:
            fit_warnings.append("Poor fit quality")
        
        # Check for numerical issues but don't fail the fit
        try:
            y_fit_full = exp_func(x_clean, a, b, c)
            if not np.all(np.isfinite(y_fit_full)):
                fit_warnings.append("Numerical overflow in fitted curve")
        except:
            fit_warnings.append("Could not evaluate fitted curve")
        
        return {
            'Experiment_ID': experiment_id,
            'Pulse_Number': pulse_number,
            'Pulse_Type': pulse_type,
            'Fit_Type': fit_type,
            'Parameter_a': a,
            'Parameter_b': b,
            'Parameter_c': c,
            'Parameter_a_Error': param_errors[0],
            'Parameter_b_Error': param_errors[1], 
            'Parameter_c_Error': param_errors[2],
            'R_Squared': r_squared,
            'Equation': f"{a:.6f} * exp({b:.6f} * x) + {c:.6f}",
            'Data_Points': len(y_clean),
            'Original_Data_Points': len(y_pulse),
            'Start_Index': x_pulse[0] if len(x_pulse) > 0 else np.nan,
            'End_Index': x_pulse[-1] if len(x_pulse) > 0 else np.nan,
            'Fit_Success': True,
            'Fit_Warnings': '; '.join(fit_warnings) if fit_warnings else 'None',
            'Data_Inverted': data_was_inverted,  # New field to track inversion
            # Store data for plotting (use final fitted data)
            'x_data': x_clean,
            'y_data': y_clean,  # This will be inverted if inversion occurred
            'y_data_original': y_original,  # Always store original data
            'fit_params': popt
        }
        
    except Exception as e:
        # Even in the exception handler, try to force a basic fit
        try:
            # Ultimate fallback: constant fit
            if 'y_pulse' in locals() and len(y_pulse) > 0:
                mean_val = np.mean(y_pulse[np.isfinite(y_pulse)])
                return {
                    'Experiment_ID': experiment_id,
                    'Pulse_Number': pulse_number,
                    'Pulse_Type': pulse_type,
                    'Fit_Type': fit_type,
                    'Parameter_a': 0.0,
                    'Parameter_b': 0.0,
                    'Parameter_c': mean_val,
                    'Parameter_a_Error': np.nan,
                    'Parameter_b_Error': np.nan,
                    'Parameter_c_Error': np.nan,
                    'R_Squared': 1.0 if np.std(y_pulse[np.isfinite(y_pulse)]) < 1e-10 else 0.0,
                    'Equation': f"0.0 * exp(0.0 * x) + {mean_val:.6f}",
                    'Data_Points': len(y_pulse),
                    'Original_Data_Points': len(y_pulse),
                    'Start_Index': x_pulse[0] if 'x_pulse' in locals() and len(x_pulse) > 0 else np.nan,
                    'End_Index': x_pulse[-1] if 'x_pulse' in locals() and len(x_pulse) > 0 else np.nan,
                    'Fit_Success': True,
                    'Fit_Warnings': f'Forced constant fit due to error: {str(e)}',
                    'Data_Inverted': False,
                    'x_data': x_pulse if 'x_pulse' in locals() else np.array([]),
                    'y_data': y_pulse if 'y_pulse' in locals() else np.array([]),
                    'y_data_original': y_pulse if 'y_pulse' in locals() else np.array([]),
                    'fit_params': [0.0, 0.0, mean_val]
                }
        except:
            pass
        
        # Absolute last resort if everything fails
        return {
            'Experiment_ID': experiment_id,
            'Pulse_Number': pulse_number,
            'Pulse_Type': pulse_type,
            'Fit_Type': fit_type,
            'Parameter_a': 0.0,
            'Parameter_b': 0.0,
            'Parameter_c': 0.0,
            'Parameter_a_Error': np.nan,
            'Parameter_b_Error': np.nan,
            'Parameter_c_Error': np.nan,
            'R_Squared': 0.0,
            'Equation': '0.0 * exp(0.0 * x) + 0.0',
            'Data_Points': len(y_pulse) if 'y_pulse' in locals() else 0,
            'Original_Data_Points': len(y_pulse) if 'y_pulse' in locals() else 0,
            'Start_Index': x_pulse[0] if 'x_pulse' in locals() and len(x_pulse) > 0 else np.nan,
            'End_Index': x_pulse[-1] if 'x_pulse' in locals() and len(x_pulse) > 0 else np.nan,
            'Fit_Success': True,
            'Fit_Warnings': f'Emergency fallback fit: {str(e)}',
            'Data_Inverted': False,
            'x_data': np.array([]),
            'y_data': np.array([]),
            'y_data_original': np.array([]),
            'fit_params': [0.0, 0.0, 0.0]
        }

def plot_pulse_comparison(full_result, first9_result, plot_dir):
    """
    Create a comparison plot showing full pulse vs first 9 points fits
    Modified to handle inverted data visualization
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Full pulse plot
        x_full = full_result['x_data']
        y_full = full_result['y_data']
        y_full_original = full_result.get('y_data_original', y_full)
        fit_params_full = full_result['fit_params']
        data_inverted_full = full_result.get('Data_Inverted', False)
        
        if len(x_full) > 0 and len(y_full) > 0:
            # Plot original data in light color if data was inverted
            if data_inverted_full:
                ax1.scatter(x_full, y_full_original, alpha=0.4, color='lightblue', s=15, 
                           label='Original Data (before inversion)', marker='x')
                ax1.scatter(x_full, y_full, alpha=0.7, color='blue', s=20, label='Inverted Data (fitted)')
            else:
                ax1.scatter(x_full, y_full, alpha=0.7, color='blue', s=20, label='Data')
            
            x_fit_full = np.linspace(x_full.min(), x_full.max(), 200)
            y_fit_full = exp_func(x_fit_full, *fit_params_full)
            ax1.plot(x_fit_full, y_fit_full, 'r-', linewidth=2, label='Full Pulse Fit')
            
            # Highlight first 9 points
            if len(x_full) >= 9:
                if data_inverted_full:
                    ax1.scatter(x_full[:9], y_full[:9], alpha=0.9, color='orange', s=30, 
                               label='First 9 Points (inverted)', edgecolors='black', linewidth=1)
                else:
                    ax1.scatter(x_full[:9], y_full[:9], alpha=0.9, color='orange', s=30, 
                               label='First 9 Points', edgecolors='black', linewidth=1)
        
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Signal Value')
        inversion_note = " (Data Inverted)" if data_inverted_full else ""
        ax1.set_title(f"{full_result['Experiment_ID']} - Pulse {full_result['Pulse_Number']} ({full_result['Pulse_Type']}){inversion_note}\n"
                     f"Full Pulse: R² = {full_result['R_Squared']:.4f}, b = {full_result['Parameter_b']:.6f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # First 9 points plot
        x_first9 = first9_result['x_data']
        y_first9 = first9_result['y_data']
        y_first9_original = first9_result.get('y_data_original', y_first9)
        fit_params_first9 = first9_result['fit_params']
        data_inverted_first9 = first9_result.get('Data_Inverted', False)
        
        if len(x_first9) > 0 and len(y_first9) > 0:
            # Plot original data in light color if data was inverted
            if data_inverted_first9:
                ax2.scatter(x_first9, y_first9_original, alpha=0.4, color='lightcoral', s=25, 
                           label='Original Data (before inversion)', marker='x')
                ax2.scatter(x_first9, y_first9, alpha=0.9, color='orange', s=30, 
                           label='First 9 Points (inverted)', edgecolors='black', linewidth=1)
            else:
                ax2.scatter(x_first9, y_first9, alpha=0.9, color='orange', s=30, 
                           label='First 9 Points', edgecolors='black', linewidth=1)
            
            x_fit_first9 = np.linspace(x_first9.min(), x_first9.max(), 100)
            y_fit_first9 = exp_func(x_fit_first9, *fit_params_first9)
            ax2.plot(x_fit_first9, y_fit_first9, 'g-', linewidth=2, label='First 9 Points Fit')
        
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Signal Value')
        inversion_note_first9 = " (Data Inverted)" if data_inverted_first9 else ""
        ax2.set_title(f"First 9 Points Only{inversion_note_first9}\n"
                     f"R² = {first9_result['R_Squared']:.4f}, b = {first9_result['Parameter_b']:.6f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add equation text
        full_eq = f"Full: y = {full_result['Parameter_a']:.4f} × exp({full_result['Parameter_b']:.6f} × x) + {full_result['Parameter_c']:.4f}"
        first9_eq = f"First9: y = {first9_result['Parameter_a']:.4f} × exp({first9_result['Parameter_b']:.6f} × x) + {first9_result['Parameter_c']:.4f}"
        
        fig.text(0.02, 0.02, full_eq, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        fig.text(0.02, 0.08, first9_eq, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{full_result['Experiment_ID']}_Pulse_{full_result['Pulse_Number']:02d}_{full_result['Pulse_Type']}_Comparison.png"
        filepath = plot_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting comparison for pulse {full_result['Experiment_ID']} - {full_result['Pulse_Number']}: {e}")
        plt.close()

def plot_parameter_comparison_extended(all_results, plot_dir):
    """
    Enhanced comparison plots including first 9 points analysis and inversion tracking
    """
    try:
        df = pd.DataFrame(all_results)
        successful_fits = df[df['Fit_Success'] == True]
        
        if len(successful_fits) == 0:
            return
        
        # Separate full and first9 fits
        full_fits = successful_fits[successful_fits['Fit_Type'] == 'full']
        first9_fits = successful_fits[successful_fits['Fit_Type'] == 'first9']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: R-squared comparison (Full vs First9) with inversion markers
        ax1 = axes[0, 0]
        for pulse_type, color in [('Odd', 'red'), ('Even', 'blue')]:
            full_type = full_fits[full_fits['Pulse_Type'] == pulse_type]
            first9_type = first9_fits[first9_fits['Pulse_Type'] == pulse_type]
            
            # Separate inverted and non-inverted data
            full_normal = full_type[~full_type.get('Data_Inverted', False)]
            full_inverted = full_type[full_type.get('Data_Inverted', False)]
            first9_normal = first9_type[~first9_type.get('Data_Inverted', False)]
            first9_inverted = first9_type[first9_type.get('Data_Inverted', False)]
            
            # Plot normal data
            if len(full_normal) > 0 and len(first9_normal) > 0:
                ax1.scatter(full_normal['R_Squared'], first9_normal['R_Squared'], 
                           alpha=0.6, color=color, label=f'{pulse_type} Pulses (Normal)', s=20)
            
            # Plot inverted data with different marker
            if len(full_inverted) > 0 and len(first9_inverted) > 0:
                ax1.scatter(full_inverted['R_Squared'], first9_inverted['R_Squared'], 
                           alpha=0.8, color=color, marker='^', s=30, 
                           label=f'{pulse_type} Pulses (Inverted)', edgecolors='black')
        
        # Add diagonal line
        max_r2 = max(full_fits['R_Squared'].max(), first9_fits['R_Squared'].max())
        ax1.plot([0, max_r2], [0, max_r2], 'k--', alpha=0.5, label='Equal R²')
        ax1.set_xlabel('Full Pulse R²')
        ax1.set_ylabel('First 9 Points R²')
        ax1.set_title('R² Comparison: Full vs First 9 Points\n(Triangles = Inverted Data)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter b comparison (Full vs First9) with inversion markers
        ax2 = axes[0, 1]
        for pulse_type, color in [('Odd', 'red'), ('Even', 'blue')]:
            full_type = full_fits[full_fits['Pulse_Type'] == pulse_type]
            first9_type = first9_fits[first9_fits['Pulse_Type'] == pulse_type]
            
            # Separate inverted and non-inverted data
            full_normal = full_type[~full_type.get('Data_Inverted', False)]
            full_inverted = full_type[full_type.get('Data_Inverted', False)]
            first9_normal = first9_type[~first9_type.get('Data_Inverted', False)]
            first9_inverted = first9_type[first9_type.get('Data_Inverted', False)]
            
            # Plot normal data
            if len(full_normal) > 0 and len(first9_normal) > 0:
                ax2.scatter(full_normal['Parameter_b'], first9_normal['Parameter_b'], 
                           alpha=0.6, color=color, label=f'{pulse_type} Pulses (Normal)', s=20)
            
            # Plot inverted data with different marker
            if len(full_inverted) > 0 and len(first9_inverted) > 0:
                ax2.scatter(full_inverted['Parameter_b'], first9_inverted['Parameter_b'], 
                           alpha=0.8, color=color, marker='^', s=30, 
                           label=f'{pulse_type} Pulses (Inverted)', edgecolors='black')
        
        # Add diagonal line
        b_range = [min(full_fits['Parameter_b'].min(), first9_fits['Parameter_b'].min()),
                   max(full_fits['Parameter_b'].max(), first9_fits['Parameter_b'].max())]
        ax2.plot(b_range, b_range, 'k--', alpha=0.5, label='Equal b')
        ax2.set_xlabel('Full Pulse Parameter b')
        ax2.set_ylabel('First 9 Points Parameter b')
        ax2.set_title('Parameter b Comparison: Full vs First 9 Points')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Inversion frequency by pulse type
        ax3 = axes[0, 2]
        
        inversion_stats = []
        for pulse_type in ['Odd', 'Even']:
            for fit_type in ['full', 'first9']:
                type_data = successful_fits[
                    (successful_fits['Pulse_Type'] == pulse_type) & 
                    (successful_fits['Fit_Type'] == fit_type)
                ]
                if len(type_data) > 0:
                    inverted_count = type_data.get('Data_Inverted', False).sum()
                    total_count = len(type_data)
                    inversion_stats.append({
                        'pulse_type': pulse_type,
                        'fit_type': fit_type,
                        'inversion_rate': inverted_count / total_count,
                        'inverted_count': inverted_count,
                        'total_count': total_count
                    })
        
        if inversion_stats:
            inv_df = pd.DataFrame(inversion_stats)
            
            # Create bar plot of inversion rates
            x_pos = np.arange(len(inv_df))
            colors = ['red' if 'Odd' in row['pulse_type'] else 'blue' for _, row in inv_df.iterrows()]
            patterns = ['///' if row['fit_type'] == 'first9' else '' for _, row in inv_df.iterrows()]
            
            bars = ax3.bar(x_pos, inv_df['inversion_rate'], color=colors, alpha=0.7)
            
            # Add pattern for first9
            for i, (bar, pattern) in enumerate(zip(bars, patterns)):
                if pattern:
                    bar.set_hatch(pattern)
            
            ax3.set_xlabel('Pulse Type and Fit Type')
            ax3.set_ylabel('Inversion Rate')
            ax3.set_title('Data Inversion Rate by Pulse Type\n(Hatched = First 9 Points)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f"{row['pulse_type']}\n{row['fit_type']}" for _, row in inv_df.iterrows()], 
                              rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Full pulse parameter b by pulse type with inversion separation
        ax4 = axes[1, 0]
        full_odd_normal = full_fits[(full_fits['Pulse_Type'] == 'Odd') & (~full_fits.get('Data_Inverted', False))]
        full_odd_inverted = full_fits[(full_fits['Pulse_Type'] == 'Odd') & (full_fits.get('Data_Inverted', False))]
        full_even_normal = full_fits[(full_fits['Pulse_Type'] == 'Even') & (~full_fits.get('Data_Inverted', False))]
        full_even_inverted = full_fits[(full_fits['Pulse_Type'] == 'Even') & (full_fits.get('Data_Inverted', False))]
        
        if len(full_odd_normal) > 0:
            ax4.hist(full_odd_normal['Parameter_b'], alpha=0.7, label='Odd Pulses (Normal)', 
                    color='red', bins=20)
        if len(full_odd_inverted) > 0:
            ax4.hist(full_odd_inverted['Parameter_b'], alpha=0.7, label='Odd Pulses (Inverted)', 
                    color='darkred', bins=20, hatch='///')
        if len(full_even_normal) > 0:
            ax4.hist(full_even_normal['Parameter_b'], alpha=0.7, label='Even Pulses (Normal)', 
                    color='blue', bins=20)
        if len(full_even_inverted) > 0:
            ax4.hist(full_even_inverted['Parameter_b'], alpha=0.7, label='Even Pulses (Inverted)', 
                    color='darkblue', bins=20, hatch='///')
        
        ax4.set_xlabel('Parameter b (Full Pulse)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Full Pulse Parameter b Distribution\n(Hatched = Inverted Data)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: First 9 points parameter b by pulse type with inversion separation
        ax5 = axes[1, 1]
        first9_odd_normal = first9_fits[(first9_fits['Pulse_Type'] == 'Odd') & (~first9_fits.get('Data_Inverted', False))]
        first9_odd_inverted = first9_fits[(first9_fits['Pulse_Type'] == 'Odd') & (first9_fits.get('Data_Inverted', False))]
        first9_even_normal = first9_fits[(first9_fits['Pulse_Type'] == 'Even') & (~first9_fits.get('Data_Inverted', False))]
        first9_even_inverted = first9_fits[(first9_fits['Pulse_Type'] == 'Even') & (first9_fits.get('Data_Inverted', False))]
        
        if len(first9_odd_normal) > 0:
            ax5.hist(first9_odd_normal['Parameter_b'], alpha=0.7, label='Odd Pulses (Normal)', 
                    color='red', bins=20)
        if len(first9_odd_inverted) > 0:
            ax5.hist(first9_odd_inverted['Parameter_b'], alpha=0.7, label='Odd Pulses (Inverted)', 
                    color='darkred', bins=20, hatch='///')
        if len(first9_even_normal) > 0:
            ax5.hist(first9_even_normal['Parameter_b'], alpha=0.7, label='Even Pulses (Normal)', 
                    color='blue', bins=20)
        if len(first9_even_inverted) > 0:
            ax5.hist(first9_even_inverted['Parameter_b'], alpha=0.7, label='Even Pulses (Inverted)', 
                    color='darkblue', bins=20, hatch='///')
        
        ax5.set_xlabel('Parameter b (First 9 Points)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('First 9 Points Parameter b Distribution\n(Hatched = Inverted Data)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Average parameter b by experiment (comparison) with inversion tracking
        ax6 = axes[1, 2]
        experiment_stats = []
        for exp_id in full_fits['Experiment_ID'].unique():
            full_exp = full_fits[full_fits['Experiment_ID'] == exp_id]
            first9_exp = first9_fits[first9_fits['Experiment_ID'] == exp_id]
            
            experiment_stats.append({
                'Experiment': exp_id,
                'Full_Avg_b': full_exp['Parameter_b'].mean(),
                'First9_Avg_b': first9_exp['Parameter_b'].mean() if len(first9_exp) > 0 else np.nan,
                'Full_Inversion_Rate': full_exp.get('Data_Inverted', False).mean(),
                'First9_Inversion_Rate': first9_exp.get('Data_Inverted', False).mean() if len(first9_exp) > 0 else np.nan
            })
        
        exp_df = pd.DataFrame(experiment_stats)
        x_pos = np.arange(len(exp_df))
        
        width = 0.35
        ax6.bar(x_pos - width/2, exp_df['Full_Avg_b'], width, 
               label='Full Pulse', color='lightblue', alpha=0.7)
        ax6.bar(x_pos + width/2, exp_df['First9_Avg_b'], width, 
               label='First 9 Points', color='lightgreen', alpha=0.7)
        
        ax6.set_xlabel('Experiment')
        ax6.set_ylabel('Average Parameter b')
        ax6.set_title('Average Parameter b by Experiment')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(exp_df['Experiment'], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = "Parameter_Comparison_Extended.png"
        filepath = plot_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating extended parameter comparison plot: {e}")
        plt.close()

def plot_individual_pulse(result, plot_dir):
    """
    Create a plot for an individual pulse fit
    Modified to show both original and inverted data when applicable
    """
    try:
        x_data = result['x_data']
        y_data = result['y_data']
        y_data_original = result.get('y_data_original', y_data)
        fit_params = result['fit_params']
        data_inverted = result.get('Data_Inverted', False)
        
        if len(x_data) == 0 or len(y_data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original data and fitted data
        if data_inverted:
            # Show both original and inverted data
            ax.scatter(x_data, y_data_original, alpha=0.5, color='lightcoral', s=15, 
                      label='Original Data (before inversion)', marker='x')
            ax.scatter(x_data, y_data, alpha=0.7, color='blue', s=20, label='Inverted Data (fitted)')
        else:
            # Show only the data (no inversion occurred)
            ax.scatter(x_data, y_data, alpha=0.7, color='blue', s=20, label='Data')
        
        # Plot fitted curve
        x_fit = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit = exp_func(x_fit, *fit_params)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Exponential Fit')
        
        # Add labels and title
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Signal Value')
        inversion_note = " (Data Inverted)" if data_inverted else ""
        ax.set_title(f"{result['Experiment_ID']} - Pulse {result['Pulse_Number']} ({result['Pulse_Type']}) - {result['Fit_Type']}{inversion_note}\n"
                    f"R² = {result['R_Squared']:.4f}, b = {result['Parameter_b']:.6f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add equation text with inversion note
        equation_text = f"y = {result['Parameter_a']:.4f} × exp({result['Parameter_b']:.6f} × x) + {result['Parameter_c']:.4f}"
        if data_inverted:
            equation_text += "\n(Fitted on inverted data: y_inverted = -y_original)"
        
        ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        filename = f"{result['Experiment_ID']}_Pulse_{result['Pulse_Number']:02d}_{result['Pulse_Type']}_{result['Fit_Type']}.png"
        filepath = plot_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting pulse {result['Experiment_ID']} - {result['Pulse_Number']}: {e}")
        plt.close()

def plot_experiment_summary(experiment_results, plot_dir):
    """
    Create a summary plot for all pulses in an experiment
    Modified to show inversion status
    """
    try:
        experiment_id = experiment_results[0]['Experiment_ID']
        
        # Separate odd and even pulses
        odd_pulses = [r for r in experiment_results if r['Pulse_Type'] == 'Odd']
        even_pulses = [r for r in experiment_results if r['Pulse_Type'] == 'Even']
        
        # Create subplot layout
        n_pulses = len(experiment_results)
        n_cols = min(4, n_pulses)
        n_rows = (n_pulses + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_pulses == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(experiment_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            x_data = result['x_data']
            y_data = result['y_data']
            y_data_original = result.get('y_data_original', y_data)
            fit_params = result['fit_params']
            data_inverted = result.get('Data_Inverted', False)
            
            if len(x_data) == 0 or len(y_data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Pulse {result['Pulse_Number']} ({result['Pulse_Type']})")
                continue
            
            # Plot data and fit
            color = 'red' if result['Pulse_Type'] == 'Odd' else 'blue'
            
            if data_inverted:
                # Show original data as light dots
                ax.scatter(x_data, y_data_original, alpha=0.3, color='gray', s=5, marker='x')
                # Show inverted/fitted data
                ax.scatter(x_data, y_data, alpha=0.6, color=color, s=10)
            else:
                ax.scatter(x_data, y_data, alpha=0.6, color=color, s=10)
            
            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = exp_func(x_fit, *fit_params)
            ax.plot(x_fit, y_fit, 'k-', linewidth=1.5)
            
            # Add inversion indicator to title
            inversion_indicator = " (INV)" if data_inverted else ""
            ax.set_title(f"P{result['Pulse_Number']} ({result['Pulse_Type']}){inversion_indicator}\nR²={result['R_Squared']:.3f}")
            ax.grid(True, alpha=0.3)
            
            # Remove labels for cleaner look
            if i // n_cols == n_rows - 1:  # Bottom row
                ax.set_xlabel('Time')
            if i % n_cols == 0:  # Left column
                ax.set_ylabel('Signal')
        
        # Hide unused subplots
        for i in range(len(experiment_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Experiment {experiment_id} - All Pulses Summary (INV = Data Inverted)', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        filename = f"{experiment_id}_All_Pulses_Summary.png"
        filepath = plot_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating summary plot for {experiment_id}: {e}")
        plt.close()

def plot_parameter_comparison(all_results, plot_dir):
    """
    Create comparison plots for parameters across experiments
    Modified to handle inverted data
    """
    try:
        df = pd.DataFrame(all_results)
        successful_fits = df[df['Fit_Success'] == True]
        
        if len(successful_fits) == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: R-squared distribution with inversion markers
        ax1 = axes[0, 0]
        odd_normal = successful_fits[(successful_fits['Pulse_Type'] == 'Odd') & (~successful_fits.get('Data_Inverted', False))]
        odd_inverted = successful_fits[(successful_fits['Pulse_Type'] == 'Odd') & (successful_fits.get('Data_Inverted', False))]
        even_normal = successful_fits[(successful_fits['Pulse_Type'] == 'Even') & (~successful_fits.get('Data_Inverted', False))]
        even_inverted = successful_fits[(successful_fits['Pulse_Type'] == 'Even') & (successful_fits.get('Data_Inverted', False))]
        
        if len(odd_normal) > 0:
            ax1.hist(odd_normal['R_Squared'], alpha=0.7, label='Odd Pulses (Normal)', color='red', bins=20)
        if len(odd_inverted) > 0:
            ax1.hist(odd_inverted['R_Squared'], alpha=0.7, label='Odd Pulses (Inverted)', color='darkred', bins=20, hatch='///')
        if len(even_normal) > 0:
            ax1.hist(even_normal['R_Squared'], alpha=0.7, label='Even Pulses (Normal)', color='blue', bins=20)
        if len(even_inverted) > 0:
            ax1.hist(even_inverted['R_Squared'], alpha=0.7, label='Even Pulses (Inverted)', color='darkblue', bins=20, hatch='///')
        
        ax1.set_xlabel('R-squared')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of R-squared Values\n(Hatched = Inverted Data)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter b distribution with inversion markers
        ax2 = axes[0, 1]
        if len(odd_normal) > 0:
            ax2.hist(odd_normal['Parameter_b'], alpha=0.7, label='Odd Pulses (Normal)', color='red', bins=20)
        if len(odd_inverted) > 0:
            ax2.hist(odd_inverted['Parameter_b'], alpha=0.7, label='Odd Pulses (Inverted)', color='darkred', bins=20, hatch='///')
        if len(even_normal) > 0:
            ax2.hist(even_normal['Parameter_b'], alpha=0.7, label='Even Pulses (Normal)', color='blue', bins=20)
        if len(even_inverted) > 0:
            ax2.hist(even_inverted['Parameter_b'], alpha=0.7, label='Even Pulses (Inverted)', color='darkblue', bins=20, hatch='///')
        
        ax2.set_xlabel('Parameter b (decay/growth rate)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Parameter b Values\n(Hatched = Inverted Data)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R-squared vs Parameter b with inversion markers
        ax3 = axes[1, 0]
        if len(odd_normal) > 0:
            ax3.scatter(odd_normal['Parameter_b'], odd_normal['R_Squared'], 
                       alpha=0.6, color='red', label='Odd Pulses (Normal)', s=20)
        if len(odd_inverted) > 0:
            ax3.scatter(odd_inverted['Parameter_b'], odd_inverted['R_Squared'], 
                       alpha=0.8, color='darkred', marker='^', s=30, 
                       label='Odd Pulses (Inverted)', edgecolors='black')
        if len(even_normal) > 0:
            ax3.scatter(even_normal['Parameter_b'], even_normal['R_Squared'], 
                       alpha=0.6, color='blue', label='Even Pulses (Normal)', s=20)
        if len(even_inverted) > 0:
            ax3.scatter(even_inverted['Parameter_b'], even_inverted['R_Squared'], 
                       alpha=0.8, color='darkblue', marker='^', s=30, 
                       label='Even Pulses (Inverted)', edgecolors='black')
        
        ax3.set_xlabel('Parameter b')
        ax3.set_ylabel('R-squared')
        ax3.set_title('Fit Quality vs Decay/Growth Rate\n(Triangles = Inverted Data)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Average parameter b by experiment with inversion rates
        ax4 = axes[1, 1]
        experiment_stats = []
        for exp_id in successful_fits['Experiment_ID'].unique():
            exp_data = successful_fits[successful_fits['Experiment_ID'] == exp_id]
            odd_exp = exp_data[exp_data['Pulse_Type'] == 'Odd']
            even_exp = exp_data[exp_data['Pulse_Type'] == 'Even']
            
            experiment_stats.append({
                'Experiment': exp_id,
                'Odd_Avg_b': odd_exp['Parameter_b'].mean() if len(odd_exp) > 0 else np.nan,
                'Even_Avg_b': even_exp['Parameter_b'].mean() if len(even_exp) > 0 else np.nan,
                'Odd_Inversion_Rate': odd_exp.get('Data_Inverted', False).mean() if len(odd_exp) > 0 else 0,
                'Even_Inversion_Rate': even_exp.get('Data_Inverted', False).mean() if len(even_exp) > 0 else 0
            })
        
        exp_df = pd.DataFrame(experiment_stats)
        x_pos = np.arange(len(exp_df))
        
        width = 0.35
        bars1 = ax4.bar(x_pos - width/2, exp_df['Odd_Avg_b'], width, 
                       label='Odd Pulses', color='red', alpha=0.7)
        bars2 = ax4.bar(x_pos + width/2, exp_df['Even_Avg_b'], width, 
                       label='Even Pulses', color='blue', alpha=0.7)
        
        # Add hatching for experiments with high inversion rates
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if exp_df.iloc[i]['Odd_Inversion_Rate'] > 0.5:
                bar1.set_hatch('///')
            if exp_df.iloc[i]['Even_Inversion_Rate'] > 0.5:
                bar2.set_hatch('///')
        
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Average Parameter b')
        ax4.set_title('Average Decay/Growth Rate by Experiment\n(Hatched = >50% Inverted)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(exp_df['Experiment'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = "Parameter_Comparison_Summary.png"
        filepath = plot_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating parameter comparison plot: {e}")
        plt.close()

def get_experiment_identifier(file_path):
    """
    Extract a unique identifier from the file path.
    For paths like '/Users/.../DS_00145/PC2-15/PCA_scores.txt'
    this will return 'DS_00145'
    """
    path_parts = Path(file_path).parts
    
    # Look for DS_ pattern in path parts
    for part in path_parts:
        if part.startswith('DS_'):
            return part
    
    # Fallback: use the parent directory name if no DS_ pattern found
    return Path(file_path).parent.name

def load_data_file(file_path):
    """
    Load data from various file formats (txt, csv, etc.)
    Returns x, y arrays or None if loading fails
    """
    try:
        # Try to load with pandas first
        if file_path.suffix.lower() in ['.csv']:
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv(file_path, sep=None, engine='python')
        
        # Assume second column contains the y data
        if data.shape[1] < 2:
            print(f"Warning: {file_path} has less than 2 columns")
            return None, None
            
        x = np.arange(len(data.iloc[:, 1]))
        y = data.iloc[:, 1].values
        return x, y
        
    except:
        # Try numpy loadtxt
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                x = np.arange(len(data))
                y = data
            else:
                x = np.arange(len(data[:, 1]))
                y = data[:, 1]
            return x, y
        except:
            # Last resort: read as text and parse
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                data = []
                for line in lines:
                    try:
                        values = line.strip().split()
                        if len(values) > 1:
                            data.append(float(values[1]))
                    except:
                        continue
                
                if len(data) == 0:
                    return None, None
                    
                x = np.arange(len(data))
                y = np.array(data)
                return x, y
            except:
                print(f"Error: Could not load data from {file_path}")
                return None, None

def process_single_file(file_path, pulse_length=91, plot_dir=None, create_plots=True):
    """
    Process a single file and return list of fit results for all pulses
    """
    print(f"Processing file: {file_path}")
    
    # Load data
    x, y = load_data_file(file_path)
    if x is None or y is None:
        print(f"Failed to load data from {file_path}")
        return []
    
    print(f"Loaded {len(y)} data points from {file_path}")
    
    # Get experiment identifier
    experiment_id = get_experiment_identifier(file_path)
    print(f"Experiment ID: {experiment_id}")
    
    # Calculate number of pulses
    num_pulses = (len(x) + pulse_length - 1) // pulse_length  # Ceiling division
    
    results = []
    
    for pulse_num in range(num_pulses):
        start_idx = pulse_num * pulse_length
        end_idx = min((pulse_num + 1) * pulse_length, len(x))
        
        # Skip if we're past the end of the data
        if start_idx >= len(x):
            break
        
        # Extract pulse data
        x_pulse = x[start_idx:end_idx] - x[start_idx]  # Reset x to start from 0 for each pulse
        y_pulse = y[start_idx:end_idx]
        
        # Determine if this is an odd or even pulse (1-indexed)
        pulse_type = 'Odd' if (pulse_num + 1) % 2 == 1 else 'Even'
        
        # Fit the full pulse
        result_full = fit_single_pulse(x_pulse, y_pulse, pulse_num + 1, pulse_type, experiment_id, "full")
        results.append(result_full)
        
        # Fit only the first 9 data points
        if len(x_pulse) >= 9:
            x_pulse_first9 = x_pulse[:9]
            y_pulse_first9 = y_pulse[:9]
            result_first9 = fit_single_pulse(x_pulse_first9, y_pulse_first9, pulse_num + 1, pulse_type, experiment_id, "first9")
            results.append(result_first9)
            
            inversion_note_full = " (inverted)" if result_full.get('Data_Inverted', False) else ""
            inversion_note_first9 = " (inverted)" if result_first9.get('Data_Inverted', False) else ""
            print(f"  Pulse {pulse_num + 1} ({pulse_type}): Full R² = {result_full['R_Squared']:.4f}{inversion_note_full}, First9 R² = {result_first9['R_Squared']:.4f}{inversion_note_first9}")
        else:
            inversion_note_full = " (inverted)" if result_full.get('Data_Inverted', False) else ""
            print(f"  Pulse {pulse_num + 1} ({pulse_type}): Full R² = {result_full['R_Squared']:.4f}{inversion_note_full} (insufficient data for first9 fit)")
    
    # Create plots if requested
    if create_plots and plot_dir is not None:
        # Individual pulse plots
        individual_plot_dir = plot_dir / "individual_pulses"
        individual_plot_dir.mkdir(exist_ok=True)
        
        # Comparison plots directory
        comparison_plot_dir = plot_dir / "pulse_comparisons"
        comparison_plot_dir.mkdir(exist_ok=True)
        
        # Plot individual pulses and comparisons
        full_results = [r for r in results if r['Fit_Type'] == 'full']
        first9_results = [r for r in results if r['Fit_Type'] == 'first9']
        
        for result in results:
            plot_individual_pulse(result, individual_plot_dir)
        
        # Create comparison plots (full vs first9)
        for full_result in full_results:
            # Find matching first9 result
            matching_first9 = [r for r in first9_results 
                             if r['Pulse_Number'] == full_result['Pulse_Number']]
            if matching_first9:
                plot_pulse_comparison(full_result, matching_first9[0], comparison_plot_dir)
        
        # Experiment summary plot
        experiment_plot_dir = plot_dir / "experiment_summaries"
        experiment_plot_dir.mkdir(exist_ok=True)
        
        plot_experiment_summary(full_results, experiment_plot_dir)  # Only use full results for summary
    
    return results

# ===================================================================
# CONFIGURATION SECTION - EDIT THESE PATHS AND SETTINGS
# ===================================================================

# List all your input files here (add as many as you want)
INPUT_FILES = [
    "/Users/danielsinausia/Documents/Experiments/DS_00127/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00131/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00132/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00133/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00134/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00135/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00136/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00137/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00138/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00139/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00140/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00141/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00142/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00143/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00144/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00145/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00146/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00147/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00148/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00149/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00150/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00152/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00153/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00163/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00180/PC2-15/PCA_scores.txt",
    "/Users/danielsinausia/Documents/Experiments/DS_00181/PC2-15/PCA_scores.txt",
]

# Output Excel file path
OUTPUT_FILE = "/Users/danielsinausia/Downloads/pulse_exponential_fits_results.xlsx"

# Output plots directory
PLOTS_DIR = "/Users/danielsinausia/Downloads/pulse_fitting_plots"

# Pulse length (number of data points per pulse)
PULSE_LENGTH = 91

# Plot settings
CREATE_PLOTS = True  # Set to False to disable plotting
PLOT_DPI = 300      # Resolution for saved plots

# ===================================================================
# END CONFIGURATION SECTION
# ===================================================================

def process_files_with_config():
    """
    Process files using the configuration defined above
    """
    print("Starting pulse exponential fitting analysis with Y-inversion capability...")
    print(f"Input files: {len(INPUT_FILES)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Pulse length: {PULSE_LENGTH} points")
    print(f"Create plots: {CREATE_PLOTS}")
    if CREATE_PLOTS:
        print(f"Plots directory: {PLOTS_DIR}")
    print("-" * 50)
    
    # Create plots directory if plotting is enabled
    plot_dir = None
    if CREATE_PLOTS:
        plot_dir = Path(PLOTS_DIR)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created plots directory: {plot_dir}")
    
    # Process all files
    all_results = []
    
    for file_path_str in INPUT_FILES:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        if not file_path.is_file():
            print(f"Warning: {file_path} is not a file, skipping...")
            continue
        
        # Process this file
        file_results = process_single_file(file_path, PULSE_LENGTH, plot_dir, CREATE_PLOTS)
        all_results.extend(file_results)
    
    if not all_results:
        print("No data was processed successfully!")
        return
    
    # Create overall comparison plots
    if CREATE_PLOTS and plot_dir is not None:
        print("Creating parameter comparison plots...")
        plot_parameter_comparison(all_results, plot_dir)
        plot_parameter_comparison_extended(all_results, plot_dir)
    
    # Remove plotting data from results before saving to Excel
    results_for_excel = []
    for result in all_results:
        excel_result = {k: v for k, v in result.items() 
                       if k not in ['x_data', 'y_data', 'y_data_original', 'fit_params']}
        results_for_excel.append(excel_result)
    
    # Create DataFrame with separate sheets for full and first9 fits
    df = pd.DataFrame(results_for_excel)
    
    # Separate full and first9 fits
    df_full = df[df['Fit_Type'] == 'full'].copy()
    df_first9 = df[df['Fit_Type'] == 'first9'].copy()
    
    # Sort by experiment ID, then pulse number
    df_full = df_full.sort_values(['Experiment_ID', 'Pulse_Number'])
    df_first9 = df_first9.sort_values(['Experiment_ID', 'Pulse_Number'])
    
    # Create summary statistics for both fit types
    def create_summary_stats(data_df, fit_type_name):
        summary_stats = []
        
        # Overall statistics
        successful_fits = data_df[data_df['Fit_Success'] == True]
        inverted_fits = successful_fits[successful_fits.get('Data_Inverted', False) == True]
        
        summary_stats.append({
            'Category': f'{fit_type_name}_Overall',
            'Subcategory': 'All Pulses',
            'Count': len(data_df),
            'Successful_Fits': len(successful_fits),
            'Success_Rate': len(successful_fits) / len(data_df) if len(data_df) > 0 else 0,
            'Inverted_Fits': len(inverted_fits),
            'Inversion_Rate': len(inverted_fits) / len(successful_fits) if len(successful_fits) > 0 else 0,
            'Avg_R_Squared': successful_fits['R_Squared'].mean() if len(successful_fits) > 0 else np.nan,
            'Avg_Parameter_b': successful_fits['Parameter_b'].mean() if len(successful_fits) > 0 else np.nan,
            'Std_Parameter_b': successful_fits['Parameter_b'].std() if len(successful_fits) > 0 else np.nan
        })
        
        # Statistics by pulse type
        for pulse_type in ['Odd', 'Even']:
            type_data = successful_fits[successful_fits['Pulse_Type'] == pulse_type]
            type_inverted = type_data[type_data.get('Data_Inverted', False) == True]
            
            summary_stats.append({
                'Category': f'{fit_type_name}_By_Pulse_Type',
                'Subcategory': pulse_type,
                'Count': len(data_df[data_df['Pulse_Type'] == pulse_type]),
                'Successful_Fits': len(type_data),
                'Success_Rate': len(type_data) / len(data_df[data_df['Pulse_Type'] == pulse_type]) if len(data_df[data_df['Pulse_Type'] == pulse_type]) > 0 else 0,
                'Inverted_Fits': len(type_inverted),
                'Inversion_Rate': len(type_inverted) / len(type_data) if len(type_data) > 0 else 0,
                'Avg_R_Squared': type_data['R_Squared'].mean() if len(type_data) > 0 else np.nan,
                'Avg_Parameter_b': type_data['Parameter_b'].mean() if len(type_data) > 0 else np.nan,
                'Std_Parameter_b': type_data['Parameter_b'].std() if len(type_data) > 0 else np.nan
            })
        
        # Statistics by experiment
        for experiment_id in data_df['Experiment_ID'].unique():
            experiment_data = successful_fits[successful_fits['Experiment_ID'] == experiment_id]
            experiment_inverted = experiment_data[experiment_data.get('Data_Inverted', False) == True]
            
            summary_stats.append({
                'Category': f'{fit_type_name}_By_Experiment',
                'Subcategory': experiment_id,
                'Count': len(data_df[data_df['Experiment_ID'] == experiment_id]),
                'Successful_Fits': len(experiment_data),
                'Success_Rate': len(experiment_data) / len(data_df[data_df['Experiment_ID'] == experiment_id]) if len(data_df[data_df['Experiment_ID'] == experiment_id]) > 0 else 0,
                'Inverted_Fits': len(experiment_inverted),
                'Inversion_Rate': len(experiment_inverted) / len(experiment_data) if len(experiment_data) > 0 else 0,
                'Avg_R_Squared': experiment_data['R_Squared'].mean() if len(experiment_data) > 0 else np.nan,
                'Avg_Parameter_b': experiment_data['Parameter_b'].mean() if len(experiment_data) > 0 else np.nan,
                'Std_Parameter_b': experiment_data['Parameter_b'].std() if len(experiment_data) > 0 else np.nan
            })
        
        return summary_stats
    
    # Create summary statistics for both fit types
    summary_stats_full = create_summary_stats(df_full, "Full_Pulse")
    summary_stats_first9 = create_summary_stats(df_first9, "First9_Points")
    
    # Combine summary statistics
    all_summary_stats = summary_stats_full + summary_stats_first9
    summary_df = pd.DataFrame(all_summary_stats)
    
    # Create comparison summary between full and first9 fits
    comparison_data = []
    for experiment_id in df_full['Experiment_ID'].unique():
        full_exp = df_full[df_full['Experiment_ID'] == experiment_id]
        first9_exp = df_first9[df_first9['Experiment_ID'] == experiment_id]
        
        for pulse_type in ['Odd', 'Even']:
            full_type = full_exp[full_exp['Pulse_Type'] == pulse_type]
            first9_type = first9_exp[first9_exp['Pulse_Type'] == pulse_type]
            
            comparison_data.append({
                'Experiment_ID': experiment_id,
                'Pulse_Type': pulse_type,
                'Full_Count': len(full_type),
                'First9_Count': len(first9_type),
                'Full_Inverted_Count': len(full_type[full_type.get('Data_Inverted', False) == True]),
                'First9_Inverted_Count': len(first9_type[first9_type.get('Data_Inverted', False) == True]),
                'Full_Inversion_Rate': len(full_type[full_type.get('Data_Inverted', False) == True]) / len(full_type) if len(full_type) > 0 else 0,
                'First9_Inversion_Rate': len(first9_type[first9_type.get('Data_Inverted', False) == True]) / len(first9_type) if len(first9_type) > 0 else 0,
                'Full_Avg_R_Squared': full_type['R_Squared'].mean() if len(full_type) > 0 else np.nan,
                'First9_Avg_R_Squared': first9_type['R_Squared'].mean() if len(first9_type) > 0 else np.nan,
                'Full_Avg_Parameter_b': full_type['Parameter_b'].mean() if len(full_type) > 0 else np.nan,
                'First9_Avg_Parameter_b': first9_type['Parameter_b'].mean() if len(first9_type) > 0 else np.nan,
                'Full_Std_Parameter_b': full_type['Parameter_b'].std() if len(full_type) > 0 else np.nan,
                'First9_Std_Parameter_b': first9_type['Parameter_b'].std() if len(first9_type) > 0 else np.nan,
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to Excel with multiple sheets
    output_path = Path(OUTPUT_FILE)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results - Full pulse fits
        df_full.to_excel(writer, sheet_name='Full_Pulse_Fits', index=False)
        
        # First 9 points fits
        if len(df_first9) > 0:
            df_first9.to_excel(writer, sheet_name='First9_Points_Fits', index=False)
        
        # Combined summary statistics
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Comparison between full and first9
        if len(comparison_df) > 0:
            comparison_df.to_excel(writer, sheet_name='Full_vs_First9_Comparison', index=False)
        
        # Separate sheets for odd and even pulses (full fits)
        odd_pulses_full = df_full[df_full['Pulse_Type'] == 'Odd']
        even_pulses_full = df_full[df_full['Pulse_Type'] == 'Even']
        
        if len(odd_pulses_full) > 0:
            odd_pulses_full.to_excel(writer, sheet_name='Full_Odd_Pulses_Only', index=False)
        
        if len(even_pulses_full) > 0:
            even_pulses_full.to_excel(writer, sheet_name='Full_Even_Pulses_Only', index=False)
        
        # Separate sheets for odd and even pulses (first9 fits)
        if len(df_first9) > 0:
            odd_pulses_first9 = df_first9[df_first9['Pulse_Type'] == 'Odd']
            even_pulses_first9 = df_first9[df_first9['Pulse_Type'] == 'Even']
            
            if len(odd_pulses_first9) > 0:
                odd_pulses_first9.to_excel(writer, sheet_name='First9_Odd_Pulses_Only', index=False)
            
            if len(even_pulses_first9) > 0:
                even_pulses_first9.to_excel(writer, sheet_name='First9_Even_Pulses_Only', index=False)
        
        # Create inversion analysis sheets
        inverted_full = df_full[df_full.get('Data_Inverted', False) == True]
        normal_full = df_full[df_full.get('Data_Inverted', False) == False]
        
        if len(inverted_full) > 0:
            inverted_full.to_excel(writer, sheet_name='Inverted_Data_Full', index=False)
        
        if len(normal_full) > 0:
            normal_full.to_excel(writer, sheet_name='Normal_Data_Full', index=False)
        
        if len(df_first9) > 0:
            inverted_first9 = df_first9[df_first9.get('Data_Inverted', False) == True]
            normal_first9 = df_first9[df_first9.get('Data_Inverted', False) == False]
            
            if len(inverted_first9) > 0:
                inverted_first9.to_excel(writer, sheet_name='Inverted_Data_First9', index=False)
            
            if len(normal_first9) > 0:
                normal_first9.to_excel(writer, sheet_name='Normal_Data_First9', index=False)
    
    # Calculate statistics for reporting
    total_pulses = len(df_full)
    total_first9_fits = len(df_first9)
    successful_full_fits = len(df_full[df_full['Fit_Success'] == True])
    successful_first9_fits = len(df_first9[df_first9['Fit_Success'] == True])
    inverted_full_fits = len(df_full[df_full.get('Data_Inverted', False) == True])
    inverted_first9_fits = len(df_first9[df_first9.get('Data_Inverted', False) == True])
    
    print(f"\nProcessing complete!")
    print(f"Processed {total_pulses} total pulses from {len(df_full['Experiment_ID'].unique())} experiments")
    print(f"Full pulse fits: {successful_full_fits}/{total_pulses} ({successful_full_fits/total_pulses*100:.1f}%)")
    print(f"Full pulse inversions: {inverted_full_fits}/{successful_full_fits} ({inverted_full_fits/successful_full_fits*100:.1f}%)")
    if total_first9_fits > 0:
        print(f"First 9 points fits: {successful_first9_fits}/{total_first9_fits} ({successful_first9_fits/total_first9_fits*100:.1f}%)")
        print(f"First 9 points inversions: {inverted_first9_fits}/{successful_first9_fits} ({inverted_first9_fits/successful_first9_fits*100:.1f}%)")
    print(f"Results saved to: {output_path}")
    
    if CREATE_PLOTS:
        print(f"\nPlots saved to:")
        print(f"  Individual pulses: {plot_dir / 'individual_pulses'}")
        print(f"  Pulse comparisons: {plot_dir / 'pulse_comparisons'}")
        print(f"  Experiment summaries: {plot_dir / 'experiment_summaries'}")
        print(f"  Parameter comparisons: {plot_dir / 'Parameter_Comparison_Summary.png'}")
        print(f"  Extended comparisons: {plot_dir / 'Parameter_Comparison_Extended.png'}")
    
    # Print summary by pulse type for both fit types including inversion rates
    if successful_full_fits > 0:
        print(f"\nFull Pulse Summary by pulse type:")
        for pulse_type in ['Odd', 'Even']:
            type_data = df_full[(df_full['Pulse_Type'] == pulse_type) & (df_full['Fit_Success'] == True)]
            type_inverted = type_data[type_data.get('Data_Inverted', False) == True]
            if len(type_data) > 0:
                inversion_rate = len(type_inverted) / len(type_data) * 100
                print(f"  {pulse_type} pulses: {len(type_data)} fits, {len(type_inverted)} inverted ({inversion_rate:.1f}%), avg R² = {type_data['R_Squared'].mean():.4f}, avg b = {type_data['Parameter_b'].mean():.6f}")
    
    if successful_first9_fits > 0:
        print(f"\nFirst 9 Points Summary by pulse type:")
        for pulse_type in ['Odd', 'Even']:
            type_data = df_first9[(df_first9['Pulse_Type'] == pulse_type) & (df_first9['Fit_Success'] == True)]
            type_inverted = type_data[type_data.get('Data_Inverted', False) == True]
            if len(type_data) > 0:
                inversion_rate = len(type_inverted) / len(type_data) * 100
                print(f"  {pulse_type} pulses: {len(type_data)} fits, {len(type_inverted)} inverted ({inversion_rate:.1f}%), avg R² = {type_data['R_Squared'].mean():.4f}, avg b = {type_data['Parameter_b'].mean():.6f}")
    
    return df_full, df_first9, summary_df

if __name__ == "__main__":
    # Run the analysis with the configured settings
    process_files_with_config()
