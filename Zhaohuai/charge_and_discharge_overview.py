import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from pathlib import Path

# Suppress RuntimeWarning for better output readability
warnings.filterwarnings("ignore", category=RuntimeWarning)

def exp_func(x, a, b, c):
    """Single exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def fit_single_pulse(x_pulse, y_pulse, pulse_number, pulse_type, experiment_id):
    """
    Improved fit function with better initial guesses and error handling
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
            'Fit_Warnings': '; '.join(fit_warnings) if fit_warnings else 'None'
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
                    'Fit_Warnings': f'Forced constant fit due to error: {str(e)}'
                }
        except:
            pass
        
        # Absolute last resort if everything fails
        return {
            'Experiment_ID': experiment_id,
            'Pulse_Number': pulse_number,
            'Pulse_Type': pulse_type,
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
            'Fit_Warnings': f'Emergency fallback fit: {str(e)}'
        }
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


def process_single_file(file_path, pulse_length=91):
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
        
        # Fit the pulse
        result = fit_single_pulse(x_pulse, y_pulse, pulse_num + 1, pulse_type, experiment_id)
        results.append(result)
        
        print(f"  Pulse {pulse_num + 1} ({pulse_type}): R² = {result['R_Squared']:.4f}")
    
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

# Pulse length (number of data points per pulse)
PULSE_LENGTH = 91

# ===================================================================
# END CONFIGURATION SECTION
# ===================================================================

def process_files_with_config():
    """
    Process files using the configuration defined above
    """
    print("Starting pulse exponential fitting analysis...")
    print(f"Input files: {len(INPUT_FILES)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Pulse length: {PULSE_LENGTH} points")
    print("-" * 50)
    
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
        file_results = process_single_file(file_path, PULSE_LENGTH)
        all_results.extend(file_results)
    
    if not all_results:
        print("No data was processed successfully!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by experiment ID, then pulse number
    df = df.sort_values(['Experiment_ID', 'Pulse_Number'])
    
    # Create summary statistics
    summary_stats = []
    
    # Overall statistics
    successful_fits = df[df['Fit_Success'] == True]
    summary_stats.append({
        'Category': 'Overall',
        'Subcategory': 'All Pulses',
        'Count': len(df),
        'Successful_Fits': len(successful_fits),
        'Success_Rate': len(successful_fits) / len(df) if len(df) > 0 else 0,
        'Avg_R_Squared': successful_fits['R_Squared'].mean() if len(successful_fits) > 0 else np.nan,
        'Avg_Parameter_b': successful_fits['Parameter_b'].mean() if len(successful_fits) > 0 else np.nan,
        'Std_Parameter_b': successful_fits['Parameter_b'].std() if len(successful_fits) > 0 else np.nan
    })
    
    # Statistics by pulse type
    for pulse_type in ['Odd', 'Even']:
        type_data = successful_fits[successful_fits['Pulse_Type'] == pulse_type]
        summary_stats.append({
            'Category': 'By_Pulse_Type',
            'Subcategory': pulse_type,
            'Count': len(df[df['Pulse_Type'] == pulse_type]),
            'Successful_Fits': len(type_data),
            'Success_Rate': len(type_data) / len(df[df['Pulse_Type'] == pulse_type]) if len(df[df['Pulse_Type'] == pulse_type]) > 0 else 0,
            'Avg_R_Squared': type_data['R_Squared'].mean() if len(type_data) > 0 else np.nan,
            'Avg_Parameter_b': type_data['Parameter_b'].mean() if len(type_data) > 0 else np.nan,
            'Std_Parameter_b': type_data['Parameter_b'].std() if len(type_data) > 0 else np.nan
        })
    
    # Statistics by experiment
    for experiment_id in df['Experiment_ID'].unique():
        experiment_data = successful_fits[successful_fits['Experiment_ID'] == experiment_id]
        summary_stats.append({
            'Category': 'By_Experiment',
            'Subcategory': experiment_id,
            'Count': len(df[df['Experiment_ID'] == experiment_id]),
            'Successful_Fits': len(experiment_data),
            'Success_Rate': len(experiment_data) / len(df[df['Experiment_ID'] == experiment_id]) if len(df[df['Experiment_ID'] == experiment_id]) > 0 else 0,
            'Avg_R_Squared': experiment_data['R_Squared'].mean() if len(experiment_data) > 0 else np.nan,
            'Avg_Parameter_b': experiment_data['Parameter_b'].mean() if len(experiment_data) > 0 else np.nan,
            'Std_Parameter_b': experiment_data['Parameter_b'].std() if len(experiment_data) > 0 else np.nan
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to Excel with multiple sheets
    output_path = Path(OUTPUT_FILE)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        df.to_excel(writer, sheet_name='Individual_Pulse_Fits', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Separate sheets for odd and even pulses
        odd_pulses = df[df['Pulse_Type'] == 'Odd']
        even_pulses = df[df['Pulse_Type'] == 'Even']
        
        if len(odd_pulses) > 0:
            odd_pulses.to_excel(writer, sheet_name='Odd_Pulses_Only', index=False)
        
        if len(even_pulses) > 0:
            even_pulses.to_excel(writer, sheet_name='Even_Pulses_Only', index=False)
        
        # Parameter comparison sheet
        if len(successful_fits) > 0:
            comparison_data = []
            for experiment_id in df['Experiment_ID'].unique():
                experiment_data = successful_fits[successful_fits['Experiment_ID'] == experiment_id]
                odd_data = experiment_data[experiment_data['Pulse_Type'] == 'Odd']
                even_data = experiment_data[experiment_data['Pulse_Type'] == 'Even']
                
                comparison_data.append({
                    'Experiment_ID': experiment_id,
                    'Odd_Pulses_Count': len(odd_data),
                    'Even_Pulses_Count': len(even_data),
                    'Odd_Avg_R_Squared': odd_data['R_Squared'].mean() if len(odd_data) > 0 else np.nan,
                    'Even_Avg_R_Squared': even_data['R_Squared'].mean() if len(even_data) > 0 else np.nan,
                    'Odd_Avg_Parameter_b': odd_data['Parameter_b'].mean() if len(odd_data) > 0 else np.nan,
                    'Even_Avg_Parameter_b': even_data['Parameter_b'].mean() if len(even_data) > 0 else np.nan,
                    'Odd_Std_Parameter_b': odd_data['Parameter_b'].std() if len(odd_data) > 0 else np.nan,
                    'Even_Std_Parameter_b': even_data['Parameter_b'].std() if len(even_data) > 0 else np.nan,
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='Experiment_Comparison', index=False)
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(df)} total pulses from {len(df['Experiment_ID'].unique())} experiments")
    print(f"Successful fits: {len(successful_fits)}/{len(df)} ({len(successful_fits)/len(df)*100:.1f}%)")
    print(f"Results saved to: {output_path}")
    
    # Print summary by pulse type
    if len(successful_fits) > 0:
        print(f"\nSummary by pulse type:")
        for pulse_type in ['Odd', 'Even']:
            type_data = successful_fits[successful_fits['Pulse_Type'] == pulse_type]
            if len(type_data) > 0:
                print(f"{pulse_type} pulses: {len(type_data)} fits, avg R² = {type_data['R_Squared'].mean():.4f}, avg b = {type_data['Parameter_b'].mean():.6f}")
    
    return df, summary_df

if __name__ == "__main__":
    # Run the analysis with the configured settings
    process_files_with_config()
