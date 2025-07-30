'''
Collects data from time range 160-180 seconds (instead of just 176s) from all experiments
and creates bar charts with error bars showing mean ± standard deviation.

This combines the functionality of scripts 2 and 3, extracting data over a time range
and creating comparative visualizations with statistical error bars.
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set matplotlib backend and font settings
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Preserve fonts in SVG

def find_csv_files(base_path, target_filename, target_folder):
    """
    Find all CSV files with the specified name inside folders with the target folder name.
    
    Args:
        base_path (str): Root directory to search
        target_filename (str): Name of the CSV file to find
        target_folder (str): Name of the folder that should contain the CSV
    
    Returns:
        list: List of full paths to found CSV files
    """
    found_files = []
    
    for root, dirs, files in os.walk(base_path):
        # Check if current directory name matches target folder
        if os.path.basename(root) == target_folder:
            # Check if target CSV file exists in this directory
            if target_filename in files:
                csv_path = os.path.join(root, target_filename)
                found_files.append(csv_path)
                print(f"Found: {csv_path}")
    
    return found_files

def extract_time_range_data(csv_path, time_start=160, time_end=180):
    """
    Extract data for time range (default 160-180 seconds) from the CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        time_start (int): Start time in seconds
        time_end (int): End time in seconds
    
    Returns:
        dict: Dictionary with peak names and their statistics (mean, std, count) in the time range
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Print column names for debugging
        print(f"Processing {csv_path}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle files with 'Statistic' column - these seem to have summary data, skip them
        if 'Statistic' in df.columns:
            print(f"Skipping file with 'Statistic' column: {csv_path}")
            return None
        
        # Find rows within the time range
        time_column = df.columns[0]  # Assuming first column is time
        
        # Ensure time column is numeric
        df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
        df = df.dropna(subset=[time_column])  # Remove rows with non-numeric time values
        
        time_range_data = df[(df[time_column] >= time_start) & (df[time_column] <= time_end)]
        
        if time_range_data.empty:
            print(f"Warning: No data found in time range {time_start}-{time_end}s in {csv_path}")
            return None
        
        print(f"Found {len(time_range_data)} data points in time range {time_start}-{time_end}s")
        
        # Extract statistics for the specified peaks
        peak_columns = ['Mean 3680', 'Mean 3520', 'Mean 3360', 'Mean 3210', 'Mean 3100', 'Mean 2870']
        result = {}
        
        for peak in peak_columns:
            if peak in df.columns:
                peak_data = time_range_data[peak].dropna()  # Remove NaN values
                if len(peak_data) > 0:
                    result[f'{peak}_mean'] = peak_data.mean()
                    result[f'{peak}_std'] = peak_data.std()
                    result[f'{peak}_count'] = len(peak_data)
                    result[f'{peak}_min'] = peak_data.min()
                    result[f'{peak}_max'] = peak_data.max()
                else:
                    print(f"Warning: No valid data for '{peak}' in time range")
                    result[f'{peak}_mean'] = np.nan
                    result[f'{peak}_std'] = np.nan
                    result[f'{peak}_count'] = 0
                    result[f'{peak}_min'] = np.nan
                    result[f'{peak}_max'] = np.nan
            else:
                print(f"Warning: Column '{peak}' not found in {csv_path}")
                result[f'{peak}_mean'] = np.nan
                result[f'{peak}_std'] = np.nan
                result[f'{peak}_count'] = 0
                result[f'{peak}_min'] = np.nan
                result[f'{peak}_max'] = np.nan
        
        # Add metadata
        result['Source_File'] = os.path.basename(csv_path)
        result['Source_Path'] = csv_path
        result['Time_Start'] = time_start
        result['Time_End'] = time_end
        result['Total_Points'] = len(time_range_data)
        
        return result
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def map_file_label(file_path):
    """
    Map file path to analysis type label.
    """
    if 'Reconstruction_based_on_CO_peak_in_eigenspectra/1101_to_3999/Integrations Water Normalized' in file_path:
        return 'Raw data'
    elif 'Raw data/Integrations Water Normalized' in file_path:
        return 'Raw data'
    elif '1101_to_3999/Raw data/Integrations Water Normalized' in file_path:
        return 'Raw data'
    elif 'Reconstruction_based_on_CO_peak_in_eigenspectra/mean-center contribution' in file_path:
        return 'MC'
    elif 'mean-center contribution' in file_path:
        return 'MC'
    elif 'Reconstruction_based_on_CO_peak_in_eigenspectra/Diffusion_layer' in file_path:
        return 'Diffusion'
    elif 'Diffusion_layer' in file_path:
        return 'Diffusion'
    elif 'Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer' in file_path:
        return 'Interfacial'
    elif 'Interfacial_layer' in file_path:
        return 'Interfacial'
    elif 'PC1' in file_path:
        return 'PC1'
    elif 'PC2-15' in file_path:
        return 'PC2-15'
    elif 'spectral_clustering' in file_path:
        return 'Spectral Clustering'
    else:
        return None

def extract_experiment_name(file_path):
    """
    Extract experiment name from file path (DS_XXXXX pattern).
    """
    path_parts = file_path.split('/')
    
    # Find the index of 'Experiments' in the path
    try:
        experiments_index = path_parts.index('Experiments')
        # The experiment name should be the next part after 'Experiments'
        if experiments_index + 1 < len(path_parts):
            experiment_name = path_parts[experiments_index + 1]
            # Verify it follows DS_XXXXX pattern
            if experiment_name.startswith('DS_') and len(experiment_name) == 8:
                return experiment_name
    except ValueError:
        pass
    
    # Fallback: look for any DS_XXXXX pattern in the path
    for part in path_parts:
        if part.startswith('DS_') and len(part) == 8:
            return part
    
    # Final fallback
    return os.path.basename(os.path.dirname(file_path))

def create_bar_charts_with_error_bars(df, output_dir, time_start=160, time_end=180):
    """
    Create bar charts with error bars for each experiment.
    
    Args:
        df (pd.DataFrame): DataFrame with mean and std data
        output_dir (str): Output directory for plots
        time_start (int): Start time for labeling
        time_end (int): End time for labeling
    """
    # Define water structure columns and their display names
    water_structure_columns = ['Mean 3680', 'Mean 3520', 'Mean 3360', 'Mean 3210', 'Mean 3100', 'Mean 2870']
    
    # Initialize color palette
    palette = sns.color_palette("magma", n_colors=len(water_structure_columns))
    
    # Define the desired order for the short labels (expanded to include more categories)
    short_label_order = ['Raw data', 'MC', 'Diffusion', 'Interfacial', 'PC1', 'PC2-15', 'Spectral Clustering']
    
    # Get unique experiments
    experiments = df['Experiment'].unique()
    
    # Create a bar chart for each experiment
    for experiment in experiments:
        # Filter data for the current experiment
        exp_data = df[df['Experiment'] == experiment].copy()
        
        print(f"Processing Experiment: {experiment}")
        print("Unique Short Labels:", exp_data['Short Label'].unique())
        
        # Define the label order
        exp_data['Short Label'] = pd.Categorical(
            exp_data['Short Label'],
            categories=short_label_order,
            ordered=True
        )
        
        # Sort by the defined label order
        exp_data = exp_data.sort_values('Short Label')
        
        # Prepare data for plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars and positions
        bar_width = 0.12
        x_positions = np.arange(len(exp_data))
        
        # Plot bars for each water structure
        for i, (peak, color) in enumerate(zip(water_structure_columns, palette)):
            means = exp_data[f'{peak}_mean'].values
            stds = exp_data[f'{peak}_std'].values
            
            # Handle NaN values
            mask = ~np.isnan(means)
            x_pos = x_positions[mask] + i * bar_width
            
            ax.bar(x_pos, means[mask], bar_width, 
                   label=peak, color=color, alpha=0.8,
                   yerr=stds[mask], capsize=3)
        
        # Customize the chart
        ax.set_title(f"Experiment: {experiment} - Time Range: {time_start}-{time_end}s", fontsize=16)
        ax.set_xlabel("Analysis Type", fontsize=12)
        ax.set_ylabel("Integrated Area (a.u.)", fontsize=12)
        ax.set_xticks(x_positions + bar_width * (len(water_structure_columns) - 1) / 2)
        ax.set_xticklabels(exp_data['Short Label'], rotation=45)
        ax.legend(title='Water Structure', fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plot_name = f"{experiment}_time_{time_start}_{time_end}_bar_chart_with_errors"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{plot_name}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{plot_name}.svg"), bbox_inches='tight')
        plt.close()
        
        # Also create a stacked bar chart with error bars (alternative visualization)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create stacked bar chart
        bottom = np.zeros(len(exp_data))
        
        for i, (peak, color) in enumerate(zip(water_structure_columns, palette)):
            means = exp_data[f'{peak}_mean'].fillna(0).values
            stds = exp_data[f'{peak}_std'].fillna(0).values
            
            bars = ax.bar(exp_data['Short Label'], means, bottom=bottom, 
                         label=peak, color=color, alpha=0.8)
            
            # Add error bars at the top of each stack segment
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                if not np.isnan(std) and std > 0:
                    ax.errorbar(bar.get_x() + bar.get_width()/2, 
                               bottom[j] + mean, 
                               yerr=std, fmt='', color='black', 
                               capsize=3, alpha=0.7)
            
            bottom += means
        
        # Customize the stacked chart
        ax.set_title(f"Experiment: {experiment} - Time Range: {time_start}-{time_end}s (Stacked)", fontsize=16)
        ax.set_xlabel("Analysis Type", fontsize=12)
        ax.set_ylabel("Integrated Area (a.u.)", fontsize=12)
        ax.legend(title='Water Structure', fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Save the stacked plot
        plot_name_stacked = f"{experiment}_time_{time_start}_{time_end}_stacked_bar_chart_with_errors"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{plot_name_stacked}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{plot_name_stacked}.svg"), bbox_inches='tight')
        plt.close()

def main():
    # Configuration
    base_search_path = "/Users/danielsinausia/Documents/Experiments"
    target_filename = "combined_cycles_percentage_data.csv"
    target_folder = "Integrations Water Normalized Method"
    output_directory = "/Users/danielsinausia/Documents/Experiments/bar charts"
    
    # Time range settings
    time_start = 160
    time_end = 180
    
    # Output filenames
    output_filename = f"extracted_time_{time_start}_{time_end}_data.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Searching for '{target_filename}' in folders named '{target_folder}'...")
    print(f"Base search path: {base_search_path}")
    print(f"Time range: {time_start}-{time_end} seconds")
    
    # Find all matching CSV files
    csv_files = find_csv_files(base_search_path, target_filename, target_folder)
    
    if not csv_files:
        print("No matching CSV files found!")
        return
    
    print(f"\nFound {len(csv_files)} matching files. Processing...")
    
    # Extract data from each file
    all_data = []
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        data = extract_time_range_data(csv_file, time_start, time_end)
        
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No data could be extracted from any files!")
        return
    
    # Create DataFrame
    result_df = pd.DataFrame(all_data)
    
    # Add analysis type and experiment labels
    result_df['Short Label'] = result_df['Source_Path'].apply(map_file_label)
    result_df['Experiment'] = result_df['Source_Path'].apply(extract_experiment_name)
    
    # Filter out rows where mapping returned None
    result_df = result_df[result_df['Short Label'].notna()]
    
    # Remove duplicates if any
    result_df = result_df.drop_duplicates(subset=['Experiment', 'Short Label'], keep='first')
    
    # Save the extracted data
    output_path = os.path.join(output_directory, output_filename)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total rows extracted: {len(result_df)}")
    
    # Create bar charts with error bars
    print("\nCreating bar charts with error bars...")
    create_bar_charts_with_error_bars(result_df, output_directory, time_start, time_end)
    
    print(f"\nBar charts with error bars saved in {output_directory}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total files processed: {len(result_df)}")
    experiments = result_df['Experiment'].unique()
    print(f"Experiments found: {len(experiments)}")
    print("Experiments:", list(experiments))
    print("\nData distribution by category:")
    print(result_df['Short Label'].value_counts())
    
    # Print some statistics
    peak_columns = ['Mean 3680', 'Mean 3520', 'Mean 3360', 'Mean 3210', 'Mean 3100', 'Mean 2870']
    print(f"\nSample statistics for time range {time_start}-{time_end}s:")
    for peak in peak_columns:
        mean_col = f'{peak}_mean'
        std_col = f'{peak}_std'
        if mean_col in result_df.columns:
            mean_values = result_df[mean_col].dropna()
            std_values = result_df[std_col].dropna()
            print(f"{peak}: Mean={mean_values.mean():.2f} ± {mean_values.std():.2f}, "
                  f"Avg StdDev={std_values.mean():.2f}")

if __name__ == "__main__":
    main()
