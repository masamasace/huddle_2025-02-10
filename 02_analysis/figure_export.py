from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def load_csv_with_metadata(file_path):
    file_path = Path(file_path)
    
    # Read CSV data without setting index
    data = pd.read_csv(file_path, header=0, dtype=np.float64)
    
    # Get metadata from JSON file
    metadata_files = list(file_path.parent.glob('*_metadata.json'))
    if not metadata_files:
        print(f"Warning: No metadata file found for {file_path}")
        return data, None
    
    metadata_path = metadata_files[0]
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return data, metadata

def apply_calibration(data, metadata, file_stem):
    
    if metadata and file_stem in metadata:
        calib_coeff = metadata[file_stem]['calibration_coefficient']
        
        # Apply calibration to each channel (excluding time_s column)
        for col in data.columns:
            if col == 'time_s':
                continue
            else:
                data[col] = data[col] * calib_coeff
          
    return data

def process_directory(dir_path):
    dir_path = Path(dir_path)
    
    # Find all CSV files with "combined_" in the name
    combined_files = list(dir_path.glob('**/combined_*.csv'))
    
    if not combined_files:
        print(f"No combined CSV files found in {dir_path}")
        return
    
    # set output directory
    output_dir = dir_path / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    all_data = []
    max_abs_value = 0
    
    # Process each file
    for file_path in combined_files:
        
        # make file name
        file_name = file_path.name
        file_stem = file_path.stem
        
        print(f"Processing {file_name}")
        data, metadata = load_csv_with_metadata(file_path)
        
        # Apply calibration
        if metadata:
            
            data = apply_calibration(data, metadata, file_stem)
        
        # Find max absolute value (excluding time_s column)
        value_columns = [col for col in data.columns if col != 'time_s']
        file_max = data[value_columns].abs().max().max()
        max_abs_value = max(max_abs_value, file_max)
        
        # Store data with file info
        all_data.append({
            'file': file_name,
            'machine_name': file_path.parents[2].name,
            'data': data
        })
    
    # Plot all channels from all files
    plot_all_channels(all_data, max_abs_value, output_dir)

def plot_all_channels(all_data, max_value, output_dir):
    # Count channels excluding time_s
    total_channels = sum(len([col for col in data_dict['data'].columns if col != 'time_s']) 
                         for data_dict in all_data)
    
    fig, axes = plt.subplots(total_channels, 1, figsize=(12, total_channels * 0.75))
    fig.subplots_adjust(hspace=0)
    
    # Ensure axes is always an array, even with a single subplot
    if total_channels == 1:
        axes = [axes]
    
    current_plot = 0
    
    for data_dict in all_data:
        data = data_dict['data']
        file_name = data_dict['file']
        machine_name = data_dict['machine_name']
        
        for col in data.columns:
            if col == 'time_s':
                continue
                
            ax = axes[current_plot]
            ax.plot(data['time_s'], data[col])
            ax.set_ylim(-max_value*1.1, max_value*1.1)
            ax.set_ylabel(f"{machine_name}\n{col}", fontsize=6)
            
            # Only show x-axis label on bottom subplot
            if current_plot < total_channels - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time')
                
            current_plot += 1
    
    output_path = output_dir / 'combined_channels_plot.svg'
    fig.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Example usage with directory path
    dir_path = Path(r'01_data\01_群馬高専')
    process_directory(dir_path)
