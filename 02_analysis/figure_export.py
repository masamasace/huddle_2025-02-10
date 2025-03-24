from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.signal.windows import parzen, hann
from scipy.signal import convolve, detrend
from main import get_config, get_file_list

class SeismicDataProcessor:
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or ['VSE-NS', 'VSE-EW', 'CH6']
        
    def load_csv_with_metadata(self, file_path):
        file_path = Path(file_path)
        
        # Read CSV data without setting index
        data = pd.read_csv(file_path, header=0, dtype=np.float64)
        
        # Get metadata from JSON file
        metadata_files = list(file_path.parent.glob('*_metadata.json'))
        if not metadata_files:
            print(f"Warning: No metadata file found for {file_path}")
            return data, None

        # drop columns if specified
        if self.drop_cols and data.columns.isin(self.drop_cols).any():
            data = data.drop(columns=self.drop_cols)
        
        metadata_path = metadata_files[0]
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return data, metadata

    def apply_calibration(self, data, metadata, file_stem):
        if metadata and file_stem in metadata:
            calib_coeff = metadata[file_stem]['calibration_coefficient']
            
            # Apply calibration to each channel (excluding time_s column)
            for col in data.columns:
                if col == 'time_s':
                    continue
                else:
                    data[col] = data[col] * calib_coeff
              
        return data
    
    def get_files_to_process(self, input_path):
        combined_files = []
        
        # Handle different input types
        if isinstance(input_path, list):
            # Input is a list of files
            for path in input_path:
                path = Path(path)
                if path.is_file() and path.name.startswith('combined_') and path.suffix == '.csv':
                    combined_files.append(path)
        else:
            # Convert to Path object if it's a string
            input_path = Path(input_path)
            
            if input_path.is_file():
                # Input is a single file
                if input_path.name.startswith('combined_') and input_path.suffix == '.csv':
                    combined_files.append(input_path)
            else:
                # Input is a directory
                combined_files = list(input_path.glob('**/combined_*.csv'))
        
        return combined_files, input_path
    
    def get_output_directory(self, combined_files, input_path):
        # Set output directory - use parent directory of the first file if input is a file or list
        if isinstance(input_path, list) or (isinstance(input_path, Path) and input_path.is_file()):
            output_dir = combined_files[0].parents[3] / 'figures'
        else:
            output_dir = input_path / 'figures'
        
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def change_column_names(self, data):
        
        rename_dict = {
            "NS": ["NS", "N-S", "N/S", "TC-NS", "CH0", "N"],
            "EW": ["EW", "E-W", "E/W", "TC-EW", "CH1", "E"],
            "UD": ["UD", "U-D", "U/D", "TC-UD", "CH2", "Z"],
        }
        
        for new_name, old_names in rename_dict.items():
            for old_name in old_names:
                if old_name in data.columns:
                    data.rename(columns={old_name: new_name}, inplace=True)
                    break
        
        # reorder columns
        data = data[['time_s', 'NS', 'EW', 'UD']]
        
        return data
    
    def process_file(self, file_path, x_min=None, x_max=None):
        # make file name
        file_name = file_path.name
        file_stem = file_path.stem
        
        print(f"Processing {file_name}")
        data, metadata = self.load_csv_with_metadata(file_path)
        
        # change column names
        data = self.change_column_names(data)
        
        # Extract unit from metadata if available
        unit = "unknown"
        if metadata and file_stem in metadata and "unit" in metadata[file_stem]:
            unit = metadata[file_stem]["unit"]
        
        # Apply calibration
        if metadata:
            data = self.apply_calibration(data, metadata, file_stem)

        value_columns = [col for col in data.columns if col != 'time_s']
        
        if unit == 'kine':
            # convert kine (cm/s) to cm/s2
            dt = data['time_s'].diff().mean()
            data[value_columns] = data[value_columns].diff() / dt
            unit = 'cm/s2'
        
        elif unit == 'g':
            # convert g to cm/s2
            data[value_columns] = data[value_columns] * 9.81 * 100
            unit = 'cm/s2'
        
        # Trim data based on x_min and x_max if provided
        if x_min is not None or x_max is not None:
            mask = pd.Series(True, index=data.index)
            if x_min is not None:
                mask = mask & (data['time_s'] >= x_min)
            if x_max is not None:
                mask = mask & (data['time_s'] <= x_max)
            data = data[mask]
        
        # Find max absolute value (excluding time_s column)
        file_max = data[value_columns].abs().max().max()
        
        # baseline correction
        baseline_offset = data[value_columns].mean()
        data[value_columns] = data[value_columns] - baseline_offset
        
        # calculate standard deviation
        std = data[value_columns].std()
        
        return {
            'file': file_name,
            'machine_name': file_path.parents[2].name,
            'data': data,
            'unit': unit,
            'std': std,
            'baseline_offset': baseline_offset,
            'max_value': file_max
        }
    
    def process_records(self, input_path, y_min=None, y_max=None, x_min=None, x_max=None):
        """
        Process seismic records from files or directories.
        
        Args:
            input_path: Can be a directory path, a file path, or a list of file paths
            y_min, y_max: Optional y-axis limits
            x_min, x_max: Optional x-axis limits
        """
        combined_files, input_path = self.get_files_to_process(input_path)
        
        if not combined_files:
            print(f"No combined CSV files found in the input")
            return
        
        output_dir = self.get_output_directory(combined_files, input_path)
        
        all_data = []
        max_abs_value = 0
        
        # Process each file
        for file_path in combined_files:
            data_dict = self.process_file(file_path, x_min, x_max)
            all_data.append(data_dict)
            max_abs_value = max(max_abs_value, data_dict['max_value'])
        
        # Create plotter and plot all channels from all files
        plotter = SeismicPlotter()
        plotter.plot_time_series(all_data, max_abs_value, output_dir, y_min, y_max, x_min, x_max)
        
        # Plot power spectrum
        plotter.plot_power_spectrum(all_data, output_dir, use_parzen=True, parzen_width=0.2, 
                                  f_min=0.01, f_max=50, amp_min=10 ** -19, amp_max=10 ** -8)
        
        return all_data

class SeismicPlotter:
    def __init__(self):
        self.color_dict = {
            'NS': 'red',
            'EW': 'green',
            'UD': 'blue'
        }
        
    def plot_time_series(self, all_data, max_value, output_dir, y_min=None, y_max=None, x_min=None, x_max=None):
        color_dict = {
            'NS': 'red',
            'EW': 'green',
            'UD': 'blue'
        }
        # Count channels excluding time_s
        total_channels = sum(len([col for col in data_dict['data'].columns if col != 'time_s']) 
                            for data_dict in all_data)
        
        fig, axes = plt.subplots(total_channels, 1, figsize=(12, total_channels * 0.75))
        fig.subplots_adjust(hspace=0)
        
        # Ensure axes is always an array, even with a single subplot
        if total_channels == 1:
            axes = [axes]
        
        current_plot = 0
        
        # Set default y limits if not provided
        if y_min is None and y_max is None:
            y_min = -max_value * 1.1
            y_max = max_value * 1.1
        elif y_min is None:
            y_min = -max_value * 1.1
        elif y_max is None:
            y_max = max_value * 1.1
        
        # Get global x limits from the data if not provided
        if x_min is None or x_max is None:
            global_x_min = min(data_dict['data']['time_s'].min() for data_dict in all_data)
            global_x_max = max(data_dict['data']['time_s'].max() for data_dict in all_data)
            
            if x_min is None:
                x_min = global_x_min
            if x_max is None:
                x_max = global_x_max
        
        for data_dict in all_data:
            data = data_dict['data']
            file_name = data_dict['file']
            machine_name = data_dict['machine_name']
            unit = data_dict['unit']
            
            for col in data.columns:
                if col == 'time_s':
                    continue
                    
                ax = axes[current_plot]
                ax.plot(data['time_s'], data[col], color=color_dict[col], linewidth=0.25)
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(x_min, x_max)
                ax.set_ylabel(f"{machine_name}\n{col}", fontsize=8)
                
                # Add unit annotation
                ax.annotate(f"Unit: {unit}", xy=(0.02, 0.90), xycoords='axes fraction', fontsize=8, ha='left', va='top')
                ax.annotate(f"Offset: {data_dict['baseline_offset'][col]:.4e}", xy=(0.1, 0.90), xycoords='axes fraction', fontsize=8, ha='left', va='top')
                ax.annotate(f"Std: {data_dict['std'][col]:.4e}", xy=(0.25, 0.90), xycoords='axes fraction', fontsize=8, ha='left', va='top')
                
                # remove bounding lines of ax
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # keep yaxis label only for first subplot
                if current_plot != 0:
                    ax.set_yticklabels([])
                # Only show x-axis label on bottom subplot
                if current_plot < total_channels - 1:
                    ax.set_xticklabels([])
                    ax.spines['bottom'].set_visible(False)
                else:
                    ax.set_xlabel('Time')
                    
                current_plot += 1
        
        output_path = output_dir / 'time_series.svg'
        fig.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    def apply_parzen_window(self, spectrum, freqs, width=0.2):
        """
        Apply Parzen window smoothing to the power spectrum using scipy's convolution.
        
        Args:
            spectrum: The power spectrum to smooth
            freqs: Frequency array corresponding to the spectrum
            width: Width of the Parzen window in Hz
            
        Returns:
            Smoothed spectrum array
        """
        df = freqs[1] - freqs[0]  # frequency resolution
        
        # Calculate window width in samples
        n_width = int(width / df)
        window_size = 2 * n_width + 1
        
        # Create Parzen window using scipy
        window = parzen(window_size)
        
        # Normalize window
        window = window / np.sum(window)
        
        # Apply convolution for smoothing
        smoothed = convolve(spectrum, window, mode='same')
        
        return smoothed
    
    def calculate_psd(self, time_series, dt):
        """
        Calculate Power Spectral Density from time series data.
        
        Args:
            time_series: Time series data
            dt: Time step in seconds
            
        Returns:
            freqs: Frequency array
            psd: Power Spectral Density array
        """
        # Detrend data
        detrended = detrend(time_series)
        
        # Apply Hann window to reduce spectral leakage
        windowed = detrended * hann(len(detrended))
        
        # Calculate FFT
        n = len(windowed)
        fft = np.fft.rfft(windowed)
        
        # Calculate frequencies
        freqs = np.fft.rfftfreq(n, dt)
        
        # Calculate PSD (normalize by window factor)
        window_factor = 8/3  # for Hann window
        psd = 2.0 * (np.abs(fft) / n)**2 / (1/dt) * window_factor
        
        return freqs, psd
    
    def plot_power_spectrum(self, all_data, output_dir, use_parzen=False, parzen_width=0.2, 
                           f_min=None, f_max=None, amp_min=None, amp_max=None):
        """
        Plot power spectrum for all data files.
        
        Args:
            all_data: List of processed data dictionaries
            output_dir: Directory for output files
            use_parzen: Whether to apply Parzen window smoothing
            parzen_width: Width of Parzen window in Hz
            f_min, f_max: Frequency axis limits
            amp_min, amp_max: Amplitude axis limits
        """
        # Get all unique machine names
        all_machines = [data_dict['machine_name'] for data_dict in all_data]
        
        if not all_machines:
            print("No time series data found to plot")
            return
        
        # Create a single figure with subplots for each machine
        fig, axes = plt.subplots(len(all_machines), 1, figsize=(5, 3*len(all_machines)), sharex=True)
        
        # Ensure axes is always a list, even with a single subplot
        if len(all_machines) == 1:
            axes = [axes]
        
        # Create a dictionary to map machines to their respective subplot
        machine_to_axis = {machine: ax for machine, ax in zip(all_machines, axes)}
        
        # Process each data file
        for data_dict in all_data:
            machine_name = data_dict['machine_name']
            data = data_dict['data']
            dt = data['time_s'].diff().mean()
            unit = data_dict['unit']
            
            # Get the appropriate axis for this machine
            ax = machine_to_axis[machine_name]
            
            # Process each channel
            for col in data.columns:
                if col == 'time_s':
                    continue
                    
                # Calculate PSD
                freqs, psd = self.calculate_psd(data[col].values, dt)
                
                # Apply Parzen window smoothing if requested
                if use_parzen:
                    psd = self.apply_parzen_window(psd, freqs, parzen_width)
                
                # Plot with appropriate color for each component
                color = self.color_dict.get(col, 'black')
                ax.loglog(freqs, psd, linewidth=1, label=col, color=color, alpha=0.7)
        
        # Configure each subplot
        for i, (machine, ax) in enumerate(machine_to_axis.items()):
            # Find the unit from the data_dict with this machine name
            unit = "unknown"
            for data_dict in all_data:
                if data_dict['machine_name'] == machine:
                    unit = data_dict['unit']
                    break
                    
            # Add unit information to labels
            unit_label = f" [{unit}²/Hz]" if unit != "unknown" else ""
            
            # Set title and labels
            ax.set_title(f'{machine}')
            ax.set_ylabel(f'PSD{unit_label}')
            
            # Only show x-label on bottom subplot
            if i == len(machine_to_axis) - 1:
                ax.set_xlabel('Frequency [Hz]')
            
            # Add grid
            ax.grid(True, which="both", ls="-", alpha=0.5, linewidth=0.5)
            
            # Add legend for components
            ax.legend(loc='lower left', fancybox=False, shadow=False, ncol=3, fontsize=6)
            
            # remove right and top bounding lines of ax
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            
            # Set frequency axis limits if provided
            if f_min is not None or f_max is not None:
                xlim = list(ax.get_xlim())
                if f_min is not None:
                    xlim[0] = f_min
                if f_max is not None:
                    xlim[1] = f_max
                ax.set_xlim(xlim)
            
            # Set amplitude axis limits if provided
            if amp_min is not None or amp_max is not None:
                ylim = list(ax.get_ylim())
                if amp_min is not None:
                    ylim[0] = amp_min
                if amp_max is not None:
                    ylim[1] = amp_max
                ax.set_ylim(ylim)
        
        # Add note about Parzen window if used
        if use_parzen:
            fig.text(0.02, 0.02, f"Parzen window applied (width: {parzen_width} Hz)", 
                   fontsize=8, va='bottom', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the combined figure
        output_path = output_dir / 'psd.svg'
        fig.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Power spectra plot saved to {output_path}")

if __name__ == "__main__":
    # Create processor instance
    processor = SeismicDataProcessor()
    
    # Get the file list from the main.py configuration
    file_list = get_file_list()
    
    # Get time limits from configuration
    time_limits = get_config('time_limits')
    x_min = time_limits.get('x_min')
    x_max = time_limits.get('x_max')
    y_min = time_limits.get('y_min')
    y_max = time_limits.get('y_max')
    
    # Process files with configuration from main.py
    processor.process_records(file_list, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
