from cv374 import T3WHandler, DBLHandler, WINHandler
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np
import json
from typing import Union

class DataLoader:
    
    def __init__(self, dir_path: Union[str, Path]):
    
        self.dir_path = Path(dir_path)
        
        # make file list
        self._make_file_list()
        
        # load data
        self._load_data()
    
    # to list all files in the directory
    def _make_file_list(self):
        
        # ts means Tokyo Sokushin co., ltd.
        ts_files = list(self.dir_path.glob('**/*.t3w'))
        
        # haku means Hakusan corporation
        hk_files = list(self.dir_path.glob('**/*'))
        hk_files = [x for x in hk_files if re.search(r'\.\d{2}$', str(x))]
        
        # na means Nanometrics
        na_files = list(self.dir_path.glob('**/*.dbl'))
        
        # convert to pandas DataFrame
        self.files = pd.DataFrame({'file_path': ts_files, 'file_type': 'ts'})
        self.files = pd.concat([self.files, pd.DataFrame({'file_path': hk_files, 'file_type': 'hk'})])
        self.files = pd.concat([self.files, pd.DataFrame({'file_path': na_files, 'file_type': 'na'})])
        
        self.files["file_stem"] = self.files["file_path"].apply(lambda x: x.stem)
        
        # reset index
        self.files = self.files.reset_index(drop=True)
        
        print(len(self.files), "files found.")
        
    
    def _load_data(self):
        
        # load data
        self.data = []
        
        for i, row in self.files.iterrows():
            
            print("\rLoading data... ", '{:<6.2f}%'.format((i+1)/len(self.files)*100), end="")
            
            if row['file_type'] == 'ts':
                self.data.append(T3WHandler(row['file_path']))
            elif row['file_type'] == 'hk':
                self.data.append(WINHandler(row['file_path']))
            elif row['file_type'] == 'na':
                self.data.append(DBLHandler(row['file_path']))
        
        print("\nData loaded.")

class DataCombiner:
    
    def __init__(self, data_loader: DataLoader):
        """Initialize with a DataLoader instance"""
        self.data_loader = data_loader
        self.combined_data = {}
        
        self.combine_by_folder()
        # Use DataSaver instead of built-in save method
        data_saver = DataSaver(self.data_loader, self.combined_data)
        data_saver.save_all_data()
    
            
    def combine_by_folder(self):
        """Combine files from the same folder based on start/end datetimes"""
        from datetime import datetime, timedelta
        from collections import defaultdict
        import numpy as np
        
        # Group files by their parent directory
        self.data_loader.files['folder'] = self.data_loader.files['file_path'].apply(lambda x: x.parent)
        self.data_loader.files['combined_index'] = -1
        folder_groups = self.data_loader.files.groupby('folder')
        
        # Process each folder
        for folder, group_df in folder_groups:
            folder_name = folder.name
            print(f"Combining files in folder: {folder_name}")
            
            # Sort files by start datetime within each folder
            group_df = group_df.copy()
            group_df['start_time'] = group_df.apply(
                lambda row: self.data_loader.data[row.name].get_header()['start_datetime'], 
                axis=1
            )
            group_df = group_df.sort_values('start_time')
            
            # Get indices of files in this folder (now sorted by start time)
            file_indices = group_df.index.tolist()
            
            # Verify all files have the same number of channels
            channel_counts = [len(self.data_loader.data[idx].get_stream()) for idx in file_indices]
            if len(set(channel_counts)) > 1:
                print(f"Warning: Files in folder {folder_name} have different channel counts: {channel_counts}")
                print("Will attempt to combine them properly by channel name")
            
            combined_index = 0
            
            # Process each handler in the folder
            for i in range(len(file_indices)):
                if i >= 1:
                    current_handler = self.data_loader.data[file_indices[i]]
                    prev_handler = self.data_loader.data[file_indices[i-1]]
                    
                    # change str to datetime
                    if isinstance(current_handler.get_header()['start_datetime'], str):
                        cur_start_time = datetime.fromisoformat(current_handler.get_header()['start_datetime'].replace('Z', '+00:00'))
                    else:
                        cur_start_time = current_handler.get_header()['start_datetime']
                    if isinstance(prev_handler.get_header()['end_datetime'], str):
                        prev_end_time = datetime.fromisoformat(prev_handler.get_header()['end_datetime'].replace('Z', '+00:00'))
                    else:
                        prev_end_time = prev_handler.get_header()['end_datetime']
                    
                    if cur_start_time < prev_end_time:
                        print("Overlap detected:")
                        print("  Previous file ends:", prev_handler.get_header()['end_datetime'])
                        print("  Current file starts:", current_handler.get_header()['start_datetime'])
                    
                    elif cur_start_time - prev_end_time > timedelta(seconds=0.01):
                        combined_index += 1
                
                self.data_loader.files.loc[file_indices[i], 'combined_index'] = combined_index
            
            # Get unique combined indices for this folder
            unique_indices = self.data_loader.files.loc[file_indices, 'combined_index'].unique()
            
            # Combine data by index
            combined_folder_data = []
            
            for idx in unique_indices:
                # Get all files with this combined index
                idx_files = self.data_loader.files.loc[
                    (self.data_loader.files['folder'] == folder) & 
                    (self.data_loader.files['combined_index'] == idx)
                ]
                
                # Get the handler indices for these files
                handler_indices = idx_files.index.tolist()
                
                # Create a list of file paths for metadata
                file_paths = [str(path) for path in idx_files['file_path'].tolist()]
                
                # Gather all channels from all files to ensure we handle all unique channels
                all_channels = set()
                for hidx in handler_indices:
                    stream = self.data_loader.data[hidx].get_stream()
                    all_channels.update(tr.stats.channel for tr in stream)
                
                print(f"Found {len(all_channels)} unique channels: {all_channels}")
                
                # Create a dictionary to hold data for each channel
                channel_data = {channel: [] for channel in all_channels}
                channel_times = {channel: [] for channel in all_channels}
                
                # First, collect all data by channel
                for hidx in handler_indices:
                    handler = self.data_loader.data[hidx]
                    stream = handler.get_stream()
                    start_time = handler.get_header()['start_datetime']
                    
                    for trace in stream:
                        channel = trace.stats.channel
                        # Store the data and start time for this channel segment
                        channel_data[channel].append(trace.data)
                        channel_times[channel].append((start_time, trace.stats.npts, trace.stats.sampling_rate))
                
                # Now create a new combined stream with one trace per channel
                from obspy import Stream, Trace
                combined_stream = Stream()
                
                for channel in all_channels:
                    segments = channel_data[channel]
                    time_info = channel_times[channel]
                    
                    # Sort segments by start time
                    sorted_segments = sorted(zip(segments, time_info), key=lambda x: x[1][0])
                    
                    # Merge the data, handling overlaps
                    merged_data = []
                    prev_end_time = None
                    
                    for segment, (start_time, npts, sampling_rate) in sorted_segments:
                        # Check if start_time is a string and convert if needed
                        if isinstance(start_time, str):
                            try:
                                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            except ValueError:
                                for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', 
                                           '%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                                    try:
                                        start_time = datetime.strptime(start_time, fmt)
                                        break
                                    except ValueError:
                                        continue
                        
                        # Calculate end time
                        try:
                            end_time = start_time + timedelta(seconds=npts/sampling_rate)
                        except (TypeError, AttributeError) as e:
                            print(f"Error calculating end time: {e}")
                            print(f"start_time type: {type(start_time)}, value: {start_time}")
                            print(f"npts: {npts}, sampling_rate: {sampling_rate}")
                            # Provide a fallback - just use the data as is
                            if prev_end_time is None:
                                merged_data.extend(segment)
                            else:
                                merged_data.extend(segment)
                            continue
                        
                        if prev_end_time is None:
                            # First segment - add all data
                            merged_data.extend(segment)
                        else:
                            # Check for overlap
                            try:
                                if prev_end_time > start_time:
                                    # Calculate overlap
                                    overlap_seconds = (prev_end_time - start_time).total_seconds()
                                    overlap_samples = int(overlap_seconds * sampling_rate)
                                    
                                    # Skip overlapping samples
                                    if overlap_samples < len(segment):
                                        merged_data.extend(segment[overlap_samples:])
                                    else:
                                        print(f"Warning: Segment for channel {channel} is entirely within overlap. Skipping.")
                                else:
                                    # No overlap - add gap if needed
                                    gap_seconds = (start_time - prev_end_time).total_seconds()
                                    if gap_seconds > 0:
                                        print(f"Gap detected for channel {channel}: {gap_seconds:.2f} seconds")
                                        # Fill with zeros
                                        gap_samples = int(gap_seconds * sampling_rate)
                                        merged_data.extend([0] * gap_samples)
                                        
                                    # Add current segment
                                    merged_data.extend(segment)
                            except (TypeError, AttributeError) as e:
                                print(f"Error comparing times: {e}")
                                print(f"prev_end_time type: {type(prev_end_time)}, value: {prev_end_time}")
                                print(f"start_time type: {type(start_time)}, value: {start_time}")
                                # Just append the data
                                merged_data.extend(segment)
                        
                        prev_end_time = end_time
                    
                    # Create a new trace with the merged data
                    stats = {'network': '', 'station': '', 'location': '', 'channel': channel, 
                             'sampling_rate': sampling_rate, 'npts': len(merged_data)}
                    trace = Trace(data=np.array(merged_data), header=stats)
                    combined_stream.append(trace)
                
                # Add combined data to the list
                combined_folder_data.append({
                    'combined_index': idx,
                    'files': file_paths,
                    'stream': combined_stream
                })
            
            # Store combined data for this folder
            self.combined_data[str(folder)] = combined_folder_data

class DataSaver:
    
    def __init__(self, data_loader: DataLoader, combined_data: dict):
        """Initialize with DataLoader and combined_data instances"""
        self.data_loader = data_loader
        self.combined_data = combined_data
        
    def save_all_data(self):
        """Save both combined and original data"""
        self.save_combined_data()
        self.save_original_data()
        
    def save_combined_data(self):
        """Save the combined data to proc/combined subdirectory"""
        import json
        import numpy as np
        import os
        
        # Check if combined_data is a dictionary
        if not isinstance(self.combined_data, dict):
            print(f"Warning: combined_data is not a dictionary. Type: {type(self.combined_data)}")
            return
            
        # Iterate through the combined data dictionary
        for folder_name, folder_data in self.combined_data.items():
            # Use the original folder path
            folder_output = Path(folder_name)
            
            # Create a subdirectory for the combined results
            combined_dir = folder_output / "proc" / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata as JSON
            metadata = {}
            
            for combined_item in folder_data:
                # Create a unique identifier for this combined data - avoid using folder name which may contain Japanese
                combined_id = f"combined_{os.path.basename(folder_output)}_{combined_item['combined_index']}"
                
                # Gather file metadata
                file_metadata = []
                for file_path in combined_item['files']:
                    # Find the index in the original files DataFrame
                    try:
                        file_idx = self.data_loader.files[self.data_loader.files['file_path'].astype(str) == file_path].index[0]
                        handler = self.data_loader.data[file_idx]
                        header = handler.get_header()
                        
                        # Get stream to extract sampling rate, samples, and channels info
                        stream = handler.get_stream()
                        
                        # Extract calibration coefficient from header (single value for the entire stream)
                        calib_coeff = None
                        if 'calib_coeff' in header:
                            calib_coeff = header['calib_coeff']
                        
                        file_metadata.append({
                            'path': file_path,
                            'start_datetime': str(header['start_datetime']),
                            'end_datetime': str(header['end_datetime']),
                            'sampling_rate': float(stream[0].stats.sampling_rate),
                            'n_samples': int(stream[0].stats.npts),
                            'n_channels': len(stream),
                            'calibration_coefficient': calib_coeff
                        })
                    except (IndexError, KeyError) as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue
                
                # Get combined stream info
                combined_stream = combined_item['stream']
                n_channels = len(set(tr.stats.channel for tr in combined_stream))
                
                # Use the calibration coefficient from the first file for the combined data
                combined_calib_coeff = None
                for file_meta in file_metadata:
                    if 'calibration_coefficient' in file_meta and file_meta['calibration_coefficient'] is not None:
                        combined_calib_coeff = file_meta['calibration_coefficient']
                        break
                
                # Save metadata
                metadata[combined_id] = {
                    'files': file_metadata,
                    'combined_index': str(combined_item['combined_index']),
                    'start_datetime': str(file_metadata[0]['start_datetime']) if file_metadata else "",  # Use first file's start time
                    'end_datetime': str(file_metadata[-1]['end_datetime']) if file_metadata else "",     # Use last file's end time
                    'sampling_rate': str(combined_stream[0].stats.sampling_rate),
                    'n_channels': str(n_channels),
                    'total_samples': str(sum(tr.stats.npts for tr in combined_stream)),
                    'calibration_coefficient': combined_calib_coeff
                }
                
                # Save stream data as CSV with epoch time
                stream = combined_item['stream']
                csv_file = combined_dir / f"{combined_id}.csv"
                
                # Convert stream to DataFrame
                stream_data = {}
                
                # Add trace data first
                for trace in stream:
                    stream_data[trace.stats.channel] = trace.data
                
                # Find the maximum length of all data arrays
                max_length = max(len(data) for data in stream_data.values())
                
                # Pad shorter arrays with NaN values
                for channel, data in stream_data.items():
                    if len(data) < max_length:
                        print(f"Padding channel {channel} data from {len(data)} to {max_length} samples")
                        padding = [np.nan] * (max_length - len(data))
                        stream_data[channel] = np.append(data, padding)
                
                # Now create epoch time array
                sampling_rate = stream[0].stats.sampling_rate
                
                # Parse the start datetime from metadata and convert to epoch
                start_datetime_str = metadata[combined_id]['start_datetime']
                
                # Handle different datetime formats
                if start_datetime_str:
                    try:
                        # Try ISO format first
                        if 'Z' in start_datetime_str:
                            start_datetime = datetime.fromisoformat(start_datetime_str.replace('Z', '+00:00'))
                        else:
                            start_datetime = datetime.fromisoformat(start_datetime_str)
                    except ValueError:
                        # Try other common formats
                        for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                            try:
                                start_datetime = datetime.strptime(start_datetime_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            # If all formats fail, use a default value and warn
                            print(f"Warning: Could not parse start datetime: {start_datetime_str}")
                            # Use Unix epoch start as fallback
                            start_datetime = datetime(1970, 1, 1)
                else:
                    # No start datetime available
                    print("Warning: No start datetime available. Using Unix epoch start.")
                    start_datetime = datetime(1970, 1, 1)
                
                # Convert to epoch seconds (timestamp)
                start_epoch = start_datetime.timestamp()
                
                # Create array of epoch seconds
                time_array = start_epoch + np.arange(0, max_length) / sampling_rate
                stream_data['time_s'] = time_array
                
                # reorder columns (time, N, E, Z) or (time, 0, 1, 2)
                # otherwise, time should be moved to the first column
                if 'N' in stream_data:
                    stream_data = {k: stream_data[k] for k in ['time_s', 'N', 'E', 'Z'] if k in stream_data}
                elif '0' in stream_data:
                    stream_data = {k: stream_data[k] for k in ['time_s', '0', '1', '2'] if k in stream_data}
                else:
                    key_list = list(stream_data.keys())
                    if 'time_s' in key_list:
                        key_list.remove('time_s')
                    key_list.insert(0, 'time_s')
                    stream_data = {k: stream_data[k] for k in key_list}
                
                pd.DataFrame(stream_data).to_csv(csv_file, index=False)
                print(f"Saved combined stream data to {csv_file}")
            
            # Save metadata - ensure proper encoding for Japanese characters
            if metadata:  # Only save if there's any metadata
                json_file = combined_dir / "combined_metadata.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"Saved combined metadata to {json_file}")
            else:
                print(f"No metadata to save for {folder_name}")
    
    def save_original_data(self):
        """Save the original data to proc/original subdirectory"""
        import json
        import numpy as np
        import os
        
        # Group files by folder (parent directory)
        if 'folder' not in self.data_loader.files.columns:
            self.data_loader.files['folder'] = self.data_loader.files['file_path'].apply(lambda x: x.parent)
        
        folder_groups = self.data_loader.files.groupby('folder')
        
        for folder, group_df in folder_groups:
            folder_name = folder.name
            print(f"Saving original files from folder: {folder_name}")
            
            # Create directory for original data
            original_dir = Path(folder) / "proc" / "original"
            original_dir.mkdir(parents=True, exist_ok=True)
            
            # Get indices of files in this folder
            file_indices = group_df.index.tolist()
            
            # Process each handler in the folder
            for i, idx in enumerate(file_indices):
                try:
                    handler = self.data_loader.data[idx]
                    file_path = self.data_loader.files.loc[idx, 'file_path']
                    file_name = file_path.name.replace('.', '')
                    
                    # Create unique ID for this file
                    file_id = f"original_{os.path.basename(folder)}_{file_name}"
                    
                    # Get header information
                    header = handler.get_header()
                    stream = handler.get_stream()
                    
                    # Extract calibration coefficient
                    calib_coeff = None
                    if 'calib_coeff' in header:
                        calib_coeff = header['calib_coeff']
                    
                    # Convert stream to DataFrame with epoch time
                    stream_data = {}
                    
                    # Add trace data
                    for trace in stream:
                        stream_data[trace.stats.channel] = trace.data
                    
                    # Skip if no data available
                    if not stream_data:
                        print(f"\rSkipping file {file_path} - no stream data", end="")
                        continue
                    
                    # Find the maximum length of all data arrays
                    max_length = max(len(data) for data in stream_data.values())
                    
                    # Pad shorter arrays with NaN values if needed
                    for channel, data in stream_data.items():
                        if len(data) < max_length:
                            print(f"Padding channel {channel} data from {len(data)} to {max_length} samples")
                            padding = [np.nan] * (max_length - len(data))
                            stream_data[channel] = np.append(data, padding)
                    
                    # Create epoch time array
                    sampling_rate = stream[0].stats.sampling_rate
                    
                    # Convert start datetime to epoch seconds
                    start_datetime = header['start_datetime']
                    if isinstance(start_datetime, str):
                        try:
                            if 'Z' in start_datetime:
                                start_datetime = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
                            else:
                                start_datetime = datetime.fromisoformat(start_datetime)
                        except ValueError:
                            for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', 
                                       '%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                                try:
                                    start_datetime = datetime.strptime(start_datetime, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                print(f"Warning: Could not parse start datetime: {start_datetime}")
                                start_datetime = datetime(1970, 1, 1)
                    
                    # Convert to epoch seconds
                    try:
                        start_epoch = start_datetime.timestamp()
                    except (AttributeError, TypeError) as e:
                        print(f"Error converting to timestamp: {e}")
                        print(f"Using Unix epoch start as fallback")
                        start_epoch = 0
                    
                    # Add time column to data
                    time_array = start_epoch + np.arange(0, max_length) / sampling_rate
                    stream_data['time_s'] = time_array
                    
                    # Reorder columns to put time first
                    if 'N' in stream_data:
                        stream_data = {k: stream_data[k] for k in ['time_s', 'N', 'E', 'Z'] if k in stream_data}
                    elif '0' in stream_data:
                        stream_data = {k: stream_data[k] for k in ['time_s', '0', '1', '2'] if k in stream_data}
                    else:
                        key_list = list(stream_data.keys())
                        if 'time_s' in key_list:
                            key_list.remove('time_s')
                        key_list.insert(0, 'time_s')
                        stream_data = {k: stream_data[k] for k in key_list}
                    
                    # Save data to CSV
                    csv_file = original_dir / f"{file_id}.csv"
                    pd.DataFrame(stream_data).to_csv(csv_file, index=False)
                    
                    # Create metadata for this specific file
                    metadata = {
                        'path': str(file_path),
                        'start_datetime': str(header['start_datetime']),
                        'end_datetime': str(header['end_datetime']),
                        'sampling_rate': str(stream[0].stats.sampling_rate),
                        'n_channels': str(len(stream)),
                        'n_samples': str(stream[0].stats.npts),
                        'calibration_coefficient': calib_coeff
                    }
                    
                    # Save metadata for this specific file
                    json_file = original_dir / f"{file_id}_metadata.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    print(f"\rProcessing file {i+1}/{len(file_indices)}: {file_id} saved", end="")
                    
                except Exception as e:
                    print(f"\nError processing file at index {idx}: {e}")
                    continue
            
            print(f"\nCompleted processing {len(file_indices)} files from folder {folder_name}")
class GraphExporter():
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
        self._make_file_list()
        
        self._update_csv()
    
    def _make_file_list(self):
        
        # find all csv files in the directory
        csv_files = list(self.data_dir.glob('**/combined*.csv'))
        
        # exclude updated csv files
        csv_files = [x for x in csv_files if not re.search(r'_updated\.csv$', str(x))]
        
        # convert to pandas DataFrame
        self.files = pd.DataFrame({'file_path': csv_files})
        
        # reset index
        self.files = self.files.reset_index(drop=True)
        
        # get calibration coefficient
        self.files['calib_coeff'] = self.files['file_path'].apply(lambda x: self._get_calib_coeff(x))
        
    
    def _get_calib_coeff(self, file_path):
        
        file_dir = file_path.parent
        
        # find metadata.json
        metadata_file = file_dir / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # find the calibration coefficient
            for key, value in metadata.items():
                return value['calibration_coefficient']
    
    def _update_csv(self):
        
        for i, row in self.files.iterrows():
            
            print("\rUpdating csv... ", '{:<6.2f}%'.format((i+1)/len(self.files)*100), end="")
            
            file_path = row['file_path']
            calib_coeff = row['calib_coeff']
            
            # load csv file
            data = pd.read_csv(file_path).astype('float64')
            
            # update data
            data.iloc[:, 1:] = data.iloc[:, 1:] * calib_coeff
            
            # save updated data
            updated_file_path = file_path.parent / (file_path.stem + "_updated.csv")
            data.to_csv(updated_file_path, index=False)
            
        print("\nCSV updated.")

