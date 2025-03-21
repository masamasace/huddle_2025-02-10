from cv374 import T3WHandler, DBLHandler, WINHandler
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np

class DataLoader:
    
    def __init__(self, dir_path):
    
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
    
    def __init__(self, data_loader):
        """Initialize with a DataLoader instance"""
        self.data_loader = data_loader
        self.combined_data = {}
        
        self.combine_by_folder()
        self.save_combined_data()
    
            
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
        
    def save_combined_data(self):
        """Save the combined data to the original folders where the files were located"""
        import json
        import numpy as np
        import os
        
        for folder_name, folder_data in self.combined_data.items():
            # Use the original folder path
            folder_output = Path(folder_name)
            
            # Create a subdirectory for the combined results
            combined_dir = folder_output / "combined_data"
            combined_dir.mkdir(exist_ok=True)
            
            # Save metadata as JSON
            metadata = {}
            
            for combined_item in folder_data:
                # Create a unique identifier for this combined data - avoid using folder name which may contain Japanese
                combined_id = f"combined_{os.path.basename(folder_output)}_{combined_item['combined_index']}"
                
                # Gather file metadata
                file_metadata = []
                for file_path in combined_item['files']:
                    # Find the index in the original files DataFrame
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
                    'start_datetime': str(file_metadata[0]['start_datetime']),  # Use first file's start time
                    'end_datetime': str(file_metadata[-1]['end_datetime']),     # Use last file's end time
                    'sampling_rate': str(combined_stream[0].stats.sampling_rate),
                    'n_channels': str(n_channels),
                    'total_samples': str(sum(tr.stats.npts for tr in combined_stream)),
                    'calibration_coefficient': combined_calib_coeff
                }
                
                # Save stream data as CSV with relative time
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
                
                # Now create and add time array based on the final data length
                sampling_rate = stream[0].stats.sampling_rate
                time_array = np.arange(0, max_length) / sampling_rate
                stream_data['time_s'] = time_array
                
                # reorder columns (time, N, E, Z) or (time, 0, 1, 2)
                # otherwise, time should be moved to the first column
                if 'N' in stream_data:
                    stream_data = {k: stream_data[k] for k in ['time_s', 'N', 'E', 'Z']}
                elif '0' in stream_data:
                    stream_data = {k: stream_data[k] for k in ['time_s', '0', '1', '2']}
                else:
                    key_list = list(stream_data.keys())
                    key_list.remove('time_s')
                    key_list.insert(0, 'time_s')
                    stream_data = {k: stream_data[k] for k in key_list}
                    
                
                pd.DataFrame(stream_data).to_csv(csv_file, index=False)
                print(f"Saved stream data to {csv_file}")
            
            # Save metadata - ensure proper encoding for Japanese characters
            json_file = combined_dir / "metadata.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Saved metadata to {json_file}")

