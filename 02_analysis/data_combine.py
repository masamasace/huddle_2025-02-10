from cv374 import DataLoader, DataCombiner
from main import get_config

# Get data directory from configuration
data_dir = get_config('paths', 'data_dir')
data = DataLoader(data_dir)
combined_data = DataCombiner(data)

