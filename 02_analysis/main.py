from cv374 import DataLoader, DataCombiner

data_dir = r"01_data"
# data_dir = r'01_data\01_群馬高専\D013'
data = DataLoader(data_dir)
combined_data = DataCombiner(data)

