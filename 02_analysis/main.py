from data_handler import DataLoader, DataCombiner
from cv374 import T3WHandler, WIN32Handler, DBLHandler, WINHandler  

data_dir = r"01_data"
data_dir = r'01_data\01_群馬高専\Trillium'
data = DataLoader(data_dir)
combined_data = DataCombiner(data)