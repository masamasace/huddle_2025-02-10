from data_handler import DataLoader, DataCombiner, DataSaver, GraphExporter
from cv374 import T3WHandler, WIN32Handler, DBLHandler, WINHandler  

data_dir = r"01_data"
data_dir = r'01_data\01_群馬高専\D013'
data = DataLoader(data_dir)
combined_data = DataCombiner(data)
data_saver = DataSaver(data, combined_data)
data_saver.save_all_data()

