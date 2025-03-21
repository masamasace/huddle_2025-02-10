from data_loader import DataLoader
from cv374 import T3WHandler, Win32Handler, DBLHandler, WINHandler  

# data = DataLoader("01_data")

temp_dbl_data = DBLHandler(r"01_data\01_群馬高専\Trillium\200327_20250210111729.dbl")
temp_win_data = WINHandler(r"01_data/01_群馬高専/D013/25021011.36")