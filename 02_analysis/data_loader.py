from cv374 import T3WHandler, Win32Handler, DBLHandler
from pathlib import Path
import pandas as pd
import re

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
        
        print(len(self.files), "files found.")
        
    
    def _load_data(self):
        
        # load data
        self.data = []
        
        for i, row in self.files.iterrows():
            
            print("Loading", row['file_path'])
            
            if row['file_type'] == 'ts':
                self.data.append(T3WHandler(row['file_path']))
            elif row['file_type'] == 'hk':
                self.data.append(Win32Handler(row['file_path']))
            elif row['file_type'] == 'na':
                self.data.append(DBLHandler(row['file_path']))
    