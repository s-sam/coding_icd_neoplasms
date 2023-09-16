import os
import pandas as pd

import initialise_data_details

class LoadIndividualDataTables():
    def __init__(self, root, subfolder):
        self.root = root
        self.subfolder = subfolder
        self.path = root + subfolder + '/'
        self.contents = os.listdir(self.path)
    
    def check_folder_contains_1_item(self):
        if len(self.contents) != 1:
            raise Exception("There should only be 1 item in the folder.")
        return self

    def check_folder_contains_1_file(self):
        self.contents_path = self.path + self.contents[0]
        if os.path.isfile(self.contents_path) == False:
            raise Exception("The folder should only contain 1 file.")
        return self
    
    def check_folder_contains_1_csv(self):
        if self.contents_path[-4:] != '.csv':
            raise Exception("The file in the folder should be a csv.")
        return self

    def load_data(self):
        self.check_folder_contains_1_item()
        self.check_folder_contains_1_file()
        self.check_folder_contains_1_csv()
        headers = initialise_data_details.headers[self.subfolder]
        return pd.read_csv(self.contents_path, header=0, names=headers)

def load_tables(root:str, tables:list):
    if 'study' in tables:
        study = LoadIndividualDataTables(root, 'study').load_data()
    else:
        study = 'Not requested.'
    if 'icd_labels' in tables:
        icd = LoadIndividualDataTables(root, 'icd_labels').load_data()
    else:
        icd = 'Not requested.'
    if 'icd_reviewed' in tables:
        icd_reviewed = LoadIndividualDataTables(root, 'icd_reviewed').load_data()
    else:
        icd_reviewed = 'Not requested.'
    return study, icd, icd_reviewed

if __name__ == "__main__":
    root_data = 'data/'

    study, icd_labels, icd_reviewed = load_tables(root=root_data,
                                           tables=['study', 'icd_labels', 'icd_reviewed'])