import pandas as pd

import load_data
import clean_icd_labels
import clean_icd_reviewed
import create_dataset

def save_as_csv(dataframe:pd.DataFrame(), dictionary:dict, filename:str, file_type='csv', index=True, header=True):
    if type(dictionary) == dict:
        dataframe = pd.DataFrame.from_dict(dictionary, orient="index")
    
    dataframe.to_csv(filename + '.' + file_type, index=index, header=header)
    print(filename + '.' + file_type + ' successfully saved')

def run(root:str):
    """loads the raw data, cleans 2 of the datasets (labels and reviewed) and merges raw study with the 2 clean datasets.
    Saves a one hot version of the data called 'prepared_one_hot.csv' in the immediate folder.
    Saves a dictionary of integer index to label names called 'label_dict.csv'.

    Args:
        root (string): folder path excluding the last slash indicating where the raw data is held
    """    ''''''
    #i load raw data
    study, icd_labels, icd_reviewed = load_data.load_tables(root, ['study', 'icd_labels', 'icd_reviewed'])    

    #ii clean 2/3 datasets
    ## clean labels
    label_dict, icd_labels_prepared = clean_icd_labels.CleanData(icd_labels, 'Neoplasms').clean_labels()
    ## clean reviewed
    icd_reviewed_prepared = clean_icd_reviewed.run(icd_reviewed)

    #iii merge raw study and 2 clean datasets
    prepared = create_dataset.merge_data(study, icd_labels_prepared, icd_reviewed_prepared, text_cols_only=True)

    #iv one hot the labels
    prepared_one_hot = create_dataset.one_hot(prepared)

    # save the one hot version and the label dict
    save_as_csv(prepared_one_hot, None,  'prepared_one_hot')
    save_as_csv(None, label_dict, 'label_dict')
    
    