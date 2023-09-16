import pandas as pd


def return_df_of_unique_ids(dataframe:pd.DataFrame(), id_column:str):
    unique_study_ids = dataframe.loc[:, id_column].unique()
    unique_study_ids_df = pd.DataFrame(unique_study_ids, columns=[id_column])
    return unique_study_ids_df


def run(icd_reviewed_data:pd.DataFrame()):
    unique_study_ids_df = return_df_of_unique_ids(icd_reviewed_data, 'study_id')
    return unique_study_ids_df

    
if __name__ == "__main__":
    root_data = 'data/'

    import load_data
    _, __, icd_reviewed = load_data.load_tables(root_data, ['icd_reviewed']) 

    icd_reviewed_prepared = run(icd_reviewed)
    print(icd_reviewed_prepared.study_id.nunique(), len(icd_reviewed_prepared))
