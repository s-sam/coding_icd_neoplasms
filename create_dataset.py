import pandas as pd


def merge_two_dfs(left:pd.DataFrame(), right:pd.DataFrame(), on:str, how:str, add_binary_indicator_col = False, binary_indicator_col_name = False):
    merged = pd.merge(left, right, on=on, how=how,
                      indicator=add_binary_indicator_col)

    if add_binary_indicator_col == True:
        merged[binary_indicator_col_name] = merged['_merge'].replace('left_only', 0).replace('both', 1).astype(bool)
        merged = merged.drop(columns=['_merge'])

    merged = merged.convert_dtypes()
    return merged


def merge_data(study:pd.DataFrame(), icd_labels_prepared:pd.DataFrame(), icd_reviewed_prepared:pd.DataFrame(),
               text_cols_only=True):
    if text_cols_only:
        text_cols = ['study_id']
        text_cols += ['short_name', 'title', 'research_summary', 'inclusion_criteria']
        study = study.loc[:,text_cols]
    
    studies_icd_reviewed = merge_two_dfs(study, icd_reviewed_prepared,
                             on='study_id', how='inner',
                             add_binary_indicator_col = False, binary_indicator_col_name = False)
    
    prepared = merge_two_dfs(left=studies_icd_reviewed,
                             right=icd_labels_prepared,
                             on='study_id', how='left',
                             add_binary_indicator_col = False,
                             binary_indicator_col_name = False)

    prepared['label'] = prepared.loc[:,'label'].fillna(0)
    
    return prepared


def one_hot(prepared:pd.DataFrame()):
    text_cols = ['study_id']
    text_cols += ['short_name', 'title', 'research_summary', 'inclusion_criteria']    
    prepared_one_hot = pd.get_dummies(prepared, columns=['label'], dtype=int)
    prepared_one_hot = prepared_one_hot.groupby(text_cols, dropna=False).sum()
    return prepared_one_hot
    

if __name__ == "__main__":
    import load_data
    import clean_icd_labels
    import clean_icd_reviewed

    # load raw data
    root_data = 'data/'
    study, icd_labels, icd_reviewed = load_data.load_tables(root_data, ['study', 'icd_labels', 'icd_reviewed'])    

    # clean 2/3 datasets
    ## clean labels
    label_dict, icd_labels_prepared = clean_icd_labels.CleanData(icd_labels, 'Neoplasms').clean_labels()
    ## clean reviewed
    icd_reviewed_prepared = clean_icd_reviewed.run(icd_reviewed)

    # merge raw study and 2 clean datasets
    prepared = merge_data(study, icd_labels_prepared, icd_reviewed_prepared,
                          text_cols_only=True)
    
    