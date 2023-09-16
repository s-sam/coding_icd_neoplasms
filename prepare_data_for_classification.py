import pandas as pd

import clean_study_text
import clean_icd_reviewed
import clean_icd_labels

def merge_two_dfs(left, right, on, add_binary_indicator_col = False, binary_indicator_col_name = False):
    merged = pd.merge(left, right, on = on, how = 'left', indicator = add_binary_indicator_col)

    if add_binary_indicator_col == True:
        merged[binary_indicator_col_name] = merged['_merge'].replace('left_only', 0).replace('both', 1).astype(bool)
        merged = merged.drop(columns=['_merge'])

    merged = merged.dropna()
    merged = merged.convert_dtypes()
    return merged

class PrepareData():
    def __init__(self, root:str, reviewed_only:bool, binary_or_multiclass:str, chapter_of_interest:str,
                 fillna_value:str, concat_separator:str, remove_empty_rows:bool,
                 lowercase=True, symbol_replace=True, symbol_fill_value=' ',
                 whole_number_replace=True, whole_number_fill_value='',
                 multiple_spaces_to_1=True, remove_start_end_spaces=True,
                 remove_stopwords=True):
        self.root = root
        self.reviewed_only = reviewed_only
        self.binary_or_multiclass = binary_or_multiclass
        self.chapter_of_interest = chapter_of_interest
        self.fillna_value = fillna_value
        self.concat_separator = concat_separator
        self.remove_empty_rows = remove_empty_rows
        self.lowercase = lowercase
        self.symbol_replace = symbol_replace
        self.symbol_fill_value = symbol_fill_value
        self.whole_number_replace = whole_number_replace
        self.whole_number_fill_value = whole_number_fill_value
        self.multiple_spaces_to_1 = multiple_spaces_to_1
        self.remove_start_end_spaces = remove_start_end_spaces
        self.remove_stopwords = remove_stopwords

    def load_icd_reviewed_prepared(self):
        self.icd_reviewed_prepared = clean_icd_reviewed.run_pipeline(self.root)
        return self

    def load_icd_labels_prepared(self):
        self.labels_dict, self.icd_prepared = (clean_icd_labels.
                                                                 CleanData(self.root, self.binary_or_multiclass,
                                                                           self.chapter_of_interest)
                                                                 .clean_labels()
        )
        return self

    def load_text_prepared(self):
        self.text_prepared = (clean_study_text
                              .CleanData(self.root, self.fillna_value, self.concat_separator,
                                         self.remove_empty_rows, self.lowercase, self.symbol_replace,
                                         self.symbol_fill_value, self.whole_number_replace,
                                         self.whole_number_fill_value, self.multiple_spaces_to_1,
                                         self.remove_start_end_spaces, self.remove_stopwords)
                              .run_pipeline()
        )
        return self

    def clean_and_merge(self):
        self.load_text_prepared(), self.load_icd_labels_prepared()
        text_and_icd_prepared = merge_two_dfs(self.text_prepared, self.icd_prepared, 'study_id')
        self.load_icd_reviewed_prepared()
        text_and_icd_and_reviewed_prepared = merge_two_dfs(text_and_icd_prepared, self.icd_reviewed_prepared,
                                                    'study_id', True, 'reviewed')
        if self.reviewed_only:
            text_and_icd_and_reviewed_prepared = text_and_icd_and_reviewed_prepared.loc[text_and_icd_and_reviewed_prepared.loc[:,'reviewed']==1]

        return self.labels_dict, text_and_icd_and_reviewed_prepared

if __name__ == "__main__":
    root_data = 'data/'

    labels_dict, text_and_icd_and_reviewed_prepared = (PrepareData(root_data, 'multiclass', 'Neoplasms',
                                                                               '', ' ', True)
                                                                    .clean_and_merge()
                                                                    )