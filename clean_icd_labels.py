import pandas as pd


def remove_rows_where_two_datapoints_exist(dataframe, column1, column1_value, column2, column2_value):
    return dataframe.loc[~( (dataframe.loc[:, column1] == column1_value) & (dataframe.loc[:, column2] == column2_value) )]


def indexes_where_datapoint_exists(dataframe, column1, column1_value):
    return dataframe.loc[dataframe.loc[:, column1] == column1_value].index.tolist()


def indexes_where_datapoint_does_not_exist(dataframe, column1, column1_value):
    return dataframe.loc[dataframe.loc[:, column1] != column1_value].index.tolist()


def unique_list_of_values(dataframe, column):
    return dataframe.loc[:, column].unique().tolist()


def remove_item_from_list(list, item):
    list.remove(item)
    return list


def unique_list_of_values_minus_item(dataframe, column, item, sort=True, reverse=False):
    unique_values = unique_list_of_values(dataframe, column)
    unique_values_minus_one = remove_item_from_list(unique_values, item)
    if sort == True:
        unique_values_minus_one.sort()
    return unique_values_minus_one


class CleanData():
    def __init__(self, label_data:pd.DataFrame(), chapter_of_interest:str):
        self.label_data = label_data
        self.chapter_of_interest = chapter_of_interest
        self.chapters_not_of_interest_name = 'NOT_' + self.chapter_of_interest
        self.len_chapter = self.label_data.chapter.nunique()
        self.len_block = self.label_data.block.nunique()
        self.indexes_chapters_of_interest = indexes_where_datapoint_exists(self.label_data, 'chapter', self.chapter_of_interest)
        self.indexes_chapters_not_of_interest = indexes_where_datapoint_does_not_exist(self.label_data, 'chapter', self.chapter_of_interest)

    def drop_columns(self, list_of_columns):
        self.label_data = self.label_data.drop(columns=list_of_columns)
        return self

    def drop_rows_with_nulls(self):
        self.label_data = self.label_data.dropna()
        return self

    def change_chapter_of_interest_to_1(self):
        self.label_data.loc[self.indexes_chapters_of_interest, 'chapter'] = 1
        return self

    def change_column_where_chapters_not_of_interest_to_0(self, column):
        self.label_data.loc[self.indexes_chapters_not_of_interest, column] = 0
        return self

    def create_label_mapping(self):
        labels_of_interest = unique_list_of_values_minus_item(self.label_data, 'block', 0)
        self.labels = [self.chapters_not_of_interest_name] + labels_of_interest
        len_labels = len(self.labels)
        self.labels_mapped_to = [i for i in range(len_labels)]
        return self

    def initialise_labels_dict(self):
        self.label_dict = dict(zip(self.labels, self.labels_mapped_to))
        return self
    
    def reverse_labels_dict(self):
        self.label_dict_rev = {v: k for k, v in self.label_dict.items()}
        return self

    def change_blocks_within_chapter_of_interest_to_numerical(self):
        self.label_data = self.label_data.replace({'block': self.label_dict})
        return self

    def drop_duplicates(self):
        self.label_data = self.label_data.drop_duplicates()
        return self

    def unique_count_of_chapters_per_study(self):
        unique_count_per_study = self.label_data.groupby(['study_id'])['chapter'].nunique()
        unique_count_per_study = unique_count_per_study.rename('unique_count_of_chapters')
        self.label_data = pd.merge(self.label_data, unique_count_per_study, on='study_id', how='outer')
        return self

    def remove_rows_with_chapter_0_where_study_has_both_labels(self):
        self.label_data = remove_rows_where_two_datapoints_exist(self.label_data, 'block', 0,
                                                          'unique_count_of_chapters', 2)
        return self
    
    def rename_column(self, column, new_name):
        self.label_data = self.label_data.rename(columns={column: new_name})
        return self
        
    def clean_labels(self):
        self.drop_rows_with_nulls()
        self.change_column_where_chapters_not_of_interest_to_0('chapter')
        self.change_column_where_chapters_not_of_interest_to_0('block') # +
        self.create_label_mapping()
        self.change_chapter_of_interest_to_1()
        self.drop_duplicates()
        self.unique_count_of_chapters_per_study()
        self.remove_rows_with_chapter_0_where_study_has_both_labels()
        self.drop_columns(['unique_count_of_chapters'])
        self.initialise_labels_dict() # +
        self.change_blocks_within_chapter_of_interest_to_numerical() # +
        self.drop_columns(['chapter']) # +
        self.rename_column('block', 'label')
        self.reverse_labels_dict()
        return self.label_dict_rev, self.label_data


if __name__ == "__main__":
    root_data = 'data/'

    import load_data
    _, icd, __ = load_data.load_tables(root_data, ['icd_labels'])    
    print(len(icd))
    print(icd.loc[:,'study_id'].nunique())
    
    label_dict, icd_labels_prepared = CleanData(icd, 'Neoplasms').clean_labels()
    print(len(icd_labels_prepared))
    print(icd_labels_prepared.loc[:,'study_id'].nunique())    
    
    print(f"We started off with {len(icd)} rows of labels from {icd.loc[:,'study_id'].nunique()} studies."
          f"\nWhen blocks not of interest were converted to 0s, \nthis reduced to {len(icd_labels_prepared)} rows of labels"
          f"  {icd_labels_prepared.loc[:,'study_id'].nunique()} studies")
    