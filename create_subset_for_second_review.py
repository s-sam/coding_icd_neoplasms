root_data = 'data/'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import scipy
random_state = 0

import load_data
import prepare_data_for_classification

def calculate_num_tokens(dataframe, text_column_name):
    dataframe['num_tokens'] = dataframe[text_column_name].str.split().str.len()
    return dataframe

def section_into_bins(dataframe, num_tokens_column_name, num_bins, reassign_edge_bins=True):
    dataframe_copy = dataframe.copy()
    dataframe_copy['bin'], bin_array = pd.cut(dataframe_copy[num_tokens_column_name], bins=num_bins, retbins=True)

    if reassign_edge_bins == True:
        bin_array[0] = 0
        bin_array[-1] = 999999999
        dataframe_copy['bin'] = pd.cut(dataframe_copy[num_tokens_column_name], bins=bin_array, labels=False)
    return dataframe_copy, bin_array

def concat_columns(dataframe, new_col_name, column1, column2, separator):
    dataframe_copy = dataframe.copy()
    dataframe_copy[new_col_name] = (dataframe_copy.loc[:, column1].astype(str) +
                                    separator +  dataframe_copy.loc[:, column2].astype(str)
                                    )
    return dataframe_copy

def split_unreviewed_reviewed(dataframe, reviewed_column_name, sort_cols, label_bin_col):
    unreviewed = dataframe[dataframe['reviewed'] == False].sort_values(sort_cols)
    reviewed = dataframe[dataframe['reviewed'] == True].sort_values(sort_cols)
    return unreviewed, reviewed

def prepare_dataframe_with_extra_cols(dataframe, text_column_name, num_tokens_column_name,
                                      num_bins, label_bin_col, reviewed_column_name,
                                      sort_cols, reassign_edge_bins=True):
    dataframe_copy = dataframe.copy()
    df_with_num_tokens = calculate_num_tokens(dataframe_copy, text_column_name)
    df_with_bins, bin_array = section_into_bins(df_with_num_tokens, num_tokens_column_name, num_bins, reassign_edge_bins=True)
    df_with_label_bins_cols = concat_columns(df_with_bins, label_bin_col, 'label', 'bin', '_')
    unreviewed, reviewed = split_unreviewed_reviewed(df_with_label_bins_cols, reviewed_column_name, sort_cols, label_bin_col)
    return unreviewed, reviewed


# # create counts dicts

def count_dict_template(labels_dict, num_bins, label=True, label_bin=True):
    label_count_dict_template = None
    label_bin_count_dict_template = None

    if label == True:
        label_count_dict_template = dict([[x, 0] for x in range(len(labels_dict))])

    if label_bin == True:
        label_bin_count_dict_template = dict([[str(x)+'_'+str(y), 0] for x in range(len(labels_dict))
                                                for y in range(num_bins)])

    return label_count_dict_template, label_bin_count_dict_template

def summary_counts_dicts(dataframe, label=True, label_bin=True):
    if label == True:
        label_count_dict = (pd.DataFrame(dataframe.groupby(['label'])['label_bin'].count())
                        .rename(columns={'label_bin': 'l_count'})
                        .T
                       .to_dict('records')[0]
        )
 
    if label_bin == True:
        label_bin_count_dict = (pd.DataFrame(dataframe.groupby(['label_bin'])['label'].count())
                        .rename(columns={'label': 'l_b_count'})
                        .T
                       .to_dict('records')[0]
        )

    return label_count_dict, label_bin_count_dict

def update_dict_with_another(dict_to_update, other_dict):
    dict_to_update_copy = dict_to_update.copy()
    dict_to_update_copy.update(other_dict)
    return dict_to_update_copy
    
def summary_counts_dicts_w_zeros(dataframe, labels_dict, num_bins, label_bin_col,
                                 reviewed_column_name, label=True, label_bin=True):
    
    label_count_dict_template, label_bin_count_dict_template = count_dict_template(labels_dict,
                                                                                   num_bins, label=True,
                                                                                   label_bin=True)
    label_count_dict, label_bin_count_dict = summary_counts_dicts(dataframe, label=True, label_bin=True)

    label_count_dict_w_zeros = update_dict_with_another(label_count_dict_template, label_count_dict)
    label_bin_count_dict_w_zeros = update_dict_with_another(label_bin_count_dict_template, label_bin_count_dict)

    return label_count_dict_w_zeros, label_bin_count_dict_w_zeros


# # create cumulative dicts


def replace_dict_values_with_cumulative(dict, end_val_included=True):
    dict_copy = dict.copy()
    dict_values_as_array = np.fromiter(dict_copy.values(), dtype=int)

    cumulative_values = np.cumsum(dict_values_as_array) - int(end_val_included)

    dict_copy = {k: cumulative_values[i] for i, k in enumerate(dict_copy)}
    return dict_copy


# # get shuffled indexes
def shuffled_indexes_within_label_bin(dataframe, label_bins, labels_dict, num_bins, label_cumulative_dict, label_bin_cumulative_dict,
                     label_bin_col):
    # label_bin only
    num_labels = len(labels_dict)
    shuffled_indexes = np.array([])

    for i, key in enumerate(label_bins):
        label, bin = map(int, key.split('_'))
        indexes = np.array(dataframe[dataframe[label_bin_col] == key].index)
        np.random.seed(random_state); np.random.shuffle(indexes)

        shuffled_indexes = np.concatenate([shuffled_indexes, indexes])
    
    return shuffled_indexes.astype(int)


def shuffled_indexes_within_label(dataframe, labels, labels_dict, num_bins, label_cumulative_dict, label_bin_cumulative_dict,
                     label_col):
    # label only
    num_labels = len(labels_dict)
    shuffled_indexes = np.array([])

    for i, key in enumerate(labels):
        indexes = np.array(dataframe[dataframe[label_col] == key].index)
        np.random.seed(random_state); np.random.shuffle(indexes)

        shuffled_indexes = np.concatenate([shuffled_indexes, indexes])
    
    return shuffled_indexes.astype(int)

# # sample counts

def sample_count_dict(dataframe, percent, labels_dict, num_bins, label_bin_col,
                                 reviewed_column_name, label=True, label_bin=True):

    _, label_bin_sample = summary_counts_dicts_w_zeros(dataframe, labels_dict, 10, 'label_bin',
                                 'reviewed', label=False, label_bin=True)
    
    #label_bin only
    label_bin_vals = np.fromiter(label_bin_sample.values(), dtype=int)
    sample_counts = np.ceil(label_bin_vals * percent).astype(int)
    label_bin_sample = dict(zip(label_bin_sample.keys(), sample_counts))
    return label_bin_sample


# # put it all together

def sample(dataframe, percent):
    sample_counts_w_zeros = sample_count_dict(unreviewed, percent, labels_dict, 10, 'label_bin',
                                 'reviewed', label=True, label_bin=True)


    chosen = np.array([])
    
    label_count_dict_w_zeros, label_bin_count_dict_w_zeros = summary_counts_dicts_w_zeros(dataframe,
                                                                                      labels_dict, 10,
                                                                                      'label_bin', 'reviewed',
                                                                                      label=True, label_bin=True)

    label_cumulative_dict = replace_dict_values_with_cumulative(label_count_dict_w_zeros, end_val_included=True)
    label_bin_cumulative_dict = replace_dict_values_with_cumulative(label_bin_count_dict_w_zeros, end_val_included=True)


    # for key in ['11_0', '11_1']:
    for key in sample_counts_w_zeros:
        label, bin = map(int, key.split('_'))
        # print(label, bin)

        if bin == 0:
            label_indexes = shuffled_indexes_within_label(dataframe, [label], labels_dict, 10,
                                                          label_cumulative_dict, label_bin_cumulative_dict,
                                                          'label')
        sample_amount = sample_counts_w_zeros[key]

        if sample_amount > 0:
            label_bin_indexes = shuffled_indexes_within_label_bin(dataframe, [key], labels_dict, 10,
                                    label_cumulative_dict, label_bin_cumulative_dict, 'label_bin')
            
            already_chosen_l_b_indexes = [np.where(label_bin_indexes == x)[0][0] for x in chosen if np.where(label_bin_indexes == x)[0].tolist()!=[]]
            label_bin_indexes_remaining = np.delete(label_bin_indexes, already_chosen_l_b_indexes)

            available_count = label_bin_indexes_remaining.size

            sequential_sample_amount = min(sample_amount, available_count)

            sequential_chosen = label_bin_indexes_remaining[0:sequential_sample_amount]

            chosen = np.append(chosen, sequential_chosen)

            random_within_label_sample_amount = sample_amount - sequential_sample_amount

            # print(sample_amount, sequential_sample_amount, random_within_label_sample_amount)


            if random_within_label_sample_amount > 0:
                print("label ", label, "bin", bin)
                print("random_within_label_sample_amount = ", random_within_label_sample_amount)


                already_chosen_indexes = [np.where(label_indexes == x)[0][0] for x in chosen if np.where(label_indexes == x)[0].tolist()!=[]]
                
                label_indexes = np.delete(label_indexes, already_chosen_indexes)

                print("labels left: ", label_indexes.size)

                random_chosen = label_indexes[0:random_within_label_sample_amount]

                chosen = np.append(chosen, random_chosen)
                print(random_chosen)

    chosen_study_ids = dataframe.loc[chosen.astype(int)].study_id.unique()

    print(chosen_study_ids.size)
    return np.array(chosen_study_ids)


def final_table(dataframe, indexes, return_label=False):
    study_table = study[['study_id', 'short_name', 'title', 'research_summary', 'inclusion_criteria']]
    study_table = study_table[study_table.study_id.isin(indexes)]
    study_table = study_table.sample(frac=1, random_state=random_state)

    study_table = study_table.reset_index(drop=True)
    study_table.index.names = ['Index']

    if return_label==True:
        study_table = study_table.reset_index(drop=False)
        return pd.merge(study_table, dataframe, on='study_id', how='left')

    study_table = study_table.drop(columns=['study_id'], axis=1)
    study_table['Text'] = study_table.astype(str).agg('. '.join, axis=1)

    study_table = study_table[['Text', 'short_name', 'title', 'research_summary', 'inclusion_criteria']]
    study_table.columns = ['Text', 'Short Name', 'Title', 'Research Summary', 'Inclusion Criteria']
    
    icd_block = ['If Neoplasms, ('+str(x+1)+') Choose ICD Blocks' for x in range(18)]    
    for i in ['Neoplasms ICD Chapter?'] + icd_block:
        study_table[i] = np.nan

    return study_table

def separate_chosen(indexes1, indexes2):
    indexes = [np.where(indexes2 == x)[0][0] for x in indexes1 if np.where(indexes2 == x)[0].tolist()!=[]]
    return np.delete(indexes2, indexes)


if __name__ == '__main__':
    labels_dict, text_and_icd_and_reviewed_prepared = (prepare_data_for_classification
                                                                    .PrepareData(root_data, False, 'multiclass',
                                                                                'Neoplasms',
                                                                                '', ' ', True)
                                                                    .clean_and_merge()
                                                                    )


    unreviewed, reviewed = prepare_dataframe_with_extra_cols(text_and_icd_and_reviewed_prepared, 'text', 'num_tokens',
                                            10, 'label_bin', 'reviewed',
                                            ['label', 'bin'])

    chosen_14 = sample(reviewed, 0.14)
    chosen_28 = sample(reviewed, 0.28)
    chosen_57 = sample(reviewed, 0.57)

    chosen_0_14 = chosen_14
    chosen_14_28 = separate_chosen(chosen_14, chosen_28)
    chosen_28_57 = separate_chosen(chosen_28, chosen_57)


    study, icd, _ = load_data.load_tables(root_data, ['study', 'icd'])

    # final_table(chosen_0_14, return_label=True).to_csv('chosen_0_14.csv', index=False)
    # final_table(chosen_14_28, return_label=True).to_csv('chosen_14_28.csv', index=False)

    reviewed.label_bin.sort_values().hist()
    unreviewed.label_bin.sort_values().hist()

    final_table(reviewed, chosen_0_14, return_label=True).label_bin.sort_values().hist()
    final_table(reviewed, chosen_14_28, return_label=True).label_bin.sort_values().hist()

    len(chosen_0_14)
    len(chosen_14_28)

    final_table(reviewed, chosen_0_14, return_label=True).head()
    final_table(reviewed, chosen_14_28, return_label=True).head()