import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix, cohen_kappa_score

import load_data
from clean_icd_labels import CleanData


def return_r1_series():
    data_folder= 'data/'
    _, icd, __ = load_data.load_tables(data_folder, ['icd_labels'])  
    labels_dict, r1 = CleanData(icd, 'Neoplasms').clean_labels()
    r1 = r1.replace(labels_dict)
    r1 = r1.set_index('study_id')
    return r1


def r2_deidentify_studies_from_excel(folder_path, excel_name, sheet_name):
    data = pd.read_excel(folder_path + excel_name,
                         sheet_name=sheet_name)
    data['study_id'] = pd.read_csv(folder_path + sheet_name + '.txt',
                                   header=None, names=['study_id'])
    data.set_index('study_id', inplace=True)
    data_r2_labels = data.iloc[:,7:]
    return data_r2_labels


def r2_unpivot_excel(excel_df):
    unpivot_excel_df = excel_df.melt(ignore_index=False, value_name='label').drop(labels='variable', axis=1).dropna()
    unpivot_excel_df = unpivot_excel_df[unpivot_excel_df.loc[:,'label']!='Neoplasms']
    unpivot_excel_df['label'] = unpivot_excel_df.loc[:,'label'].str[8:]
    change_to_non_neo = {'lasms':'NOT_Neoplasms'}
    unpivot_excel_df = unpivot_excel_df.replace(change_to_non_neo)
    return unpivot_excel_df
    

def return_r2_series():
    excel_name = '2022-10-11 - Copy of 2022-09-27 Studies to review.xlsx'
    folder_path = 'data/icd_additional_review/'
    r2_group_1 = r2_deidentify_studies_from_excel(folder_path, excel_name, 'Group 1')
    r2_group_2 = r2_deidentify_studies_from_excel(folder_path, excel_name, 'Group 2')
    r2_pivoted = pd.concat([r2_group_1, r2_group_2])
    r2 = r2_unpivot_excel(r2_pivoted)
    return r2

    
# one hot encoded version
def one_hot_encode_pd_series(series:pd.Series()):
    return pd.get_dummies(series).astype(int).groupby('study_id').sum()


def return_label_index(label__name):
    label_dict_df = pd.read_csv('label_dict.csv')
    label_dict = dict(zip(label_dict_df.iloc[:,1], label_dict_df.iloc[:,0]))
    label_name = label__name.replace('label_', '')
    chosen_label = label_dict[label_name]
    return chosen_label
    
    
def create_comparison_for_one_label(r1_col_index):    
    r1 = return_r1_series()
    r1_one_hot = one_hot_encode_pd_series(r1)
    label = r1_one_hot.columns[r1_col_index]

    r2 = return_r2_series()
    r2_one_hot = one_hot_encode_pd_series(r2)

    r1_for_label = r1_one_hot.loc[:,label]
    # if r2 did not label anything as this
    if r2_one_hot.columns.isin([label]).sum()>0:
        r2_for_label = r2_one_hot.loc[:,label]
    else:
        r2_for_label = pd.Series(data=[0]*len(r2_one_hot),
                                index=r2_one_hot.index,
                                name=label)
        
    compare_for_label = pd.merge(left=r1_for_label,
                                right=r2_for_label,
                                how='inner',
                                left_index=True,
                                right_index=True,
                                suffixes=['_r1', '_r2'])

    compare_for_label['agree'] = (compare_for_label.iloc[:,0]==compare_for_label.iloc[:,1]).astype(int)
    
    chosen_label = return_label_index(label)
    
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_interrater.csv"
    compare_for_label.to_csv(path_or_buf=file_name,
                             index=True,
                             header=True)
    print(f"{file_name} successfully saved")
    # return compare_for_label
    # print(f"{label}: {compare_for_label.loc[:,'agree'].sum()}, {len(compare_for_label)}")


# for chosen_label in range(19):
#     create_comparison_for_one_label(chosen_label)

interrater_for_one_label = pd.read_csv(f"label_{str(chosen_label)}/label_{str(chosen_label)}_interrater.csv",
                                    index_col='study_id')


def calculate_n(interrater_for_one_label):
    n_reviewed = len(interrater_for_one_label)
    pos_class_n = interrater_for_one_label.iloc[:,0].sum()
    neg_class_n = n_reviewed - pos_class_n
    agree_by_class = pd.DataFrame(interrater_for_one_label.groupby(interrater_for_one_label.columns[0])['agree'].sum()).T
    pos_agree_n = agree_by_class.loc[:,1].values[0]
    neg_agree_n = agree_by_class.loc[:,0].values[0]
    return [n_reviewed, pos_class_n, neg_class_n, pos_agree_n, neg_agree_n]


def save_interrater_performance():
    score_df = pd.DataFrame(columns=['label_index',
                                    'interrater_cohen_kappa_score',
                                    'n_reviewed',
                                    'pos_class_n',
                                    'neg_class_n',
                                    'pos_agree_n',
                                    'neg_agree_n'])
    for chosen_label in range(19):
        interrater_for_one_label = pd.read_csv(f"label_{str(chosen_label)}/label_{str(chosen_label)}_interrater.csv",
                                            index_col='study_id')
        score = cohen_kappa_score(y1=interrater_for_one_label.iloc[:,0],
                          y2=interrater_for_one_label.iloc[:,1])
        counts = calculate_n(interrater_for_one_label)
        new_row = pd.DataFrame([[chosen_label, score]+counts], columns=score_df.columns)
        score_df = pd.concat([score_df, new_row])
    score_df.to_csv(f"interrater_performance.csv", index=False)
    print(f"interrater_performance.csv successfully saved")        
        
        
# save_interrater_performance()


# jaccard score
## per class
jaccard_score(y_true=interrater_for_one_label.iloc[:,0],
              y_pred=interrater_for_one_label.iloc[:,1],
              average=None)

# confusion matrix
multilabel_confusion_matrix(y_true=data_r1_one_hot,
                 y_pred=data_r2_one_hot_r1_layout)

