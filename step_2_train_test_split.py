import pandas as pd
from os import path, makedirs
from sklearn.model_selection import train_test_split


def create_chosen_label_col_name(chosen_label:int):
    chosen_label_col_name = 'label_' + str(chosen_label)
    return chosen_label_col_name


def check_folder_exists_and_create(folder_path):
    if path.isdir(folder_path)==False:
        makedirs(folder_path)
        print(f"{folder_path} folder created")


def list_of_chosen_labels_and_non(chosen_label:int, n_labels:int):
    chosen_label_col_name = create_chosen_label_col_name(chosen_label)
    not_chosen_col_names = []
    not_chosen_col_names.extend(range(n_labels))
    not_chosen_col_names.remove(chosen_label)
    not_chosen_col_names = ['label_' + str(i) for i in not_chosen_col_names]
    return chosen_label_col_name, not_chosen_col_names


def save_df_index_as_txt(dataframe:pd.DataFrame(), filename:str):
    pd.Series(dataframe.index).to_csv(filename + '.txt', index=False, header=False)
    print(filename + '.txt successfully saved')
    
    
def convert_csv_with_header_to_dict(csv_filename:str):
    label_dict = pd.read_csv(csv_filename + '.csv')
    return dict(label_dict.to_dict(orient='split')['data'])


def prepare_x_y(chosen_label):
    prepared_one_hot = pd.read_csv('prepared_one_hot.csv', index_col='study_id')
    n_labels = len(convert_csv_with_header_to_dict('label_dict'))
 
    chosen_label_col_name, not_chosen_col_names = list_of_chosen_labels_and_non(chosen_label, n_labels)
    
    x = (prepared_one_hot.drop(columns=not_chosen_col_names)
                            .drop(columns=[chosen_label_col_name])
    )
    y = prepared_one_hot.loc[:,chosen_label_col_name]
    return x, y


def chosen_label_split_train_test_and_save_study_ids(chosen_label:int, chosen_random_states:list):
    chosen_random_state = chosen_random_states[chosen_label]
    
    x,y = prepare_x_y(chosen_label)
    
    # split x and y
    x_train_for_chosen_label, x_test_for_chosen_label, _, __ = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=chosen_random_state,
                                                        shuffle=True,
                                                        stratify=y)
    
    # save the studyids of train and test
    chosen_label_col_name = create_chosen_label_col_name(chosen_label)
    folder_name = chosen_label_col_name + '/'
    
    check_folder_exists_and_create(folder_name)    
    save_df_index_as_txt(x_train_for_chosen_label, filename=folder_name+chosen_label_col_name+'_train_study_ids')
    save_df_index_as_txt(x_test_for_chosen_label, filename=folder_name+chosen_label_col_name+'_test_study_ids')


def all_labels_split_train_test_and_save_study_ids(chosen_random_states:list):
    n_labels = len(convert_csv_with_header_to_dict('label_dict'))
    
    for i in range(n_labels):
        chosen_label_split_train_test_and_save_study_ids(chosen_label=i, chosen_random_states=chosen_random_states)

