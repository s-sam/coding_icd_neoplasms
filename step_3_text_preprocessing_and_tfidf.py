from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump
from scipy.sparse import save_npz

from step_2_train_test_split import prepare_x_y, create_chosen_label_col_name, convert_csv_with_header_to_dict
import clean_study_text


def load_txt_one_col_file(file_path):
    with open(file_path, 'r') as file_contents:
        contents_as_list = list(map(int, file_contents.read().splitlines()))
    return contents_as_list


def load_study_ids(train_or_test:str, chosen_label:int):
    chosen_label_col_name = create_chosen_label_col_name(chosen_label)
    file_path = chosen_label_col_name + '/' + chosen_label_col_name + '_' + train_or_test + '_study_ids.txt'
    
    study_ids = load_txt_one_col_file(file_path)
    return study_ids


def prepare_x_y_for_chosen_label(train_or_test:str, chosen_label:int):
    study_ids = load_study_ids(train_or_test, chosen_label)
    x,y = prepare_x_y(chosen_label)
    x_for_chosen_label = x.loc[study_ids]
    y_for_chosen_label = y.loc[study_ids]
    
    return x_for_chosen_label, y_for_chosen_label


def save_tfidf(chosen_label:int, tfidf, train_or_test:str):
    # save the test tfidf data
    # save the tfidf too as scipy format
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_x_{train_or_test}_tfidf.npz"
    save_npz(file_name, tfidf)
    print(file_name + ' successfully saved')
    

def save_vectoriser(chosen_label:int, vectoriser):
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_tfidf_vectoriser.pkl"
    dump(vectoriser, open(file_name, 'wb'))
    print(file_name + ' successfully saved')
    
    
def chosen_label_train_text_preprocessing_and_save_tfidf(chosen_label:int):
    x_train_for_chosen_label, _ = prepare_x_y_for_chosen_label(train_or_test='train', chosen_label=chosen_label)
    x_train_prepared = clean_study_text.CleanData(study_data=x_train_for_chosen_label).run()
    
    tfidf_vectoriser = TfidfVectorizer()
    x_train_prepared_tfidf = tfidf_vectoriser.fit_transform(x_train_prepared)
    
    # save the vectoriser so that it can be used on the test set
    save_vectoriser(chosen_label=chosen_label, vectoriser=tfidf_vectoriser)
    
    # save the tfidf too as scipy format
    save_tfidf(chosen_label=chosen_label,
               tfidf=x_train_prepared_tfidf,
               train_or_test='train')


def all_labels_train_text_preprocessing_and_save_tfidf():
    n_labels = len(convert_csv_with_header_to_dict('label_dict'))
    
    for i in range(n_labels):
        chosen_label_train_text_preprocessing_and_save_tfidf(chosen_label=i)

