import joblib

import clean_study_text
from step_2_train_test_split import create_chosen_label_col_name
from step_3_text_preprocessing_and_tfidf import prepare_x_y_for_chosen_label, save_tfidf


def load_vectoriser(chosen_label:int):
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_tfidf_vectoriser.pkl"
    vectoriser = joblib.load(file_name)
    return vectoriser


def clean_and_tfidf_test_data(chosen_label:int):
    vectoriser = load_vectoriser(chosen_label)
    x_test, _ = prepare_x_y_for_chosen_label(train_or_test='test',
                                                  chosen_label=chosen_label)
    x_train_prepared = clean_study_text.CleanData(study_data=x_test).run()
    x_test_prepared_tfidf = vectoriser.transform(x_train_prepared)
    
    save_tfidf(chosen_label=chosen_label,
               tfidf=x_test_prepared_tfidf,
               train_or_test='test')
    

if __name__ == '__main__':
    chosen_label=18
    vectoriser = load_vectoriser(chosen_label=chosen_label)
    x_test, _ = prepare_x_y_for_chosen_label(train_or_test='test',
                                                    chosen_label=chosen_label)
    x_train_prepared = clean_study_text.CleanData(study_data=x_test).run()
    
    x_test_prepared_tfidf = vectoriser.transform(x_train_prepared)
    
    print(len(x_test),     len(x_train_prepared), x_test_prepared_tfidf.shape)
        
    chosen_label_col_name = create_chosen_label_col_name(chosen_label)
    file_path = chosen_label_col_name + '/' + chosen_label_col_name + '_' + 'test' + '_study_ids.txt'
    
    with open(file_path, 'r') as file_contents:
        study_ids = list(map(int, file_contents.read().splitlines()))
        
    print(len(study_ids))