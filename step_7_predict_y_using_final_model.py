import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve

import xgboost as xgb

from step_3_text_preprocessing_and_tfidf import prepare_x_y_for_chosen_label

# from step_3_text_preprocessing_and_tfidf import load_study_ids


def load_final_model(chosen_label:int, algorithm:str, file_name_n_iter:str, file_name_min_pos_instances:str):
    if algorithm == 'svm' or algorithm == 'tree' or algorithm=='linear_only_svm':
        final_model_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.pkl"
        svm_or_tree_final_model = joblib.load(final_model_file_name)
        return svm_or_tree_final_model
    elif algorithm == 'xgboostrf':
        raise Exception("this cannot be used for xgboost. the load_model for xgboost returns None")


def load_tfidf_data(chosen_label:int, train_or_test:str):
    x_tfidf_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_x_{train_or_test}_tfidf.npz"
    x_tfidf = load_npz(x_tfidf_file_name)
    return x_tfidf


# def save_optimal_threshold(optimal_threshold:float, chosen_label:int,
#                            algorithm:str, file_name_n_iter:str,
#                            file_name_min_pos_instances:str):
#     file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_optimal_threshold_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
#     with open(file_name, 'w') as file:
#         file.write(str(optimal_threshold))
#     print(f"{file_name} successfully saved")


def predict_y(chosen_label:int, algorithm:str, file_name_n_iter:str, file_name_min_pos_instances:str,
              train_or_test:str):
    if algorithm == 'svm' or algorithm == 'tree' or algorithm=='linear_only_svm':
        svm_or_tree_final_model = load_final_model(chosen_label,
                                    algorithm,
                                    file_name_n_iter,
                                    file_name_min_pos_instances)
        svm_or_tree_x_tfidf = load_tfidf_data(chosen_label, train_or_test)
        svm_or_tree_y_pred = svm_or_tree_final_model.predict(svm_or_tree_x_tfidf)
        return svm_or_tree_y_pred
        
    elif algorithm == 'xgboostrf':
        final_model_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
        xgboostrf_final_model = xgb.XGBRFClassifier()
        booster = xgb.Booster()        
        booster.load_model(final_model_file_name)
        xgboostrf_final_model._Booster = booster
        
        xgboostrf_x_tfidf = load_tfidf_data(chosen_label, train_or_test)
        xgboostrf_y_pred = xgboostrf_final_model.predict_proba(xgboostrf_x_tfidf)[:,1]

        # find threshold using pr curve that returns the best f1 value
        # _, y_true = prepare_x_y_for_chosen_label('test', chosen_label)
        # precision, recall, thresholds = precision_recall_curve(y_true, xgboostrf_y_pred)
        # optimal_idx = np.argmax(2* (precision*recall)/(precision+recall))
        # optimal_threshold = thresholds[optimal_idx]
        # print(optimal_threshold)
        # save_optimal_threshold(optimal_threshold, chosen_label,
        #                    algorithm, file_name_n_iter,
        #                    file_name_min_pos_instances)
        return (xgboostrf_y_pred >= 0.5).astype(int)
    

def predict_y_and_save(chosen_label:int, algorithm:str, file_name_n_iter:str, file_name_min_pos_instances:str,
              train_or_test:str):
    y_pred = predict_y(chosen_label,
                  algorithm,
                  file_name_n_iter,
                  file_name_min_pos_instances,
                  train_or_test)
    
    filename = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos_y_{train_or_test}_pred.txt"
    pd.Series(y_pred).to_csv(filename, index=False, header=False)
    print(filename + ' successfully saved')


# # %%time
# # chosen_label = 18
# # algorithm = 'xgboostrf'
# # file_name_n_iter = '200'
# # file_name_min_pos_instances = '50'

# # # for algorithm in ['svm', 'tree']:
# #     # , 'xgboostrf'
# # for chosen_label in [6]:
# # # for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
# #     a = predict_y(chosen_label,
# #             algorithm,
# #             file_name_n_iter,
# #             file_name_min_pos_instances,
# #             train_or_test='test')

# #     print(f"{chosen_label}: {a}")
# # # %%
