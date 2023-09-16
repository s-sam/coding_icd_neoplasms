import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import load_npz
import shap
import xgboost as xgb

import initialise_data_details
from step_3_text_preprocessing_and_tfidf import load_txt_one_col_file, prepare_x_y_for_chosen_label
from step_4_hyperparameter_tuning import load_tfidf
from step_6_tfidf_test import load_vectoriser
from step_7_predict_y_using_final_model import load_final_model, load_tfidf_data


chosen_random_states = initialise_data_details.chosen_random_states
shap.initjs()


# shapley values
# for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    # file_name_min_pos_instances='50'
for chosen_label in [4,9,14]:
    file_name_min_pos_instances='5'
    algorithm='xgboostrf'
    file_name_n_iter='2500'


    final_model_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    xgboostrf_final_model = xgb.XGBRFClassifier()
    booster = xgb.Booster()
    booster.load_model(final_model_file_name)
    xgboostrf_final_model._Booster = booster

    vectoriser = load_vectoriser(chosen_label)
    xgboostrf_final_model.get_booster().feature_names = list(vectoriser.vocabulary_.keys())

    xgboostrf_x_train_tfidf = load_tfidf_data(chosen_label, 'train')

    shapley_values = xgboostrf_final_model.get_booster().predict(xgb.DMatrix(xgboostrf_x_train_tfidf, feature_names=vectoriser.vocabulary_.keys()),
                                                        pred_contribs=True)

    shapley_values_df = pd.DataFrame(data=shapley_values,
                        columns=list(vectoriser.vocabulary_.keys())+['_base'])

    # shapley_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.csv"
    # shapley_values_df.to_csv(shapley_file_name, index=False)
    # print(f"{shapley_file_name} successfully saved")
    
    from scipy.sparse import csr_matrix, save_npz, load_npz
    shapley_values_sparse = csr_matrix(shapley_values_df.values)
    shapley_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.npz"
    save_npz(shapley_file_name, shapley_values_sparse)
    print(shapley_file_name + ' successfully saved')
    

def return_xgboostrf_features_used_count(chosen_label:int, file_name_n_iter:int, file_name_min_pos_instances:int, algorithm='xgboostrf'):
    xgboostrf_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_features_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    xgboostrf_features_used = pd.read_csv(xgboostrf_features_file_name, header=None)
    # return xgboostrf_features_used
    return len(xgboostrf_features_used)


def save_shap_mean_abs_bar_chart(chosen_label:int, file_name_n_iter:int, file_name_min_pos_instances:int, algorithm='xgboostrf'):
    xgboostrf_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_mean_abs_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    shap_mean_abs = pd.read_csv(xgboostrf_features_file_name,header=None)
    shap_mean_abs_top_10 = shap_mean_abs.sort_values(by=1, ascending=False).iloc[0:10]
    shap_mean_abs_not_top_10 = shap_mean_abs.sort_values(by=1, ascending=False).iloc[10:, 1]
    shap_mean_abs_not_top_10_val_sum = shap_mean_abs_not_top_10.sum()
    
    actual_features_n = return_xgboostrf_features_used_count(chosen_label,
                                     file_name_n_iter,
                                     file_name_min_pos_instances)

    # print(len(shap_mean_abs))
    # print(len(shap_mean_abs_top_10))
    # print(len(shap_mean_abs_not_top_10))
    # print(actual_features_n)
    
    if len(shap_mean_abs) != actual_features_n:
        raise Exception("number of features coming through is not the same as being used in classifier")
    else:
        shap_mean_abs_top_10 = shap_mean_abs_top_10.sort_values(by=1, ascending=True)
        
        not_top_10_n = len(shap_mean_abs_not_top_10)
        shap_mean_abs_top_10_and_else = pd.concat([
            pd.DataFrame([[f"Sum of {not_top_10_n} other features", shap_mean_abs_not_top_10_val_sum]]),
            shap_mean_abs_top_10.copy()
        ])
        
        # return shap_mean_abs_top_10_and_else
        dpi=200
        plt.figure(dpi=dpi)
        shap_mean_abs_top_10_and_else.plot.barh(x=0, y=1, legend=False,
                                                                        xlabel='Mean absolute Shapley value',
                                                                        ylabel='',
                                                                        title='Feature importance')
        file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_mean_abs_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.eps"
        plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"{file_name} saved successfully")
        # return shap_mean_abs


def shap_summary_plot(chosen_label:int, file_name_n_iter:int, file_name_min_pos_instances:int, algorithm='xgboostrf'):
    xgboostrf_shap_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.npz"
    xgboostrf_shap = load_npz(xgboostrf_shap_file_name)
    # remove the last col, base, which represents the value if no required features exist
    xgboostrf_shap = xgboostrf_shap[:,:-1]
    # convert to np array
    xgboostrf_shap = xgboostrf_shap.toarray()

    x_train_tfidf = load_tfidf(train_or_test='train',
                            chosen_label=chosen_label)
    # convert to np array
    x_train_tfidf = x_train_tfidf.toarray()

    vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
        
    # print(xgboostrf_shap.shape, type(xgboostrf_shap),
    # x_train_tfidf.shape, type(x_train_tfidf),
    # len(vocab), type(vocab)
    # )

    shap.summary_plot(shap_values=xgboostrf_shap,
                features=x_train_tfidf,
                feature_names=vocab,
                max_display=10,
                show=False)
    plt.title('Shapley values for each feature for each study (ten features with highest importance only)')
    # plt.show()
    
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.eps"
    plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"{file_name} saved successfully")



chosen_label = 0
algorithm = 'xgboostrf'
file_name_n_iter = 2500

file_name_min_pos_instances = 50
for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
# file_name_min_pos_instances = 5
# for chosen_label in [4,14]: # 9 has no features
    # save_shap_mean_abs_bar_chart(chosen_label=chosen_label,
    #                         file_name_n_iter=file_name_n_iter,
    #                         file_name_min_pos_instances=file_name_min_pos_instances)

    shap_summary_plot(chosen_label,
                        file_name_n_iter,
                        file_name_min_pos_instances)
