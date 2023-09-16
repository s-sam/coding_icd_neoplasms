import joblib
from xgboost import XGBRFClassifier

from step_3_text_preprocessing_and_tfidf import prepare_x_y_for_chosen_label
from step_4_hyperparameter_tuning import load_tfidf


# step 0 for the chosen label, select the random state - remove once this step is complete
import initialise_data_details
chosen_random_states = initialise_data_details.chosen_random_states


def load_best_hyperparameters(algorithm:str, chosen_label:int, file_name_n_iter:str, file_name_min_pos_instances:str):
    best_hyperparameters_file_path = f"label_{str(chosen_label)}/label_{str(chosen_label)}_hyperparam_tuning_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.pkl"
    best_hyperparameters = joblib.load(best_hyperparameters_file_path)
    return best_hyperparameters


def run_retrain_for_one_alg_one_label(algorithm:str, chosen_label:int, file_name_n_iter:str, file_name_min_pos_instances:str):
    best_hyperparameters_model = load_best_hyperparameters(algorithm, chosen_label, file_name_n_iter, file_name_min_pos_instances)
    
    # using the new clf, retrain the whole train set
    x_train = load_tfidf('train', chosen_label)
    _, y_train = prepare_x_y_for_chosen_label('train', chosen_label)
     
    clf_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos"
    
    if algorithm == 'svm' or algorithm == 'tree' or algorithm=='linear_only_svm':
        clf = best_hyperparameters_model.best_estimator_
        clf.fit(x_train, y_train)
        # save the final model
        joblib.dump(clf, f"{clf_file_name}.pkl")
        print(f"{clf_file_name}.pkl successfully saved")
    elif algorithm == 'xgboostrf':
        clf = XGBRFClassifier(**best_hyperparameters_model.best_params_).fit(x_train, y_train)
        clf.save_model(f"{clf_file_name}.txt")
        print(f"{clf_file_name}.txt successfully saved")
   

if __name__ == '__main__':
    
    for algorithm in ['svm', 'tree', 'xgboostrf']:
        for i in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
            
            algorithm = 'xgboostrf'
            i = 12
            
            file_name_n_iter_dict = {
                'svm': 200,
                'tree': 2500,
                'xgboostrf': 2500
                }
            best_hyperparams = load_best_hyperparameters(algorithm=algorithm,
                                    chosen_label=i,
                                    file_name_n_iter=file_name_n_iter_dict[algorithm],
                                    file_name_min_pos_instances=50).best_estimator_
            
            
            
            best_hyperparams.get_params()