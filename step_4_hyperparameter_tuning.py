import joblib
from scipy.sparse import load_npz
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBRFClassifier

from scipy.stats import uniform, randint

from step_2_train_test_split import create_chosen_label_col_name
from step_3_text_preprocessing_and_tfidf import prepare_x_y_for_chosen_label

    
def load_tfidf(train_or_test:str, chosen_label:int):
    """load tfidf npz data from already specified path. indicate whether train or test required and chosen label.

    Args:
        train_or_test (str): train or test required
        chosen_label (int): an integer representing the label in the label_dict.csv

    Returns:
        scipy.sparse._csr.csr_matrix: tfidf representation of chosen label documents
    """    
    chosen_label_col_name = create_chosen_label_col_name(chosen_label)
    x_filepath = chosen_label_col_name + '/' + chosen_label_col_name + '_x_' + train_or_test + '_tfidf.npz'
    x = load_npz(x_filepath)
    return x


def return_param_dists(algorithm:str) -> dict:
    """return pre-specified search parameter grids of hyperparameters for select algorithms: svm, tree and xgboostrf.

    Args:
        algorithm (str): svm, tree or xgboostrf required

    Raises:
        Exception: when algorithm string is not in list

    Returns:
        dict: search parameter grid of hyperparameters
    """    
    if algorithm=='svm':
        param_dists = {
            'C': uniform(1.0, 100.0-1.0), # 1-100
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': randint(1, 5), # inclusive of low and not inclusive of high so 1,2,3...,10
            'gamma': ['scale', 'auto'],
            'tol': uniform(0.00001, 1.0-0.00001)
        }
    elif algorithm=='tree':
        param_dists = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10), # int
            'min_samples_split': uniform(0.0, 1.0), #fraction of samples
            'min_samples_leaf': uniform(0.0, 1.0), #fraction of samples
            'max_features': uniform(0.0, 1.0), #fraction of samples
            'max_leaf_nodes': randint(2, 100), # int
            'min_impurity_decrease': uniform(0.0, 1.0), #is impurity bound between 0 and 1?
            'ccp_alpha': uniform(0.0, 1.1) #float, default=0.0
        }
    elif algorithm=='xgboostrf': # boosted
        param_dists = {
            'n_estimators': randint(20, 200), # supposedly, more trees mean better generalisation score
            'max_depth': randint(1, 7), # int
            'max_leaves': randint(2, 70), # int
            'learning_rate': uniform(0.01, 0.5-0.01), #==eta
            'booster': ['gbtree', 'gblinear', 'dart'],
            'min_split_loss': uniform(0.01, 10.0-0.01), # maybe 5 for upper limit? ==gamma
            'min_child_weight': randint(1, 100), # could try upper limit of 200
            'max_delta_step': randint(0, 11),
            'subsample': uniform(0.5, 1-0.5),
            'colsample_bytree': uniform(0.5, 0.8-0.5),
            'reg_lambda': uniform(1.0, 5.0-1.0),
            'reg_alpha': uniform(0.1, 5.0-0.1),
            'grow_policy': ['depthwise', 'lossguide'],
            # 'sample_type': ['uniform', 'weighted'],
            # 'normalize_type': ['tree', 'forest'],
            'rate_drop': uniform(0.000001, 0.9-0.000001),
            'feature_selector': ['cyclic', 'shuffle']
        }
    else:
        raise Exception("algorithm should be svm, tree or xgboostrf")
    return param_dists


def return_clf(algorithm:str, chosen_random_state:int, y_train=None):
    """based on algorithm chosen, a pre-specified class weighted classifier is returned

    Args:
        algorithm (str): svm, tree or xgboostrf required
        chosen_random_state (int): the random state associated with the chosen_label for repeatability
        y_train (pd.DataFrame()): this is required only for xgboostrf to calculate the class weights

    Raises:
        Exception: when algorithm string is not in list

    Returns:
        classifier
    """    
    if algorithm=='svm':
        clf = SVC(random_state=chosen_random_state,
                  class_weight='balanced',
                  shrinking=True)
            #    'max_iter':-1, - this does not need to be set because the tolerance stops it for us
            #    'decision_function_shape':'ovr', - i think this doesnt matter because there are only 2 classes but need to check this
    elif algorithm=='tree':
        clf = DecisionTreeClassifier(random_state=chosen_random_state,
                                     class_weight='balanced',)
    elif algorithm=='xgboostrf':
        len_y = len(y_train)
        pos_instances = sum(y_train)
        neg_instances = len_y - pos_instances
        clf = XGBRFClassifier(random_state=chosen_random_state,
                              n_jobs=6,
                              scale_pos_weight=neg_instances/pos_instances,
                              eval_metric='aucpr',
                              objective='binary:hinge',
                              seed=chosen_random_state,
                              tree_method = 'hist' # this used because scalable
                              )
    else:
        raise Exception("algorithm should be svm, tree or xgboostrf")
    return clf


def run_hyperparam_tuning_for_one_algorithm(algorithm:str, n_iter:int, x_train, y_train, chosen_random_state:int, verbose=1):
    """runs hyperparameter tuning for one specified algorithm and a specified x_train and y_train.

    Args:
        algorithm (str): svm, tree or xgboostrf required
        n_iter (int): number of iterations of the cross validation. n_iter=1 means that cross validation cycles through once.
        x_train (pd.DataFrame()): x_train in tfidf format
        y_train (pd.DataFrame()): actual labels for x_train
        chosen_random_state (int): the random state associated with the chosen_label for repeatability
        verbose (int, optional): verbose. Defaults to 1.

    Returns:
        RandomizedSearchCV(): returns the fitted hyperparameter tuning from x_train and y_train
    """    
    param_dists = return_param_dists(algorithm)
    clf = return_clf(algorithm, chosen_random_state, y_train=y_train)
    hyperparameter_tuning = RandomizedSearchCV(estimator=clf,
                                                   param_distributions=param_dists,
                                                   n_iter=n_iter,
                                                   scoring='f1',
                                                   n_jobs=6,
                                                   refit=True,
                                                   cv=StratifiedKFold(n_splits=5, shuffle=False),
                                                   verbose=verbose,
                                                   random_state=chosen_random_state,
                                                   error_score='raise',
                                                   return_train_score=False)
    hyperparameter_tuning.fit(x_train, y_train)
    return hyperparameter_tuning
    

def run_hyperparam_for_labels_and_algs(n_iter:int, chosen_labels:list, chosen_random_states:list, chosen_algs:list,
                                       min_pos_instances:int, min_pos_instances_action:str, verbose=1):
    """run hyperparameter tuning and save the files in a specific location with specific file name.

    Args:
        n_iter (int): number of iterations of the cross validation. n_iter=1 means that cross validation cycles through once
        chosen_labels (list): list of integer representations of the labels that are required. should match label_dict.csv
        chosen_random_states (list): list of random states associated with each of the chosen_label for repeatability
        chosen_algs (list): list of chosen algorithms, svm, tree or xgboostrf
        min_pos_instances (int): the training will only run for labels that have >= min_pos_instances in the training set
        min_pos_instances_action (str): for labels that have < min_pos_instances, what to do here. break or skip required. break raises an exception and skip skips to the next label.
        verbose (int, optional): verbose. Defaults to 1.

    Raises:
        Exception: an exception is raised if min_pos_instances_action=='break' and any labels have < min_pos_instances
    """    
    
    for chosen_label in chosen_labels:
        chosen_random_state = chosen_random_states[chosen_label]
        _, y_train = prepare_x_y_for_chosen_label('train', chosen_label)
        pos_instances = sum(y_train)
        
        if pos_instances < min_pos_instances:
            if min_pos_instances_action=='break':
                raise Exception(f"not enough pos_instances. there are {str(pos_instances)} and there needs be {str(min_pos_instances)} or more.")
            elif min_pos_instances_action=='skip':
                print(f"label {chosen_label} skipped. not enough pos_instances. there are {str(pos_instances)} and there needs be {str(min_pos_instances)} or more.")
            
        else:
            x_train = load_tfidf('train', chosen_label)
            
            for algorithm in chosen_algs:
                hyperparameter_tuning = run_hyperparam_tuning_for_one_algorithm(algorithm=algorithm,
                                                                                    n_iter=n_iter,
                                                                                    x_train=x_train,
                                                                                    y_train=y_train,
                                                                                    chosen_random_state=chosen_random_state,
                                                                                    verbose=verbose
                                                                                    )
                file_name= f"label_{str(chosen_label)}/label_{str(chosen_label)}_hyperparam_tuning_{algorithm}_{n_iter}n_iter_{min_pos_instances}min_pos.pkl"
                joblib.dump(hyperparameter_tuning, file_name)
                print(f"{file_name} successfully saved")

    print("run_hyperparam_for_labels_and_algs function successfully completed")


if __name__ == '__main__':
    import initialise_data_details
    chosen_random_states = initialise_data_details.chosen_random_states
    
    # # step 4 hyperparameter tuning, save the tuned hyperparameters
    # requires xgboost==1.0.2
    # for algorithm in ['svm', 'tree', 'xgboostrf']:
    # for algorithm in ['tree', 'xgboostrf']:
    for algorithm in ['svm', 'tree', 'xgboostrf']:
        for i in range(19):
            print(f"{algorithm}: {i}")
            run_hyperparam_for_labels_and_algs(n_iter=1,chosen_labels=[i],chosen_random_states=chosen_random_states,chosen_algs=[algorithm],min_pos_instances=50,min_pos_instances_action='skip',verbose=1)