if __name__ == "__main__":
    import initialise_data_details
    import step_1_clean_and_save_data
    import step_2_train_test_split
    import step_3_text_preprocessing_and_tfidf
    import step_4_hyperparameter_tuning
    import step_5_retrain_on_full_train_data
    import step_6_tfidf_test
    import step_7_predict_y_using_final_model
    import step_8_performance_results

    # step 0 for the chosen label, select the random state
    chosen_random_states = initialise_data_details.chosen_random_states
    ##################### TEXT PREPROCESSING AND FEATURE SELECTION

    # # step 1 clean and save data, save prepared_one_hot.csv and label_dict.csv
    # run_create_dataset = input("Do you want to clean and save the data? Enter y or n")
    # if run_create_dataset == 'y':
    #     step_1_clean_and_save_data.run(root='data/')
    # elif run_create_dataset != 'n':
    #     raise Exception("Please enter y or n")
    # else:
    #     pass


    # # step 2 save train test split study ids, save label_i_train_study_ids.txt and label_i_text_study_ids.txt
    # run_train_test_split_save_study_ids = input("Do you want to split the data into train and test and save the study ids? Enter y or n")
    # if run_train_test_split_save_study_ids == 'y':
    #     step_2_train_test_split.all_labels_split_train_test_and_save_study_ids(chosen_random_states)
    # elif run_train_test_split_save_study_ids != 'n':
    #     raise Exception("Please enter y or n")
    # else:
    #     pass


    # # step 3 train text preprocessing and save tfidf and vectoriser, label_i_tfidf_vectoriser.pkl and label_i_x_train_tfidf.npz
    # run_text_preprocessing_and_tfidf = input("Do you want to preprocess the text and save the vectoriser and the tfidf matrices?")
    # if run_text_preprocessing_and_tfidf == 'y':
    #     step_3_text_preprocessing_and_tfidf.all_labels_train_text_preprocessing_and_save_tfidf()
    # elif run_text_preprocessing_and_tfidf != 'n':
    #     raise Exception("Please enter y or n")
    # else:
    #     pass


    ########################## TRAINING #################################

    # step 4 hyperparameter tuning, save the tuned hyperparameters
    # requires xgboost==1.0.2
    for algorithm in ['tree', 'xgboostrf']:
        for i in [4,9,14]:
            step_4_hyperparameter_tuning.run_hyperparam_for_labels_and_algs(
                n_iter=2500,
                chosen_labels=[i],
                chosen_random_states=chosen_random_states,
                chosen_algs=[algorithm],
                min_pos_instances=5,
                min_pos_instances_action='skip',
                verbose=1)

    # step 5 retrain the model with best hyperparameters on the full train set, save the final model
    for algorithm in ['tree', 'linear_only_svm', 'xgboostrf']:
    # for algorithm in ['svm', 'tree', 'xgboostrf']:

        for i in [4,9,14]:
            step_5_retrain_on_full_train_data.run_retrain_for_one_alg_one_label(
            algorithm=algorithm, chosen_label=i, file_name_n_iter='2500',
            file_name_min_pos_instances='5')

    ################ TESTING ###############

    # step 6 use the vectoriser created with train data to tfidf the test set, save the tfidf test data
    # for i in range(19):
    #     step_6_tfidf_test.clean_and_tfidf_test_data(i)

    ## requires xgboost==1.7.6
    # # step 7 predict y using final model
    for algorithm in ['xgboostrf']:
    # for algorithm in ['linear_only_svm', 'tree', 'xgboostrf']:
        for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
        # for chosen_label in [4,9,14]:
            step_7_predict_y_using_final_model.predict_y_and_save(chosen_label,
            algorithm=algorithm,
            file_name_n_iter='2500',
            file_name_min_pos_instances='50',
            # file_name_min_pos_instances='5',
            train_or_test='test')


    # # # step 8 save performance results, f1 score and confusion matrix terms in a csv
    chosen_labels = [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]
    file_name_min_pos_instances = '50'
    # chosen_labels = [4,9,14]
    # file_name_min_pos_instances = '5'
    file_name_n_iter = '2500'
    algorithms = ['xgboostrf', 'linear_only_svm', 'tree']
    step_8_performance_results.compute_f1_score_and_conf_matrix_for_multiple(chosen_labels,
                                                  algorithms,
                                                  file_name_n_iter,
                                                  file_name_min_pos_instances,
                                                  train_or_test='test')

