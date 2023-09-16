import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from step_3_text_preprocessing_and_tfidf import load_txt_one_col_file, prepare_x_y_for_chosen_label


def load_y_pred_file(chosen_label:int, algorithm:str, file_name_n_iter:str, file_name_min_pos_instances:str,
              train_or_test:str):
    y_pred_filename = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos_y_{train_or_test}_pred.txt"
    y_pred = load_txt_one_col_file(y_pred_filename)
    return y_pred


def compute_f1_score_and_conf_matrix(chosen_label:int, algorithm:str, file_name_n_iter:str, file_name_min_pos_instances:str,
              train_or_test:str):
    y_pred = load_y_pred_file(chosen_label, algorithm, file_name_n_iter, file_name_min_pos_instances, train_or_test)
    _, y_actual = prepare_x_y_for_chosen_label(train_or_test, chosen_label)
    
    f1_returned = f1_score(y_actual, y_pred)
    
    f1_record_info = [[chosen_label, algorithm, 'f1', 'n/a', 'n/a', f1_returned]]
    
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    conf_matrix_record_info = ([
    [chosen_label, algorithm, 'confusion_matrix', 0, 0, tn],
    [chosen_label, algorithm, 'confusion_matrix', 1, 1, tp],
    [chosen_label, algorithm, 'confusion_matrix', 1, 0, fn],
    [chosen_label, algorithm, 'confusion_matrix', 0, 1, fp],  
    ])
    
    record_info = f1_record_info + conf_matrix_record_info
    
    return record_info


def compute_f1_score_and_conf_matrix_for_multiple(chosen_labels:list, algorithms:list, file_name_n_iter:str, file_name_min_pos_instances:str,
              train_or_test:str):    
    concat_record_info = []
    for chosen_label in chosen_labels:
        for algorithm in algorithms:
            concat_record_info += compute_f1_score_and_conf_matrix(chosen_label,
                  algorithm,
                  file_name_n_iter,
                  file_name_min_pos_instances,
                  train_or_test)
            
    record_cols = ['label', 'model', 'measure_group', 'actual', 'predicted', 'value']
    performance_results = pd.DataFrame(concat_record_info, columns=record_cols)
    performance_results.to_csv('performance_results.csv', index=False)
    print('performance_results.csv successfully saved')


if __name__ == '__main__':
    chosen_label = 18
    algorithm = 'xgboostrf'
    file_name_n_iter = '10'
    file_name_min_pos_instances = '50'

    load_y_pred_file(chosen_label,
                    algorithm,
                    file_name_n_iter,
                    file_name_min_pos_instances,
                    train_or_test='test')

    compute_f1_score_and_conf_matrix(chosen_label,
                    algorithm,
                    file_name_n_iter,
                    file_name_min_pos_instances,
                    train_or_test='test')


    compute_f1_score_and_conf_matrix_for_multiple(chosen_labels=[18,18],
                    algorithms=['xgboostrf'],
                    file_name_n_iter=file_name_n_iter,
                    file_name_min_pos_instances=file_name_min_pos_instances,
                    train_or_test='test')



    import pandas as pd
    a = pd.concat([pd.read_csv('performance_results_over_50_min_pos.csv'),
                pd.read_csv('performance_results_under_50_min_pos.csv')])

    a.head()

    def return_f1():
        return a.loc[a.loc[:,'measure_group']=='f1'].pivot_table(index='label',
                    columns='model',
                    values='value')

    def return_actual_n(pos:bool):
        return (a.loc[(a.loc[:,'measure_group']=='confusion_matrix')&
                (a.loc[:,'actual']==pos)]
        .pivot_table(index='label', columns='model', values='value', aggfunc='sum')
        ).mean(axis=1)

    def return_label_accuracy(pos:bool):
        label_accuracy_count = (a.loc[(a.loc[:,'measure_group']=='confusion_matrix')&
            (a.loc[:,'actual']==pos)&
            (a.loc[:,'predicted']==pos)]
        .pivot_table(index='label',
                    columns='model',
                    values='value')
        )
        
        label_accuracy_count['actual_n'] = return_actual_n(pos)
        label_accuracy = pd.DataFrame()
        label_accuracy['actual_n'] = label_accuracy_count.loc[:,'actual_n']
        for i in label_accuracy_count.columns[0:3]:
            label_accuracy[i] = label_accuracy_count.loc[:,i]/label_accuracy_count.loc[:,'actual_n']
        return label_accuracy

    return_label_accuracy(pos=True).to_csv('accuracy_pos_label.csv')
    return_label_accuracy(pos=False).to_csv('accuracy_neg_label.csv')
