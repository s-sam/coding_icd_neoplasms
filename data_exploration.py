from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.tree import plot_tree, export_graphviz
from scipy.sparse import load_npz
import xgboost as xgb

from load_data import load_tables
import clean_icd_labels
import clean_study_text
from step_1_clean_and_save_data import save_as_csv
from step_3_text_preprocessing_and_tfidf import load_study_ids, prepare_x_y_for_chosen_label, load_txt_one_col_file
from step_5_retrain_on_full_train_data import load_best_hyperparameters
from step_6_tfidf_test import load_vectoriser
from step_7_predict_y_using_final_model import load_final_model

# load tables
data_folder = 'data/'
study, icd_labels, icd_reviewed = load_tables(root=data_folder,
                                       tables=['study', 'icd_labels', 'icd_reviewed'])


# studies to start with
print(study.loc[:,'study_id'].nunique())
# check how many have blank text
text_cols = ['short_name', 'title', 'research_summary', 'inclusion_criteria']
print(study.loc[:,text_cols].astype(str).agg(''.join, axis=1).isna().sum())
# of these studies, how many have been reviewed
reviewed_studies = pd.merge(study.loc[:,'study_id'], icd_reviewed.loc[:,'study_id'].drop_duplicates())
print(len(reviewed_studies))
# of these studies, how many have at least one ICD-10 code
code_exists = pd.merge(reviewed_studies, icd_labels.loc[:,'study_id'].drop_duplicates(), how='left', indicator=True)
print(code_exists.groupby('_merge').count())
# have a look at studies with no icd_labels codes
studies_with_no_icd_codes = code_exists.loc[code_exists.loc[:,'_merge']=='left_only'].loc[:,'study_id']
study_text_with_no_icd_codes = pd.merge(studies_with_no_icd_codes, study, how='inner')
study_text_with_no_icd_codes.head()

# study count by neoplasms icd_labels blocks
labels_dict, icd_clean = clean_icd_labels.CleanData(icd_labels, 'Neoplasms').clean_labels()
icd_clean.loc[:,'study_id'].nunique()
all_studies = pd.merge(reviewed_studies, icd_clean, how='left').fillna('-1')
reporting_table = all_studies.groupby('label').count()
reporting_table["ratio to else, 1:"] = ((len(reviewed_studies)/reporting_table.loc[:,'study_id'])-1)
reporting_table["ratio int"] = reporting_table.loc[:,"ratio to else, 1:"].round(0).astype(int)
reporting_table = reporting_table.sort_values(by="ratio to else, 1:", ascending=False)
reporting_table = reporting_table.rename(index=labels_dict)

reporting_table.iloc[0:18].median()

# study example with mixed digits and characters
text_prepared = clean_study_text.CleanData(study_data=study).run()

####### negation stats start
prepared = pd.read_csv('prepared.csv')
prepared = prepared.drop(columns=['Unnamed: 0', 'label']).drop_duplicates()
len(prepared)

prepared['concat'] = prepared.loc[:,['short_name','title', 'research_summary', 'inclusion_criteria']].astype(str).agg('. '.join, axis=1)

def str_contain_count(study, col, search_string):
    str_contain_true_or_false = study.loc[:, col].str.contains(search_string)
    examples = study.loc[str_contain_true_or_false]
    one_example = examples.loc[:,'concat'].sample(n=1, random_state=55).values[0]
    count = str_contain_true_or_false.sum()
    percent = count / len(study)*100
    return [[f"'{search_string}'", count, percent, one_example]]


def save_stats(study, str_col, search_string_list):
    stats = pd.DataFrame()
    for search_string in search_string_list:
        search_string_stats = str_contain_count(study=study,
                                col=str_col,
                                search_string=search_string)
        search_string_stats_df = pd.DataFrame(search_string_stats,
                                                columns=['search_string', 'study_count', 'percent_of_studies', 'example_text'])
        stats = pd.concat([stats, search_string_stats_df])
    # return stats
    stats.to_csv('negation_statistics.csv', index=False)

search_string_list = [' no ',
                ' not ',
                ' without ',
                ' negative ',
                ' exclu']
save_stats(prepared, str_col='concat', search_string_list=search_string_list)
######## negation stats end


#### text lengths boxplot
prepared = pd.read_csv('prepared.csv')
prepared = prepared.drop(columns=['Unnamed: 0', 'label']).drop_duplicates()
prepared = prepared.fillna('###NULL###')
len(prepared)

prepared_len = pd.DataFrame()
for col in prepared.columns:
    prepared_len[f"{col}_len"] = prepared.loc[:,col].astype(str).str.strip().str.replace('###NULL###', '').str.len()

prepared_len = prepared_len.drop(columns=['study_id_len'])

prepared_len.boxplot(vert=False)
file_name = "len_text_fields.eps"
plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
print(f"{file_name} saved successfully")
plt.close()
# plt.show()

prepared_len.loc[:,['short_name_len', 'title_len']].boxplot(vert=False)
file_name = "len_short_name_title.eps"
plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
print(f"{file_name} saved successfully")
plt.close()
# plt.show()

def jitter_scatter_box_plot(df):
    vals, names, xs = [],[],[]
    
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))
    
    plt.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.show()

jitter_scatter_box_plot(prepared_len.loc[:,['short_name_len']])
jitter_scatter_box_plot(prepared_len.loc[:,['research_summary_len']])


# for col in prepared_len.columns:
#     print((prepared_len.loc[:, col] == 0).sum())
    
# for col in prepared_len.columns:
#     print((prepared_len.loc[:, col] > 0).sum())

# for col in prepared_len.columns:
#     print(pd.DataFrame(prepared_len.loc[prepared_len.loc[:,col]>0, col].describe()).T)

# for col in prepared_len.columns:
#     print(prepared_len.loc[prepared_len.loc[:,col]>0, col].quantile(0.05),
#           prepared_len.loc[prepared_len.loc[:,col]>0, col].quantile(0.95)
#           )

# for col in prepared_len.columns:
#     plt.figure()
#     prepared_len.loc[prepared_len.loc[:,col]>0, col].hist(bins=10)
#     plt.show()
    
    
    
    
prepared_len_w_label = pd.read_csv('prepared.csv')
prepared_len_w_label = prepared_len_w_label.drop(columns=['Unnamed: 0']).drop_duplicates()
prepared_len_w_label = prepared_len_w_label.fillna('###NULL###')

prepared_len_w_label.groupby('label')['study_id'].count()

for col in prepared_len_w_label.columns:
    if col == 'label' or col == 'study_id':
        pass
    else:
        prepared_len_w_label[f"{col}_len"] = prepared_len_w_label.loc[:,col].astype(str).str.strip().str.replace('###NULL###', '').str.len()

prepared_len_w_label = prepared_len_w_label.drop(columns=['study_id', 'short_name', 'title', 'research_summary', 'inclusion_criteria'])

for col in prepared_len_w_label.columns:
    if col=='label':
        pass
    else:
        prepared_len_w_label.loc[prepared_len_w_label.loc[:,col]>0, ['label', col]].groupby('label').median().to_csv(f"{col}.csv")
        # prepared_len_w_label.loc[prepared_len_w_label.loc[:,col]==0, ['label', col]].groupby('label').count().to_csv(f"{col}.csv")
    
    
(prepared_len_w_label==0).sum()
study['go_live_date'] = pd.to_datetime(study.go_live_date)
study['go_live_date'].describe()
study.query("go_live_date=='2016-03-31'").study_id.nunique()

studies = pd.merge(left=prepared.loc[:,'study_id'],
                            right=study,
                            how='outer',
                            on='study_id',
                            indicator=True
                            )
studies['go_live_date'] = pd.to_datetime(studies.go_live_date)

study_dates = studies.loc[:,['study_id', 'go_live_date', '_merge']].drop_duplicates()
study_dates.groupby('_merge')['go_live_date'].describe()
study_dates = study_dates.loc[study_dates.loc[:, '_merge']!='left_only']
study_dates['_merge'] = study_dates.loc[:, '_merge'].astype(str)

study_dates['go_live_year'] = study_dates.loc[:,'go_live_date'].dt.strftime('%Y')
study_dates = study_dates.rename(columns={'_merge':'Review Status',
                                          'go_live_year': 'Go Live Year'})
study_dates = study_dates.replace('both', 'Reviewed').replace('right_only', 'Unreviewed')

study_dates.pivot_table(values='study_id', aggfunc='count',
                        index='Go Live Year',
                        columns='Review Status').plot.bar()

study_dates.pivot_table(values='study_id', aggfunc='count',
                        index='Go Live Year',
                        columns='Review Status')

studies.loc[studies.loc[:,'_merge']=='both',['study_id', 'go_live_date', '_merge']].drop_duplicates().sort_values(by='go_live_date', ascending=True).groupby('go_live_date')['study_id'].count()

studies.groupby('_merge')['go_live_date'].describe()

studies.to_csv('check.csv')
pd.to_datetime(studies_included.go_live_date).describe()

prepared.count()

studies_included.count()

studies_included.study_id.nunique() - pd.merge(studies_included, icd_labels, how='inner').study_id.nunique()

prepared.columns

## counts how many docs have a word
word_counts = Counter()
for i in text_prepared.loc[:,'text'].str.split(' '):
    unique_words_only = set(i)
    word_counts.update(unique_words_only)
    # if i == 2:
    #     break

words_w_digits = {}
for word, count in word_counts.items():
    if re.search('[0-9]', word):
        words_w_digits.update({word: count})
    # break

pd.DataFrame.from_dict(words_w_digits, orient='index').sort_values(0, ascending=False).iloc[0:20]

word_counts

# return train_n and test_n
for i in range(19):
    print(f"train: {len(load_study_ids(train_or_test='train', chosen_label=i))}")
    print(f"test: {len(load_study_ids(train_or_test='test', chosen_label=i))}")


# return pos class n
for i in range(19):
    _, y_train = prepare_x_y_for_chosen_label('train', i)
    __, y_test = prepare_x_y_for_chosen_label('test', i)
    print(f"pos: {sum(y_train)+sum(y_test)}")


def save_best_hyperparams():
    cols = ['label_index',
            'model',
            'optimise_performance_metric',
            'best_hyperparameters',
            'xgboostrf_optimal_threshold',
            'vocab_size']
    best_hyperparams_df = pd.DataFrame(columns=cols)

    # for algorithm in ['linear_only_svm', 'tree', 'xgboostrf']:
    #     for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    for algorithm in ['linear_only_svm', 'tree', 'xgboostrf']:
        for chosen_label in [4,9,14]:
            # algorithm = 'xgboostrf'
            # chosen_label = 12
            performance_metric = 'f1'
            xgboostrf_optimal_threshold = '-'
            
            file_name_n_iter_dict = {
                'svm': 200,
                'tree': 2500,
                'xgboostrf': 2500,
                'linear_only_svm': 2500
                }
            
            file_name_n_iter = file_name_n_iter_dict[algorithm]
        
            # file_name_min_pos_instances = 50
            file_name_min_pos_instances = 5   
            
            # hyperparams
            best_hyperparams = load_best_hyperparameters(algorithm=algorithm,
                                    chosen_label= chosen_label,
                                    file_name_n_iter=file_name_n_iter,
                                    file_name_min_pos_instances=file_name_min_pos_instances).best_estimator_.get_params()
            
            # xgboostrf optimal threshold
            if algorithm=='xgboostrf':
                file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_optimal_threshold_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"        
                with open(file_name, 'r') as file_contents:
                    xgboostrf_optimal_threshold = file_contents.read()
            
            # vocab size
            vocab_size = len(load_vectoriser(chosen_label).vocabulary_.keys())
            
            # save
            best_hyperparams_df = pd.concat([best_hyperparams_df,
                    pd.DataFrame([[
                        chosen_label,
                        algorithm,
                        performance_metric,
                        best_hyperparams,
                        xgboostrf_optimal_threshold,
                        vocab_size
                        ]],
                        columns=cols)])
            
        # return best_hyperparams_df
        # best_hyperparams_df = pd.melt(best_hyperparams_df, id_vars=['label_index', 'model'])
        
        # best_hyperparams_df.to_csv('best_hyperparams.csv', index=False)
        best_hyperparams_df.to_csv('best_hyperparams2.csv', index=False)

# save_best_hyperparams()

# a = pd.read_csv('best_hyperparams2.csv')
# b = pd.read_csv('best_hyperparams.csv')
# pd.concat([a,b]).to_csv('best_hyper.csv', index=False)

# tree diagrams and features
algorithm = 'tree'
for chosen_label in [1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    file_name_min_pos_instances = 50

# for chosen_label in [4,9,14]:
#     file_name_min_pos_instances = 5
    
    # algorithm = 'tree'
    # chosen_label = 12
    
    file_name_n_iter_dict = {
        'svm': 200,
        'tree': 2500,
        'xgboostrf': 2500
        }
    
    file_name_n_iter = file_name_n_iter_dict[algorithm]
    
    final_model = load_final_model(chosen_label,
                                    algorithm,
                                    file_name_n_iter,
                                    file_name_min_pos_instances)
    
    vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
    
    # save tree plot
    plt.figure()
    plot_tree(final_model, feature_names=vocab)
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_tree_plot_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.eps"
    plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
    print(f"{file_name} saved successfully")
    
    
    # save the feature names and their importances
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    pd.Series(final_model.feature_importances_,
                index=vocab).sort_values(ascending=False).replace(0, np.nan).dropna().to_csv(file_name, header=False)
    print(f"{file_name} saved successfully")
    
    
    chosen_label
    
    file_name
        

# xgboostrf diagrams
# for algorithm in ['svm', 'tree', 'xgboostrf']:
    for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    # for chosen_label in [4,9,14]:        
        algorithm = 'xgboostrf'
        file_name_n_iter_dict = {
            'svm': 200,
            'tree': 2500,
            'xgboostrf': 2500,
            'linear_only_svm': 2500,
            }
        file_name_n_iter = file_name_n_iter_dict[algorithm]
        file_name_min_pos_instances = 50
        # file_name_min_pos_instances = 5

        # open the final model
        if algorithm == 'xgboostrf':
            final_model_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_final_model_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
            xgboostrf_final_model = xgb.XGBRFClassifier()
            booster = xgb.Booster()
            booster.load_model(final_model_file_name)
            xgboostrf_final_model._Booster = booster
xgboostrf_final_model.get_booster().get_score(importance_type='weight')


            # save the feature map in the txt format required
            # vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
            # feature_map = pd.DataFrame(data=vocab)
            # feature_map[1] = ['i']*len(vocab)
            feature_map_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_map_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
            # feature_map.to_csv(feature_map_file_name, header=False, index=True,
            #                    sep='\t')
            # print(f"{feature_map_file_name} successfully saved")
            
            # plot the xth tree in the random forest
            for x in range(100):
                xgb.plot_tree(xgboostrf_final_model,
                            fmap=feature_map_file_name,
                            num_trees=x)
            
            # save the first tree
            # image = xgb.to_graphviz(xgboostrf_final_model, fmap=feature_map_file_name)
            # image_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_forest_plot_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos"
            # image.render(image_file_name, format='svg')
            # print(f"{image_file_name}.svg successfully saved")
            

    # xgboostrf features used
            features = pd.DataFrame(xgboostrf_final_model._Booster.get_score(fmap=feature_map_file_name).keys())
            features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_features_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos"
            
            save_as_csv(dataframe=features, dictionary=None, filename=features_file_name, file_type='txt', index=False, header=False)
        

# save features csv for tree and for xgboostrf        
        # xgboostrf optimal threshold
#         if algorithm=='xgboostrf':
#             file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_optimal_threshold_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"        
#             with open(file_name, 'r') as file_contents:
#                 xgboostrf_optimal_threshold = file_contents.read()
        
#         # xgboostrf mean shapley values not 0 - save
            file_name_n_iter = file_name_n_iter_dict[algorithm]
            xgboostrf_shap_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.npz"

            xgboostrf_shap = load_npz(xgboostrf_shap_file_name)
            vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
            xgboostrf_shap_df = pd.DataFrame(xgboostrf_shap.todense(), columns=vocab+['_base'])
                
            mean_shapley_values_df = xgboostrf_shap_df.mean(axis=0)
            mean_shapley_values_df_not_zero = mean_shapley_values_df[mean_shapley_values_df!=0]

            xgboostrf_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_mean_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
            mean_shapley_values_df_not_zero.to_csv(xgboostrf_features_file_name, header=None)
            print(f"{xgboostrf_features_file_name} successfully saved")
        
        
chosen_label=18        
# for chosen_label in [0,1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    for chosen_label in [4,9,14]:        
        # file_name_min_pos_instances = 50
        file_name_min_pos_instances = 5
        algorithm = 'xgboostrf'
        file_name_n_iter = 2500
        
#         # xgboostrf mean ABSOLUTE shapley values not 0 - save
        xgboostrf_shap_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_values_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.npz"

        xgboostrf_shap = load_npz(xgboostrf_shap_file_name)
        # remove the last col, base, which represents the value if no required features exist
        xgboostrf_shap = xgboostrf_shap[:,:-1]
        # convert to abs
        xgboostrf_shap_abs = abs(xgboostrf_shap)
        
        vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
        xgboostrf_shap_abs_df = pd.DataFrame(xgboostrf_shap_abs.todense(), columns=vocab)
            
        mean_abs_shapley_values_df = xgboostrf_shap_abs_df.mean(axis=0)
        mean_abs_shapley_values_df_not_zero = mean_abs_shapley_values_df[mean_abs_shapley_values_df!=0]

        xgboostrf_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_mean_abs_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
        mean_abs_shapley_values_df_not_zero.to_csv(xgboostrf_features_file_name, header=None)
        print(f"{xgboostrf_features_file_name} successfully saved")
        

#         # tree feature importances
#         if algorithm=='tree':
#             tree_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
#             tree_features_df = pd.read_csv(tree_features_file_name, header=None)
#             tree_features_df


# # save svm linear models feature weights
        if algorithm == 'svm':
            final_model = load_final_model(chosen_label,
                                       algorithm,
                                       file_name_n_iter,
                                       file_name_min_pos_instances)
            kernel = final_model.get_params()['kernel']
            if kernel == 'linear':
                vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
                feat_importances = pd.DataFrame(final_model.coef_.toarray(), columns=vocab).T
                feat_importances_not_zero = feat_importances.loc[feat_importances.loc[:,0]!=0]
                feat_importances_not_zero = feat_importances_not_zero.sort_values(by=0, ascending=False)
                svc_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"                
                feat_importances_not_zero.to_csv(svc_features_file_name, header=False)
                print(chosen_label, feat_importances_not_zero)
            else:
                print(f"kernel: {kernel}")
                
                
# save linear_only_svm linear models feature weights
    for chosen_label in [4,9,14]:
        file_name_min_pos_instances=5
        algorithm='linear_only_svm'
        if algorithm == 'linear_only_svm':
            final_model = load_final_model(chosen_label,
                                       algorithm,
                                       file_name_n_iter,
                                       file_name_min_pos_instances)
            vocab = list(load_vectoriser(chosen_label).vocabulary_.keys())
            feat_importances = pd.DataFrame(final_model.coef_, columns=vocab).T
            feat_importances_not_zero = feat_importances.loc[feat_importances.loc[:,0]!=0]
            feat_importances_not_zero = feat_importances_not_zero.sort_values(by=0, ascending=False)
            svc_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"                
            feat_importances_not_zero.to_csv(svc_features_file_name, header=False)
            print(f"{svc_features_file_name} successfully saved")
            print(chosen_label, feat_importances_not_zero)


def open_feat_importances(chosen_label:str, file_name_min_pos_instances:str):
    file_name_n_iter_dict = {
        'svm': 200,
        'tree': 2500,
        'xgboostrf': 2500,
        'linear_only_svm': 2500
        }
    
    algorithm='linear_only_svm'
    file_name_n_iter = file_name_n_iter_dict[algorithm]
    svc_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    svc_features = pd.read_csv(svc_features_file_name, header=None)
    svc_features[2] = algorithm
    svc_features[3] = 'weight'
    
    algorithm='tree'
    file_name_n_iter = file_name_n_iter_dict[algorithm]
    tree_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    import os
    if os.stat(tree_features_file_name).st_size!=0:
        tree_features = pd.read_csv(tree_features_file_name, header=None)
        tree_features[2] = algorithm
        tree_features[3] = 'split_value_based_on_splitting_criteria'
    else:
        tree_features=pd.DataFrame(data=[['*no features are used in the classifier',
                                        np.nan,
                                        algorithm,
                                        'split_value_based_on_splitting_criteria']],
                                   columns=[0,1,2,3])
    
    algorithm='xgboostrf'
    file_name_n_iter = file_name_n_iter_dict[algorithm]
    xgboostrf_features_file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_shapley_mean_abs_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.txt"
    if os.stat(xgboostrf_features_file_name).st_size!=0:
        xgboostrf_features = pd.read_csv(xgboostrf_features_file_name, header=None)
        xgboostrf_features[2] = algorithm
        xgboostrf_features[3] = 'mean_absolute_shapley_value'
    else:
        xgboostrf_features=pd.DataFrame(data=[['*no features are used in the classifier',
                                        np.nan,
                                        algorithm,
                                        'mean_absolute_shapley_value']],
                                   columns=[0,1,2,3])    
    
    concat = (pd.concat([svc_features, tree_features, xgboostrf_features])
              .rename(columns={0:'feature_name',
                               1:'value',
                               2:'algorithm',
                               3:'value_type'})
    )
    
    label_dict = pd.read_csv('label_dict.csv', index_col=0)
    concat['label'] = label_dict.iloc[chosen_label].values[0]
    concat['label_index'] = chosen_label
    return concat


## save feat importances
concat = pd.DataFrame()
for chosen_label in [1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    file_name_min_pos_instances = 50
# for chosen_label in [4,9,14]:    
#     file_name_min_pos_instances = 5
        
    concat = pd.concat([concat,open_feat_importances(chosen_label=chosen_label, file_name_min_pos_instances=file_name_min_pos_instances)])

concat.drop_duplicates()
concat.to_csv('feature_importances.csv', index=False)        
open_feat_importances(chosen_label=14, file_name_min_pos_instances=5)
open_feat_importances(chosen_label=6, file_name_min_pos_instances=50)

concat.groupby(['algorithm','label'])['value_type'].nunique()
# .round(decimals=5)
concat = pd.read_csv('feature_importances.csv')
concat.pivot_table(values='value',
              columns='algorithm',
              index='label_index',
              aggfunc='count')

def return_top_n_pos_neg(df, value_col_name, n:int, pos_or_neg:str):
    if pos_or_neg == 'pos':
        return df.loc[df.loc[:,value_col_name]>0].sort_values(by=value_col_name, ascending=False).iloc[0:n,:]
    elif pos_or_neg == 'neg':
        return df.loc[df.loc[:,value_col_name]<0].sort_values(by=value_col_name, ascending=True).iloc[0:n, :]
    else:
        raise Exception("pos_or_neg should be 'pos' or 'neg")


def return_top_n_abs(df, value_col_name, n:int):
    top_n_abs_loc = df.loc[:,value_col_name].abs().sort_values(ascending=False).iloc[0:n].index
    return df.loc[top_n_abs_loc].sort_values(by=value_col_name, ascending=False)

        
return_top_n_abs(concat, 'value', 10)

def return_count_pos_neg(df, value_col_name, pos_or_neg:str):
    if pos_or_neg == 'pos':
        return len(df.loc[df.loc[:,value_col_name]>0])
    elif pos_or_neg == 'neg':
        return len(df.loc[df.loc[:,value_col_name]<0])
    else:
        raise Exception("pos_or_neg should be 'pos' or 'neg")

svc_features = concat.loc[concat.loc[:,'algorithm']=='linear_only_svm']
svc_features_pos = return_top_n_pos_neg(svc_features, 1, n=10, pos_or_neg='pos')
svc_features_neg = return_top_n_pos_neg(svc_features, 1, n=10, pos_or_neg='neg')

return_count_pos_neg(svc_features, 1, pos_or_neg='neg')
svc_features.plot.bar()

def concat_performance_files():
    files=['performance_results_linear_only_svm',
           'performance_results_tree_xgboost',
           'performance_results_under_50_min_pos']
    concat_df = pd.DataFrame()
    
    for file in files:
        file_name = file + '.csv'
        df = pd.read_csv(file_name)
        concat_df = pd.concat([concat_df, df])
    
    concat_df.to_csv(index=False, path_or_buf='performance_results_concat.csv')
    
# concat_performance_files()


a = pd.read_csv('performance_results_concat.csv')

a.loc[a.loc[:,'measure_group']=='f1', ['label', 'model', 'value']].pivot_table(values='value',
                                                                               index='label',
                                                                               columns='model')


def feature_chart_top5_bottom5(algorithm:str, chosen_label:int):
    alg_chosen_label = concat.loc[(concat.loc[:,'algorithm']==algorithm)&(concat.loc[:,'label_index']==chosen_label),
                                  ['feature_name', 'value', 'label']]
    alg_chosen_label = alg_chosen_label
    alg_chosen_label = alg_chosen_label.loc[alg_chosen_label.loc[:,'feature_name']!='_base']
    
    if len(alg_chosen_label) > 10:
        top_5 = return_top_n_pos_neg(alg_chosen_label, 'value', n=5, pos_or_neg='pos')
        bottom_5 = return_top_n_pos_neg(alg_chosen_label, 'value', n=5, pos_or_neg='neg')
        top_5_bottom_5 = pd.concat([bottom_5, top_5])
        
        show_on_chart = top_5_bottom_5.copy().sort_values(by='value', ascending=True)
    else:
        show_on_chart = alg_chosen_label.copy().sort_values(by='value', ascending=True)
        
    show_on_chart.plot.barh(x='feature_name', y='value',
                             color=(show_on_chart.loc[:,'value']>0).map({True:'green', False:'darkorange'}))
     
    return alg_chosen_label


def feature_chart_top_abs_n(algorithm:str, chosen_label:int, n:int):
    alg_chosen_label = concat.loc[(concat.loc[:,'algorithm']==algorithm)&(concat.loc[:,'label_index']==chosen_label),
                                  ['feature_name', 'value', 'label']]
    alg_chosen_label = alg_chosen_label
    alg_chosen_label = alg_chosen_label.loc[alg_chosen_label.loc[:,'feature_name']!='_base']
    
    top_abs_10 = return_top_n_abs(alg_chosen_label, 'value', n).sort_values(by='value', ascending=True)
    top_abs_10_indexes = top_abs_10.index
    
    else_group = alg_chosen_label.loc[~alg_chosen_label.index.isin(top_abs_10_indexes)]
    else_group_pos = else_group.loc[else_group.loc[:,'value']>0, 'value']
    else_group_pos_count = len(else_group_pos)
    else_group_pos_sum = else_group_pos.sum()
    else_group_pos_row = pd.DataFrame([[f"Sum of {else_group_pos_count} other features with positive importance",
                                        else_group_pos_sum,
                                        '_']],
                                      columns=top_abs_10.columns)

    else_group_neg = else_group.loc[else_group.loc[:,'value']<0, 'value']
    else_group_neg_count = len(else_group_neg)
    else_group_neg_sum = else_group_neg.sum()
    
    label = alg_chosen_label.loc[:,'label'].unique()[0]
    else_group_neg_row = pd.DataFrame([[f"Sum of {else_group_neg_count} other features with negative importance",
                                        else_group_neg_sum,
                                        label]],
                                      columns=top_abs_10.columns)
    
    show_on_chart = pd.concat([else_group_pos_row, else_group_neg_row, top_abs_10])
    
    show_on_chart.plot.barh(x='feature_name', y='value',
                            fontsize=9,
                            legend=False,
                            xlabel='Feature weight',
                            ylabel='Feature name',
                            title=f"Linear SVC feature weights for {label}",
                            color=(show_on_chart.loc[:,'value']>0).map({True:'green', False:'darkorange'}))
    # return show_on_chart


##### save linear svc feature weight charts
algorithm='linear_only_svm'
file_name_n_iter=2500
for chosen_label in [1,3,5,6,7,8,10,11,12,13,15,16,17,18]:
    file_name_min_pos_instances = 50
# for chosen_label in [4,9,14]:    
#     file_name_min_pos_instances = 5
    if chosen_label == 2 or chosen_label == 0:
        pass
    else:
        feature_chart_top_abs_n(algorithm=algorithm, chosen_label=chosen_label, n=20)
        file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_feature_importance_{algorithm}_{file_name_n_iter}n_iter_{file_name_min_pos_instances}min_pos.eps"
        plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
        

#### best hyperparameters save
a = pd.read_csv('best_hyperparameters_cocat.csv')
b = pd.read_csv('label_dict.csv')
b.columns = ['label_index', 'label']
a = pd.merge(a,b, how='inner')

a.pivot_table(values='best_hyperparameters',
              index=['label_index', 'label'],
              columns='model',
              aggfunc=lambda x: ''.join(str(v) for v in x))
# .to_csv('best_hyperparameters_to_present.csv')

a.pivot_table(values='vocab_size',
              index=['label_index', 'label']
).to_csv('vocab_size.csv')
### save list of vocab
for chosen_label in range(19):
    vocab = load_vectoriser(chosen_label).vocabulary_.keys()
    file_name = f"label_{str(chosen_label)}/label_{str(chosen_label)}_vocabulary.txt"
    pd.Series(list(vocab)).to_csv(file_name, index=False, header=False)



import os

for num in range(19):
    if num==2:
        pass
    else:
        l = 'label_' + str(num)
        for i in os.listdir(l):
            j = i.replace('5min', '6min')
            print(j)
            os.rename(l + '/' + i, l + '/' + j)