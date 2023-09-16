random_state=0
import math
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, svm
# import load_data
# import prepare_data_for_classification

# This is to then be used for cross-validation
def reviewed_group_subset(dataframe, reviewed_group):
    subset = dataframe[dataframe['reviewed'] == reviewed_group]
    return subset

def calculate_sample_size(dataframe, imbalanced_method: str):
    if imbalanced_method=='under_sample':
        count_each_label = dataframe.groupby('label')['text'].count().to_numpy()
        sample_size = np.nanmin(count_each_label)
        return sample_size

def return_balanced_sample(dataframe, reviewed_group, imbalanced_method: str):
    reviewed_subset = reviewed_group_subset(dataframe, reviewed_group)
    sample_size = calculate_sample_size(reviewed_subset, imbalanced_method)
    balanced_sample = reviewed_subset.groupby('label').sample(n=sample_size, replace=False, random_state=random_state)
    return balanced_sample

def split_train_test(dataframe, x_col, y_col, test_percent):
    x_train, x_test, y_train, y_test = train_test_split(dataframe[x_col], dataframe[y_col]
                                                        ,stratify = dataframe[y_col]
                                                        ,test_size=test_percent,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test

def run_pipeline(dataframe, test_percent: float):
    balanced_sample = return_balanced_sample(dataframe, True, 'under_sample')
    x_train, x_test, y_train, y_test = split_train_test(balanced_sample, 'text', 'label', test_percent)
    return x_train, x_test, y_train.astype('int'), y_test.astype('int')