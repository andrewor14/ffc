#!/usr/bin/python


import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

from scipy.stats import randint as sp_randint


NUM_JOBS = 4


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
def fillMissing(inputcsv, outputcsv):
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    num.to_csv(outputcsv, index=False)


def no_null_mask(labels, selector=None):
    if selector:
        good_row_mask = ~(labels.loc[:,selector].isnull())
    else:
        good_row_mask = ~(labels.iloc[:,1:].isnull().any(axis=1))

    return good_row_mask


#####################################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--background", help="Path to background.csv", required=True)
    parser.add_argument("-t", "--train", help="Path to train.csv", required=True)
    parser.add_argument("-o", "--outfile", help="Path for output predictions.csv", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    parser.add_argument("-s", "--selector", help="Selector. Typing accuracy counts here for now.", required=True)

    options = parser.parse_args()

    run(options)


def run(options):
    ofile = open(options.outfile, 'w')

    if options.verbose:
        print "Reading background file"
    data = pd.read_csv(options.background, low_memory=False)

    # TODO(tfs;2017-03-20): Select labels/samples per variable
    if options.verbose:
        print "Parsing and extracting non-null labels"
    train_labels = pd.read_csv(options.train)
    good_label_mask = no_null_mask(train_labels, selector=options.selector)
    # good_sample_ids = good_labels.loc[:,'challengeID'].values.flatten()

    if options.verbose:
        print "Selecting samples with enough data (non-null labels)"
    good_ids = train_labels.loc[good_label_mask,'challengeID'].values.flatten()

    # See http://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    #     for information on isin and the pattern used here
    good_data = data[data['challengeID'].isin(good_ids)]

    # See http://stackoverflow.com/questions/34246336/python-randomforest-unknown-label-error
    # gpa_labels = np.asarray(train_labels.loc[good_label_mask,options.selector], dtype=np.float64)
    # gpa_labels = train_labels.loc[good_label_mask,['challengeID',options.selector]]
    
    if options.verbose:
        print "Ensuring ordering between samples and labels"
    gpas = dict(train_labels.loc[good_label_mask,['challengeID',options.selector]].values)

    ordered_labels = []
    for cid in good_data.loc[:,'challengeID'].values:
        ordered_labels.append(gpas[cid])

    # if options.verbose:
    #     print "Start training at: " + time.strftime("%H:%M:%S")
    # # See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # # and http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html
    # clf = linear_model.LogisticRegression(penalty='l1', C=0.8)
    # # clf.fit(X=good_data.iloc[:,:9001].values, y=gpa_labels.values.flatten())
    # clf.fit(X=good_data.iloc[:,:9001].values, y=gpa_labels)
    # if options.verbose:
    #     print "End training at:   " + time.strftime("%H:%M:%S")


    X = good_data.iloc[:,:].values
    y = ordered_labels


    if options.verbose:
        print "Scaling data"
    # See http://scikit-learn.org/stable/modules/preprocessing.html
    #     for information on preprocessing
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)


    # See http://scikit-learn.org/stable/modules/cross_validation.html
    #     for cross validation
    v = 0
    if options.verbose:
        print "Prep cross validation at: " + time.strftime("%H:%M:%S")
        v = 10
    svm_reg = svm.SVR()
    lasso_reg = linear_model.Lasso()
    rf_reg = RandomForestRegressor()
    # reg = MLPRegressor() # Not fast


    # Hyperparameter tuning for random forest regressor
    rf_hyperparameter_space = {
        "n_estimators":      [50, 75],
        "criterion":         ["mse"],
        "max_features":      ["auto"],
        "min_samples_split": [10, 25]
    }

    lasso_hyperparameter_space = {
        "alpha":         [1000, 10000],
        "fit_intercept": [True],
        "normalize":     [True, False],
        "tol":           [0.01, 0.1, 1.0],
        "positive":      [True, False],
        "selection":     ["random"],
        "max_iter":      [1000]
    }

    svm_hyperparameter_space = {
        "kernel":   ["rbf", "linear", "poly", "sigmoid"],
        "epsilon":  [0.01, 0.1, 1.0],
        "C":        [0.1, 1.0, 10],
        "degree":   [1, 2, 3, 4],
        "max_iter": [5000]
    }

    num_iterations = 32



    ####################################################################################################################################################
    # See http://scikit-learn.org/stable/modules/grid_search.html                                                                                      #
    #     for information on grid search for hpyerparameter tuning                                                                                     #
    # And http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV  #
    #     for information on randomized search                                                                                                         #
    # See http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html                                                          #
    #     for example usage (adapted below)                                                                                                            #
    ####################################################################################################################################################
    search = GridSearchCV(rf_reg, param_grid=rf_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    # search = GridSearchCV(lasso_reg, param_grid=lasso_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    # search = RandomizedSearchCV(svm_reg, param_distributions=svm_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS, n_iter=num_iterations)

    if options.verbose:
        print "Start cross validation at: " + time.strftime("%H:%M:%S")
    search.fit(X, y)
    # search.fit(X_scaled, y) # Lasso seems to do worse with scaled data

    if options.verbose:
        print "End cross validation at:   " + time.strftime("%H:%M:%S")

    # print search.cv_results_
    # print search.best_estimator_
    print search.best_score_
    print search.best_params_
    print search.best_index_
    print search.scorer_
    return search

    # print reg.get_params()

    # print scores
    # return scores

    # if options.verbose:
    #     print "Predicting rest of labels"
    # gpa_predicts = gpa_model.predict(data.values)
    # grit_predicts = grit_model.predict(data.values)
    # hard_predicts = hard_model.predict(data.values)

    # Output
    if options.verbose:
        print "Writing predictions to " + options.outfile
    dumbpredict = False

    # for cid in data.loc[:,'challengeID'].values:
        


if __name__ == "__main__":
    main()