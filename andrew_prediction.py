#!/usr/bin/env python

from os.path import join
import time

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def main():
  base_dir = "data"
  background_file = join(base_dir, "background.out.csv")
  label_file = join(base_dir, "train.out.csv")
  print "Reading in %s..." % background_file
  background_df = pd.read_csv(background_file, low_memory=False)
  print "Reading in %s..." % label_file
  label_df = pd.read_csv(label_file, low_memory=False)

  name_to_model = { 
    'bnb': lambda: BernoulliNB(),
    'mnb': lambda: MultinomialNB(),
    #'svm': lambda: make_svm_model(),
    'dt': lambda: make_dt_model(),
    'rf': lambda: make_rf_model()
  }
  
  # Do the training!
  # TODO: test other labels as well
  n_folds = 5
  label_name = "jobTraining"
  for model_name in name_to_model:
    print "Running model '%s'..." % model_name
    start_time = time.time()
    kf = KFold(n_splits = n_folds)
    result = []
    for i, (train_indices, test_indices) in enumerate(kf.split(label_df)):
      fold_start_time = time.time()
      train_df = label_df.transpose()[train_indices].transpose()
      test_df = label_df.transpose()[test_indices].transpose()
      train_features = get_features(train_df, background_df)
      test_features = get_features(test_df, background_df)
      train_labels = get_label(train_df, label_name)
      test_labels = get_label(test_df, label_name)
      if model_name == "svm":
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
      model = name_to_model[model_name]()
      model.fit(train_features, train_labels)
      predicted_labels = [float(model.predict_proba([tf])[0][1]) for tf in test_features]
      expected_labels = test_labels.tolist()
      result += [(predicted_labels, expected_labels)]
      print "  - Finished running on fold %i, took %s s" % (i, time.time() - fold_start_time())
    elapsed = time.time() - start_time
    print_result(result, elapsed)

def make_dt_model():
  param_grid = [{'min_samples_leaf':[1, 10, 100], 'max_features':[0.5, 0.75, 0.9]}]
  return GridSearchCV(DecisionTreeClassifier(), param_grid)

def make_rf_model():
  return RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_features=0.75)

def make_svm_model():
  #param_grid = [\
  #  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},\
  #  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
  #]  
  param_grid = [{'C':[0.1, 1, 10, 100]}]
  return BaggingClassifier(GridSearchCV(LinearSVC(), param_grid), n_jobs=4)

def get_label(df, label_name):
  '''
  Extract the specified label from the given dataframe.
  :return a numpy array with a single column for the label.
  '''
  labels = df[label_name].as_matrix().tolist()
  labels = [1 if label else 0 for label in labels]
  return np.array(labels)

def get_features(label_df, background_df):
  '''
  Extract relevant features from the given dataframe.
  :return a numpy array with a multiple columns, one for each feature.
  '''
  features = []
  for challenge_id in label_df["challengeID"]:
    row = background_df[background_df["challengeID"] == challenge_id]
    row = row.drop(["challengeID"], axis=1).as_matrix()[0].tolist()
    features += [row]
  return np.array(features)

def print_result(label_pairs, elapsed):
  '''
  Convenience method to print the test result of binary classification.
  :param a list of (predicted_labels, expected_labels) pairs, one for each fold
  '''
  accuracy, roc_auc, f1, precision, recall = [], [], [], [], []
  num_labels = len(label_pairs[0][1])
  for (predicted_labels, expected_labels) in label_pairs:
    assert len(predicted_labels) == len(expected_labels)
    num_correct = 0 
    rounded_predicted_labels = [round(l) for l in predicted_labels]
    for i in range(len(predicted_labels)):
      if rounded_predicted_labels[i] == expected_labels[i]:
        num_correct += 1
    accuracy += [float(num_correct) * 100 / len(expected_labels)]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(expected_labels, predicted_labels)
    roc_auc += [auc(false_positive_rate, true_positive_rate)]
    f1 += [f1_score(expected_labels, rounded_predicted_labels)]
    precision += [precision_score(expected_labels, rounded_predicted_labels)]
    recall += [recall_score(expected_labels, rounded_predicted_labels)]
  # Take the average of the things
  accuracy = float(sum(accuracy)) / len(accuracy)
  roc_auc = float(sum(roc_auc)) / len(roc_auc)
  f1 = float(sum(f1)) / len(f1)
  precision = float(sum(precision)) / len(precision)
  recall = float(sum(recall)) / len(recall)
  num_correct = int(num_labels * accuracy / 100)
  print "\n================================"
  print "You guessed %s/%s = %.3f%% correct." % (num_correct, num_labels, accuracy)
  #print "  - False positive rate: %s" % false_positive_rate.tolist()
  #print "  - True positive rate: %s" % true_positive_rate.tolist()
  #print "  - Thresholds: %s" % thresholds.tolist()
  print "  - AUC: %s" % roc_auc
  print "  - Precision: %s" % precision
  print "  - Recall: %s" % recall
  print "  - F1: %s" % f1
  print "  - Time (s): %.3f" % elapsed
  print "================================\n"

if __name__ == "__main__":
  main()

