#!/usr/bin/env python

from os.path import join

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def main():
  base_dir = "data"
  background_file = join(base_dir, "background.out.csv")
  label_file = join(base_dir, "train.out.csv")
  print "Reading in %s..." % background_file
  background_df = pd.read_csv(background_file, low_memory=False)
  print "Reading in %s..." % label_file
  label_df = pd.read_csv(label_file, low_memory=False)
  
  # TODO: make this k-fold CV
  num_labels = label_df.shape[0]
  num_train_examples = int(num_labels * 0.8)
  train_df = label_df[:num_train_examples]
  test_df = label_df[num_train_examples:]
  
  # TODO: do other labels as well
  label_name = "jobTraining"
  print "Building train features..."
  train_features = get_features(train_df, background_df)
  print "Building train labels..."
  train_labels = get_label(train_df, label_name)
  print "Building test features..."
  test_features = get_features(test_df, background_df)
  print "Building test labels..."
  test_labels = get_label(test_df, label_name)
  
  # TODO: do other models as well
  print "Training the model on %s train examples..." % num_train_examples
  model = BernoulliNB()
  model.fit(train_features, train_labels)
  predicted_labels = [float(model.predict_proba([tf])[0][1]) for tf in test_features]
  expected_labels = test_labels.tolist()
  print_result(predicted_labels, expected_labels)

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

def print_result(predicted_labels, expected_labels):
  '''
  Convenience method to print the test result of binary classification.
  '''
  assert len(predicted_labels) == len(expected_labels)
  num_correct = 0 
  rounded_predicted_labels = [round(l) for l in predicted_labels]
  for i in range(len(predicted_labels)):
    if rounded_predicted_labels[i] == expected_labels[i]:
      num_correct += 1
  percent_correct = float(num_correct) * 100 / len(expected_labels)
  false_positive_rate, true_positive_rate, thresholds = roc_curve(expected_labels, predicted_labels)
  roc_auc = auc(false_positive_rate, true_positive_rate)
  f1 = f1_score(expected_labels, rounded_predicted_labels)
  precision = precision_score(expected_labels, rounded_predicted_labels)
  recall = recall_score(expected_labels, rounded_predicted_labels)
  print "\n================================"
  print "You guessed %s/%s = %s%% correct." % (num_correct, len(expected_labels), percent_correct)
  #print "  - False positive rate: %s" % false_positive_rate.tolist()
  #print "  - True positive rate: %s" % true_positive_rate.tolist()
  #print "  - Thresholds: %s" % thresholds.tolist()
  print "  - AUC: %s" % roc_auc
  print "  - Precision: %s" % precision
  print "  - Recall: %s" % recall
  print "  - F1: %s" % f1
  print "================================\n"

if __name__ == "__main__":
  main()

