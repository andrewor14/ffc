#!/usr/bin/env python

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

import pandas as pd
import numpy as np
from os.path import join

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
train_labels = train_df[label_name].as_matrix()
test_labels = train_df[label_name].as_matrix()
def get_features(df):
  features = []
  for challenge_id in df["challengeID"]:
    row = background_df[background_df["challengeID"] == challenge_id]
    row = row.drop(["challengeID"], axis=1).as_matrix()[0]
    features += [row]
  return np.array(features)
print "Building train features..."
train_features = get_features(train_df)
print "Building test features..."
test_features = get_features(test_df)

print "Training the model on %s train examples..." % num_train_examples
model = BernoulliNB()
model.fit(train_features, train_labels)

#  train_labels = np.array(train_labels)
#    bow_data = np.array(bow_data)
#      model.fit(bow_data, train_labels)


