#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
def parse_data(inputcsv):
    df = pd.read_csv(inputcsv, low_memory=False)
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    return df


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
def fillMissing(df):
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1

    return num


def column_na_portions(data):
    return (data.isnull().sum() / float(len(data.index)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--background", help="Path to background.csv", required=True)
    parser.add_argument("-o", "--output_filename", help="Path to output file", required=True)
    parser.add_argument("-c", "--cutoff", help="[OPTIONAL] Remove columns that have more than cutoff portion (float) null values", default=0, type=float)
    parser.add_argument("-v", "--verbose", help="[OPTIONAL] Verbosity (flag)", action="store_true")

    options = parser.parse_args()

    run(options)


def run(options):
    if options.verbose:
        print "Loading data"
    # filled = fillMissing(options.background)
    data = parse_data(options.background)

    if options.verbose:
        print "Calculating portions of entries that are NA"
    na_portions = column_na_portions(data)

    if options.verbose:
        print "Filling NA values"
    data = fillMissing(data)

    if options.verbose:
        print "Removing columns with NA portions that are too high"
    data = data.loc[:,~(na_portions.gt(float(options.cutoff)))]

    if options.verbose:
        print "Writing preprocessed data to " + str(options.output_filename)
    data.to_csv(options.output_filename, index=False)



if __name__ == "__main__":
    main()

