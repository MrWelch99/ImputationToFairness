import numpy as np
import pandas as pd
import sys
import itertools


def mean_mode_imputation(train):
    for c in train.columns:
        #print("\n"+c)
        #print(train.dtypes[c])
        #print(train[c].values)
        if train.dtypes[c] == np.object:
            #Find most frequent Value
            col_wihout_nan = train[c][~pd.isnull(train[c])].values

            unique,pos = np.unique(col_wihout_nan,return_inverse=True) #Finds all unique elements and their positions
            counts = np.bincount(pos)                     #Count the number of each unique element
            maxpos = counts.argsort()[::-1][0]

            most_frequent_value = unique[maxpos]
            #print(most_frequent_value)

            #Replace nan with most frequent Value
            col_wih_nan = train[c][pd.isnull(train[c])].index
            train[c][col_wih_nan] = most_frequent_value
            #print(train[c])
        else:
            #Find the mean
            col_wihout_nan = train[c][~pd.isnull(train[c])].values
            #print(col_wihout_nan)
            mean_value = col_wihout_nan.mean()
            #print(mean_value)



            #Replace nan with mean
            col_wih_nan = train[c][pd.isnull(train[c])].index
            train[c][col_wih_nan] = mean_value

            if np.array_equal(col_wihout_nan, col_wihout_nan.astype(int)):
                train[c] = np.round(train[c].values)
    
    return train
        