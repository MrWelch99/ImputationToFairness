import numpy as np
import pandas as pd
import sys
import itertools
from data.objects.imputation_utils import *
from sklearn.impute import KNNImputer

#One hot enode sensitive variable
#Imputate values
#Split feature names based on the last _ of the name and organize groups based on first part of name
#Make only one feature of each group have a value of 1 with all others having 0s
#Make sensitive variables categorical again

def knn_imputation(train,sensitive_atributes):
    #One hot enode sensitive variable
    ohe_train, ohe_train_cols = one_hot_encode_sensitive_data(train,sensitive_atributes)

    #Imputate values
    imputer = KNNImputer(n_neighbors=7)    
    train = imputer.fit_transform(ohe_train)

    #Make a dataframe again
    train = pd.DataFrame(train, index=ohe_train.index, columns=ohe_train_cols)


    #Make only one feature of each group have a value of 1 with all others having 0s
    train =  verify_ohe(train, ohe_train_cols)
    
    #print(train)
    #Make sensitive variables categorical again
    train =  undummify_sensitive(train,sensitive_atributes)
   
    return train