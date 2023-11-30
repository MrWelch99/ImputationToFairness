import numpy as np
import pandas as pd
import sys
import itertools
from data.objects.imputation_utils import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#One hot enode sensitive variable
#Imputate values
#Split feature names based on the last _ of the name and organize groups based on first part of name
#Make only one feature of each group have a value of 1 with all others having 0s
#Make sensitive variables categorical again

def mice_imputation(train,sensitive_atributes):
    #One hot enode sensitive variable
    ohe_train, ohe_train_cols = one_hot_encode_sensitive_data(train,sensitive_atributes)

    #Imputate values
    imp = IterativeImputer(missing_values=np.nan,initial_strategy='median',max_iter=10, verbose=2,random_state=0)

    try:
        train=imp.fit_transform(ohe_train)
    except Exception as e:
        ohe_train.to_csv("mice_exception_dataset_pre_imputation.csv",index=False)
        train.to_csv("mice_exception_dataset.csv",index=False)
        f = open("mice_exception.txt", "a")
        f.write(str(e))
        f.close()
        print("IT GOT SCREWED AGAIN")
        sys.exit(1)

    #Make a dataframe again
    train = pd.DataFrame(train, index=ohe_train.index,columns=ohe_train_cols)


    #Make only one feature of each group have a value of 1 with all others having 0s
    train =  verify_ohe(train, ohe_train_cols)
    
    #Make sensitive variables categorical again
    train =  undummify_sensitive(train,sensitive_atributes)

    #print(train)
    

    
    return train