import numpy as np
import pandas as pd
import sys
import itertools
from sklearn.impute import KNNImputer


def one_hot_encode_sensitive_data(train,sensitive_atributes):
    #One hot enode sensitive varialbe
    ohe_train = pd.get_dummies(train,columns = sensitive_atributes)
    ohe_train_cols = list(ohe_train.columns)

    return ohe_train, ohe_train_cols


def undummify_sensitive(df, sensitive_attributes, prefix_sep="_"):
    #Make sensitive variables categorical again
    cols2collapse = {
        item.split(prefix_sep)[0] if item.split(prefix_sep)[0] in sensitive_attributes else item :
             (item.split(prefix_sep)[0] in sensitive_attributes) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            new_col=col.split(prefix_sep)[0]
            undummified = (
                df.filter(regex= "^"+new_col+"_.+$")
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(new_col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df    

def verify_ohe(train,ohe_train_cols):
    #Split sensitive feature names based on the last _ of the name and organize groups based on first part of name
    ohe_features = [string.split("_") for string in ohe_train_cols if "_" in string]
    #print(ohe_features)

    ohe_feature_subgroups = {}
    for ohe_feature in ohe_features:
        if ohe_feature[0] not in ohe_feature_subgroups.keys():
            ohe_feature_subgroups[ohe_feature[0]] = 1
        else:
            ohe_feature_subgroups[ohe_feature[0]] += 1

    #Make only one feature of each group have a value of 1 with all others having 0s
    itr = 0
    for ohe_feature in ohe_feature_subgroups:
        #print(ohe_feature)
        #Get names for all features in subgroup
        ohe_features_subgroup = ohe_features[itr:itr+ohe_feature_subgroups[ohe_feature]]
        #Revert back to original name
        ohe_features_subgroup = ["_".join(k) for k in ohe_features_subgroup]
        
        train_subgroup = train[ohe_features_subgroup]
        
        m = np.zeros_like(train_subgroup.values)
        m[np.arange(len(train_subgroup)), train_subgroup.values.argmax(1)] = 1

        train[ohe_features_subgroup] = pd.DataFrame(m, index = train.index ,columns = train_subgroup.columns).astype(int)

        itr += ohe_feature_subgroups[ohe_feature]
    
    return train