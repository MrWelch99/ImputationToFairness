import numpy as np
import pandas as pd
import itertools
import sys
import math
from imblearn.over_sampling import SMOTENC
 


class Oversampler:

    @staticmethod
    def undummify(df, prefix_sep="_"):
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        categorical_columns = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                new_col=col.split(prefix_sep)[0]
                undummified = (
                    df.filter(regex= "^"+new_col+"_.+$")
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
                if col not in categorical_columns:
                    categorical_columns.append(col)
            else:
                series_list.append(df[col])

        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df
    
    @staticmethod
    def oversample(data_obj, dataset: pd.DataFrame, sensitive_atributes) -> pd.DataFrame:
        num_obs = len(dataset)

        #Get class col
        class_col = data_obj.get_class_attribute()
        class_labels =  dataset[class_col].values

        #Distribute dataset acording to sensitive attributes

        #Last attribute is the joint attribute
        joint_sen_atr = sensitive_atributes[-1]
        sensitive_class_lst=dataset[joint_sen_atr].unique()


        #Create pairwise list of idx for each sensitive classes
        sensitive_class_pairwise_idx_lst=[]
        for sen in sensitive_class_lst:         
            # Get rid of 0 len combinations
            temp_lst = dataset[joint_sen_atr] == sen

            if len(temp_lst)!=0:
                sensitive_class_pairwise_idx_lst.append(temp_lst)
            else:
                #print("OI")
                sensitive_class_lst.remove(sen)

        dataset = Oversampler.undummify(dataset)

        categorical_features = data_obj.get_categorical_nominal_features() + data_obj.get_sensitive_attributes_with_joint()
        dataset=dataset[[c for c in dataset if c != class_col] + [class_col]]
        
        #Fractal SMOTE

        #Check Indices
        oversampled_set = pd.DataFrame(columns=dataset.columns)
        for i in range(len(sensitive_class_lst)): 

            sen_dataset = dataset[sensitive_class_pairwise_idx_lst[i]]
            #Get X and Y
            X = sen_dataset.loc[:, sen_dataset.columns != class_col]
            #Get Categorical Feature
            col_ind = [X.columns.get_loc(col) for col in categorical_features]
            
            y = sen_dataset[class_col]

            class_count = {}
            for element in y:
                if element not in class_count:
                    class_count[element] = 1
                else:
                    class_count[element] += 1

            min_value=len(y)
            max_value=0

            for i in class_count:
                if class_count[i] < min_value:
                    min_value = class_count[i]
                if class_count[i] > max_value:
                    max_value = class_count[i]

            if len(class_count)>1 and (min_value>=2) and (min_value != max_value):
                
                k_neighbors = 5
                if min_value-1 < 5:
                    k_neighbors = min_value-1

                #Increase minotity class by 30 %
                #min_value = int(min_value*1.3)
                old_min_value = min_value
                min_value = math.ceil(min_value*1.3)

                if min_value > max_value:
                    min_value = max_value
                minority_rate = min_value/max_value

                smote_nc = SMOTENC(categorical_features = col_ind,sampling_strategy = minority_rate, k_neighbors=k_neighbors)


                #Oversample the dataset
                X_new = X
                y_new = y
                try:
                    X_new, y_new = smote_nc.fit_resample(X,y)
                except Exception as e:
                    f = open("exception.txt", "w")
                    f.write(str(old_min_value)+" "+str(min_value)+" "+str(max_value)+" "+str(minority_rate)+"\n")
                    f.write(str(e)+"\n")
                    f.close()
                
                
                #X_new, y_new = smote_nc.fit_resample(X,y)
                
                #Get only the new values
                X_oversampled =  X_new[len(X):]
                y_oversampled =  y_new[len(X):]

                #Add y back into the dataset
                X_oversampled[class_col] = y_oversampled
                X_oversampled = X_oversampled[oversampled_set.columns]
                
                #Merge all new intances
                oversampled_set = pd.concat([oversampled_set, X_oversampled], ignore_index=True)

        oversampled_set = oversampled_set.sample(frac=1).reset_index(drop=True)
        oversampled_set.index = range(100000,len(oversampled_set)+100000)
        
            
        dataset = pd.concat([dataset, oversampled_set])

        dataset = pd.get_dummies(dataset,columns=data_obj.get_categorical_nominal_features())



        return dataset
        #X, y = oversample.fit_resample(X, y)
        #sys.exit()