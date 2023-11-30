import numpy as np
import pandas as pd
import itertools
import sys
import random



class MultivariateMAR:

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
    def verify_ohe(dataset, nominal_features,test_columns):
        '''Really ineficient FIX this'''
        #Get Possible combinations

        for feature in test_columns:
            if feature not in dataset.columns.values:
                dataset[feature] = 0
        
        dataset = dataset[test_columns]
        

        ohe_features = [string.split("_") for string in dataset.columns.values if ("_" in string and string.split("_")[0] in nominal_features)]
        
        #print(ohe_features)

        ohe_feature_subgroups = {}
        for ohe_feature in ohe_features:
            if ohe_feature[0] not in ohe_feature_subgroups.keys():
                ohe_feature_subgroups[ohe_feature[0]] = 1
            else:
                ohe_feature_subgroups[ohe_feature[0]] += 1



        #Make feature with all 0 into nan
        itr = 0

        for ohe_feature in ohe_feature_subgroups:
            #print(ohe_feature)
            #Get names for all features in subgroup
            ohe_features_subgroup = ohe_features[itr:itr+ohe_feature_subgroups[ohe_feature]]
            #Revert back to original name
            ohe_features_subgroup = ["_".join(k) for k in ohe_features_subgroup]
            
            train_subgroup = dataset[ohe_features_subgroup]
            
            m = train_subgroup.values.astype(float)
            
            for row in np.where(~m.any(axis=1))[0]:
                m[row, :] = np.nan

            dataset[ohe_features_subgroup] = pd.DataFrame(m, index = dataset.index,columns = train_subgroup.columns)

            itr += ohe_feature_subgroups[ohe_feature]
        
        return dataset

    

    @staticmethod
    def amputate(data_obj, dataset: pd.DataFrame, missing_rate: float,test_columns) -> pd.DataFrame:

        #Split sensitive feature names based on the last _ of the name and organize groups based on first part of name
        dataset = MultivariateMAR.undummify(dataset)

        dataset = dataset.copy()
        num_obs = len(dataset)
        num_feat = len(dataset.columns.values)
        sum_mvs = round(missing_rate * num_obs * num_feat)
        
        #Get list of nominal features and numerical and ordinal features 
        categorical_nominal_features = data_obj.get_categorical_nominal_features() + data_obj.get_sensitive_attributes_with_joint()
        numerical_and_ordinal_features = [feature for feature in dataset.columns.values.tolist() if feature not in categorical_nominal_features]
        categorical_ordinal_features = data_obj.get_categorical_ordinal_features()
        
        #print(dataset)

        mar_feature_triplets = {}

        
        #Create triplets
        while numerical_and_ordinal_features:
            '''
            if there are 4 elements left 
                1 numerical -> 3 others

            if there are 2 elements left
                1 numerical -> 1 others

            else
                1 - 1 numerical -> 2 categorical
                2 - 1 numerical -> 1 numerical and 1 categorical
                3 - 1 numerical -> 2 numerical
            '''
            #Feature which will not be amputated
            num_feature = random.choice(numerical_and_ordinal_features)
            numerical_and_ordinal_features.remove(num_feature)
            #Features which will be amputated
            features_to_amputate =[]
            
            if len(numerical_and_ordinal_features)+len(categorical_nominal_features) == 3:
                #If there are 3 left form a group of 4
                features_to_amputate = numerical_and_ordinal_features + categorical_nominal_features
                #Pop the last ones
                numerical_and_ordinal_features = []
                categorical_nominal_features = []
            elif len(numerical_and_ordinal_features)+len(categorical_nominal_features) == 1:
                #If there is only 1 left form a pair
                if categorical_nominal_features:
                    features_to_amputate = categorical_nominal_features
                else:
                    features_to_amputate = numerical_and_ordinal_features
                #Pop the last ones
                numerical_and_ordinal_features = []
                categorical_nominal_features = []
            else:
                if len(categorical_nominal_features) >= 2:
                    #Choose two random nominal features
                    features_to_amputate = random.sample(categorical_nominal_features, 2)
                    #Pop the last ones
                    for feature in features_to_amputate:
                        categorical_nominal_features.remove(feature)
                elif len(categorical_nominal_features) == 1: 

                    features_to_amputate.append(categorical_nominal_features[0])
                    categorical_nominal_features = []

                    temp = random.choice(numerical_and_ordinal_features)
                    features_to_amputate.append(temp)
                    numerical_and_ordinal_features.remove(temp)                
                else:
                    #Choose two random nominal features
                    features_to_amputate = random.sample(numerical_and_ordinal_features, 2)
                    #Pop the last ones
                    for feature in features_to_amputate:
                        numerical_and_ordinal_features.remove(feature)
                
            mar_feature_triplets[num_feature] = features_to_amputate


        # Randomly distribute missing values by the triples.
        max_mvs_feat = round(min(missing_rate*2.25, 0.9) * num_obs)
        num_mvs_per_tripplet = np.zeros(len(mar_feature_triplets))
        
        while sum_mvs > 0:
            itr=0
            for triplet in mar_feature_triplets:
                if num_mvs_per_tripplet[itr] < max_mvs_feat:
                    num_mvs_it = np.random.randint(0, max(min(max_mvs_feat - num_mvs_per_tripplet[itr] + 1, sum_mvs/len(mar_feature_triplets[triplet]) + 1),2))
                    sum_mvs -= num_mvs_it * len(mar_feature_triplets[triplet])
                    num_mvs_per_tripplet[itr] += num_mvs_it
                    if sum_mvs == 0:
                        break
                itr+=1


        # Amputate the values.

        itr=0
        for triplet in mar_feature_triplets:
            num_mv = round(num_mvs_per_tripplet[itr])
            num_mv = num_mv if num_mv > 0 else 0
            dataset[triplet] = dataset[triplet].astype(str).astype(float)
            
            try:
                cols = dataset.nsmallest(num_mv, triplet).index.tolist()
            except Exception as e:
                dataset.to_csv("train_exception.csv",index=False)
                f = open("exception.txt", "w")
                f.write(str(e))
                f.close()
            
            for mar_feature in mar_feature_triplets[triplet]:
                dataset.loc[cols,mar_feature] = np.nan
            itr +=1

        
        '''
        sum_mvs_cols = 0
        for c in dataset.columns.values:
            num_mvs = pd.isnull(dataset.loc[:, c]).values.astype(int).sum()
            print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(dataset)) * 100)}%).")
            sum_mvs_cols += num_mvs

        print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(dataset) * len(dataset.columns.values))) * 100)}%.")
        '''

        dataset = pd.get_dummies(dataset, columns = data_obj.get_categorical_nominal_features())
        dataset = MultivariateMAR.verify_ohe(dataset, data_obj.get_categorical_nominal_features(),test_columns)



        return dataset


# Example with the Iris dataset.
if __name__ == '__main__':
    iris_ds = pd.read_csv('data/preprocessed/ricci_numerical_binsensitive.csv')
    iris_ds = MultivariateMAR.amputate(iris_ds, missing_rate=0.4)
    print(iris_ds, end="\n\n")

    sum_mvs_cols = 0
    for c in iris_ds.columns.values:
        num_mvs = pd.isnull(iris_ds.loc[:, c]).values.astype(int).sum()
        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(iris_ds)) * 100)}%).")
        sum_mvs_cols += num_mvs

    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(iris_ds) * len(iris_ds.columns.values))) * 100)}%.")
