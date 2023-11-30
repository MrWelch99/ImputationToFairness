import imp
from os import system
import pandas as pd
import sys
import math
import numpy
import numpy.random
from data.objects.Oversample import Oversampler
from data.objects.multivariate_mcar import MultivariateMCAR
from data.objects.multivariate_mar import MultivariateMAR
from data.objects.multivariate_mnar import MultivariateMNAR
from data.objects.true_multivariate_mcar import TrueMultivariateMCAR
from data.objects.MeanModeImputation import mean_mode_imputation
from data.objects.KNNImputation import knn_imputation
from data.objects.MICEImputation import mice_imputation
from data.objects.Oversample import Oversampler





#TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive"]
TAGS = ["numerical"]
TRAINING_PERCENT = 2.0 / 3.0


class ProcessedData():
    def __init__(self, data_obj, DEBUG_FLAG):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False
        self.DEBUG_FLAG = DEBUG_FLAG

    def get_processed_filename(self, tag):
        return self.data.get_filename(tag)

    def get_dataframe(self, tag):
        return self.dfs[tag]

    def get_params(algorithm):
        params = ""
        if algorithm.get_name() == "SVM":
            params =  str(algorithm.classifier.C) + ";" + algorithm.classifier.gamma
        elif algorithm.get_name() == 'Random_Forest':
            params  = str(algorithm.classifier.n_estimators)+";"+ str(algorithm.classifier.max_depth)+";"+algorithm.classifier.max_features
        return params

    def create_train_test_splits(self, num, sensitive_atributes, missing_rate, imputation_method,missing_mechanism,algorithm,start_run,pipeline):
        if self.has_splits:
            return self.splits

        for i in range(0, num):
            # we first shuffle a list of indices so that each subprocessed data
            # is split consistently
            n = len(list(self.dfs.values())[0])

            a = numpy.arange(n)
            numpy.random.shuffle(a)

            split_ix = int(n * TRAINING_PERCENT)
            train_fraction = a[:split_ix]
            test_fraction = a[split_ix:]

            for (k, v) in self.dfs.items():

                if pipeline == "missing_pipeline" or pipeline == "missing_pipeline_baseline":
                    #Handle pipeline where datasets already have missing data

                    ordinal_features = self.data.get_categorical_ordinal_features()

                    if pipeline == "missing_pipeline":
                        complete_dataset = self.missing_pipeline(sensitive_atributes, missing_rate, imputation_method, missing_mechanism, self.dfs[k], algorithm, start_run+i, ordinal_features, k,pipeline)
                    elif pipeline == "missing_pipeline_baseline":
                        dataset = ProcessedData.undummify_ordinal(self.dfs[k], ordinal_features)
                        complete_dataset, ordinal_dict =  self.data.convert_ordinal_2_numerical(dataset)


                    #Make test set
                    test = complete_dataset.iloc[test_fraction]

                    #Make train set
                    train = complete_dataset.iloc[train_fraction]

                    #print(class_col)

                    #For Debugging 

                    #print(train, end="\n\n")
                    '''
                    sum_mvs_cols = 0
                    for c in train.columns.values:
                        num_mvs = pd.isnull(train.loc[:, c]).values.astype(int).sum()
                        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(train)) * 100)}%).")
                        sum_mvs_cols += num_mvs

                    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(train) * len(train.columns.values))) * 100)}%.")
                    '''

                    #Check if train didn't lose ohe collmuns
                    
                    test = test[train.columns]

                    class_col = self.data.get_class_attribute()
                    train[class_col] = train[class_col].astype("int")

                    self.splits[k].append((train, test))

                else:
                    #Pipeline of complete datasets

                    #Make test set
                    test = self.dfs[k].iloc[test_fraction]
                    #Make ordinal data numerical
                    ordinal_features = self.data.get_categorical_ordinal_features()
                    test = ProcessedData.undummify_ordinal(test, ordinal_features)
                    test, ordinal_dict =  self.data.convert_ordinal_2_numerical(test)

                    #Make train set
                    train = self.dfs[k].iloc[train_fraction]

                    #Save Class Labels From Amputation
                    class_col = self.data.get_class_attribute()
                    #print(class_col)

                    if pipeline == "oversample_then_imputation":
                        train = self.oversample_then_imputation(sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, start_run+i, class_col, ordinal_features, k,pipeline)
                    elif pipeline == "imputation_then_oversample":
                        train = self.imputation_then_oversample(sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, start_run+i, class_col, ordinal_features, k,pipeline)
                    elif pipeline == "just_oversample":
                        train = self.just_oversample(sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, start_run+i, class_col, ordinal_features, k,pipeline)
                    elif pipeline == "baseline":
                        #Make ordinal numerical      
                        train = ProcessedData.undummify_ordinal(train, ordinal_features)
                        train, ordinal_dict =  self.data.convert_ordinal_2_numerical(train)


                    #For Debugging 
                    '''
                    sum_mvs_cols = 0
                    for c in train.columns.values:
                        num_mvs = pd.isnull(train.loc[:, c]).values.astype(int).sum()
                        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(train)) * 100)}%).")
                        sum_mvs_cols += num_mvs

                    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(train) * len(train.columns.values))) * 100)}%.")
                    '''

                    #Check if train didn't lose ohe collmuns
                    test = test[train.columns]
                    '''
                    try:
                        test = test[train.columns]
                    except Exception as e:
                        f = open("Student_exception.txt", "a")
                        f.write("Dataset "+ self.data.get_dataset_name())
                        f.write("Sensitive_atributes: "+str(sensitive_atributes)+"\n")
                        f.write("Algorithm: "+str(algorithm.name)+"\n")
                        f.write("Missing_rate: "+str(missing_rate)+"\n")
                        f.write("Imputation_method: "+str(imputation_method)+"\n")
                        f.write("Missing_mechanism: "+str(missing_mechanism)+"\n")
                        f.write("Pipeline: "+str(pipeline)+"\n")
                        f.write(str(e)+"\n")
                        f.write("\n\n")
                        f.close()
                    '''

                    train[class_col] = train[class_col].astype("int")

                    #

                    self.splits[k].append((train, test))

        self.has_splits = True
        return self.splits

    def oversample_then_imputation(self, sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, run_num, class_col, ordinal_features, tag,pipeline):
        
        #Make ordinal numerical      
        train = ProcessedData.undummify_ordinal(train, ordinal_features)
        train, ordinal_dict =  self.data.convert_ordinal_2_numerical(train)
        
        #Oversample file
        train = Oversampler.oversample(self.data, train, sensitive_atributes)

        #CHANGE THIS LATER
        f = self.data.get_oversampled_data_filename(algorithm.get_name(), tag, missing_rate, run_num, imputation_method, missing_mechanism,pipeline)
        # f = self.data.get_hyperparametrization_oversampled_data_filename(self,algorithm, tag, missing_rate, run_num, imputation_method, missing_mechanism):
        if not self.DEBUG_FLAG:
            train.to_csv(f)

        class_labels =  train[class_col].values

        train= train.drop(columns=[class_col])


        #Amputate train split
        if missing_mechanism == "mcar":
            train = MultivariateMCAR.amputate(self.data, train, sensitive_atributes, missing_rate)
        elif missing_mechanism == "mar":
            #Make sure both collumns are equal
            test_columns =  test.columns.values.tolist()
            test_columns.remove(class_col)
            train = MultivariateMAR.amputate(self.data, train, missing_rate,test_columns)
        elif missing_mechanism == "mnar":
            #Make sure both collumns are equal
            categorical_nominal_features = self.data.get_categorical_nominal_features() + self.data.get_sensitive_attributes_with_joint()
            train = MultivariateMNAR.amputate(train, missing_rate,True,categorical_nominal_features)
        elif missing_mechanism == "true_mcar":
            train = TrueMultivariateMCAR.amputate(train, missing_rate)

        
        #Imputate train split
        if imputation_method == "mean":
            train = mean_mode_imputation(train)
        elif imputation_method == "knn":
            train = knn_imputation(train,sensitive_atributes)
        elif imputation_method == "mice":
            train = mice_imputation(train,sensitive_atributes)


        train[class_col]=class_labels
        
        return train

    def imputation_then_oversample(self, sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, run_num, class_col, ordinal_features, tag,pipeline):
        '''
        FIX THIS 
        '''

        class_labels =  train[class_col].values

        train= train.drop(columns=[class_col])

        #Make ordinal numerical      
        train = ProcessedData.undummify_ordinal(train, ordinal_features)
        train, ordinal_dict =  self.data.convert_ordinal_2_numerical(train)



        #Amputate train split
        if missing_mechanism == "mcar":
            train = MultivariateMCAR.amputate(self.data, train, sensitive_atributes, missing_rate)
        elif missing_mechanism == "mar":
            #Make sure both collumns are equal
            test_columns =  test.columns.values.tolist()
            test_columns.remove(class_col)
            train = MultivariateMAR.amputate(self.data, train, missing_rate,test_columns)
        elif missing_mechanism == "mnar":
            #Make sure both collumns are equal
            categorical_nominal_features = self.data.get_categorical_nominal_features() + self.data.get_sensitive_attributes_with_joint()
            train = MultivariateMNAR.amputate(train, missing_rate,True,categorical_nominal_features)
        elif missing_mechanism == "true_mcar":
            train = TrueMultivariateMCAR.amputate(train, missing_rate)


        #Imputate train split
        if imputation_method == "mean":
            train = mean_mode_imputation(train)
        elif imputation_method == "knn":
            train = knn_imputation(train,sensitive_atributes)
        elif imputation_method == "mice":
            train = mice_imputation(train,sensitive_atributes)


        train[class_col]=class_labels

        #Oversample file
        train = Oversampler.oversample(self.data, train, sensitive_atributes)

        #CHANGE THIS LATER
        f = self.data.get_oversampled_data_filename(algorithm.get_name(), tag, missing_rate, run_num, imputation_method, missing_mechanism,pipeline)
        # f = self.data.get_hyperparametrization_oversampled_data_filename(self,algorithm, tag, missing_rate, run_num, imputation_method, missing_mechanism):
        if not self.DEBUG_FLAG:
            train.to_csv(f)
        
        return train

    def just_oversample(self, sensitive_atributes, missing_rate, imputation_method, missing_mechanism, train, test, algorithm, run_num, class_col, ordinal_features, tag,pipeline):
        
        #Make ordinal numerical      
        train = ProcessedData.undummify_ordinal(train, ordinal_features)
        train, ordinal_dict =  self.data.convert_ordinal_2_numerical(train)
        
        #Oversample file
        train = Oversampler.oversample(self.data, train, sensitive_atributes)

        #CHANGE THIS LATER
        f = self.data.get_oversampled_data_filename(algorithm.get_name(), tag, missing_rate, run_num, imputation_method, missing_mechanism,pipeline)
        # f = self.data.get_hyperparametrization_oversampled_data_filename(self,algorithm, tag, missing_rate, run_num, imputation_method, missing_mechanism):
        if not self.DEBUG_FLAG:
            train.to_csv(f)

        class_labels =  train[class_col].values

        train= train.drop(columns=[class_col])

        train[class_col]=class_labels
        
        return train

    def missing_pipeline(self, sensitive_atributes, missing_rate, imputation_method, missing_mechanism, dataset, algorithm, run_num, ordinal_features, tag,pipeline):
        '''
        FIX THIS 
        '''



        #Make ordinal numerical      
        dataset = ProcessedData.undummify_ordinal(dataset, ordinal_features)
        dataset, ordinal_dict =  self.data.convert_ordinal_2_numerical(dataset)



        #Imputate train split
        if imputation_method == "mean":
            dataset = mean_mode_imputation(dataset)
        elif imputation_method == "knn":
            dataset = knn_imputation(dataset,sensitive_atributes)
        elif imputation_method == "mice":
            dataset = mice_imputation(dataset,sensitive_atributes)


        #Oversample file
        dataset = Oversampler.oversample(self.data, dataset, sensitive_atributes)

        #CHANGE THIS LATER
        f = self.data.get_oversampled_data_filename(algorithm.get_name(), tag, missing_rate, run_num, imputation_method, missing_mechanism,pipeline)
        # f = self.data.get_hyperparametrization_oversampled_data_filename(self,algorithm, tag, missing_rate, run_num, imputation_method, missing_mechanism):
        if not self.DEBUG_FLAG:
            dataset.to_csv(f)
        
        return dataset


    def get_sensitive_values(self, tag):
        """
        Returns a dictionary mapping sensitive attributes in the data to a list of all possible
        sensitive values that appear.
        """
        df = self.get_dataframe(tag)
        all_sens = self.data.get_sensitive_attributes_with_joint()
        sensdict = {}
        for sens in all_sens:
             sensdict[sens] = list(set(df[sens].values.tolist()))
        return sensdict

    def undummify_ordinal(df, ordinal_features, prefix_sep="_"):
        cols2collapse = {
            item.split(prefix_sep)[0] if item.split(prefix_sep)[0] in ordinal_features else item:
                (prefix_sep in item and item.split(prefix_sep)[0] in ordinal_features) for item in df.columns
        }
        series_list = []

        itr = 0 

        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                )

                for i in range(len(undummified)):
                    if not pd.isnull(undummified[i]):
                        undummified[i] = undummified[i].split(prefix_sep, maxsplit=1)[1]

                undummified = undummified.rename(col)
                #.apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                #.rename(col)
                series_list.append(undummified)
            else:
                series_list.append(df[col])

        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df