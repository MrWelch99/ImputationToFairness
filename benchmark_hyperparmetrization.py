from sklearn import datasets
import fire
import psutil
import multiprocessing as mp
import os
import statistics
import sys
import pandas as pd
import numpy as np

import results
from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from algorithms.baseline.SVM import SVM
from algorithms.baseline.Random_Forest import Random_Forest
from metrics.list import get_metrics

from algorithms.ParamGridSearch import ParamGridSearch

NUM_TRIALS_DEFAULT = 1
START_RUN_ID = 0

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def get_params(algorithm):
    params = ""
    if algorithm.get_name() == "SVM":
        params =  str(algorithm.classifier.C) + ";" + algorithm.classifier.gamma
    elif algorithm.get_name() == 'Random_Forest':
        params  = str(algorithm.classifier.n_estimators)+";"+ str(algorithm.classifier.max_depth)+";"+algorithm.classifier.max_features
    return params


def run(dataset, algorithm, num_trials = NUM_TRIALS_DEFAULT, start_run = START_RUN_ID, imputation_method = "mean", missing_mechanism = "mcar", pipeline = "oversample_then_imputation"):
    
    missing_rates = [0]
    #missing_rates = [0,0.05,0.1,0.2,0.4]

    print("Datasets: '%s'" % dataset)
    for missing_rate in missing_rates:
        print("Missing rate: "+str(missing_rate))

        for dataset_obj in DATASETS:
            if not dataset_obj.get_dataset_name() in dataset:
                continue

            print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())


            #all_sensitive_attributes = dataset_obj.get_sensitive_joint_attributes()
            #Reverter depois de corrigir o problema
            all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()

            processed_dataset = ProcessedData(dataset_obj)
            train_test_splits = processed_dataset.create_train_test_splits(num_trials,all_sensitive_attributes,missing_rate,imputation_method,missing_mechanism,algorithm,start_run,pipeline)

            
            dict_sensitive_lists = {}            

            #all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()

            print("    Algorithm: %s" % algorithm.get_name())
            print("       supported types: %s" % algorithm.get_supported_data_types())
        
            for supported_tag in algorithm.get_supported_data_types():
                #REMOVE AFTER TEST
                if supported_tag != "numerical":
                    continue
                
                for sensitive in all_sensitive_attributes:

                    print("Sensitive attribute:" + sensitive)

                    detailed_files = dict((k, create_detailed_file(
                                                dataset_obj.get_hyperparametrization_results_filename(sensitive, k, missing_rate,imputation_method,missing_mechanism),
                                                dataset_obj,
                                                processed_dataset.get_sensitive_values(k), k))
                        for k in train_test_splits.keys())


                    for i in range(0, num_trials):
                        train, test = train_test_splits[supported_tag][i]

                        frames =  [train, test]  #SHHHIIIIIIIIT 
                        train_test = pd.concat(frames)
                        train_test = train_test.sort_index()
                        f = dataset_obj.get_hyperparametrization_imputation_data_filename(algorithm.get_name(),get_params(algorithm),supported_tag,missing_rate,start_run + i,imputation_method, missing_mechanism)
                        train.to_csv(f)

                        '''
                        try:
                            

                            predicted, params,prob_predictions, predictions_list, actual, dict_sensitive_lists, privileged_vals, positive_val =  \
                                run_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                            all_sensitive_attributes, supported_tag, dataset_obj.get_hyperparametrization_results_data_filename(algorithm.get_name(),get_params(algorithm),supported_tag,missing_rate,start_run + i,imputation_method,missing_mechanism))
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        '''
                        predicted, params,prob_predictions, predictions_list, actual, dict_sensitive_lists, privileged_vals, positive_val =  \
                                run_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                            all_sensitive_attributes, supported_tag, dataset_obj.get_hyperparametrization_results_data_filename(algorithm.get_name(),get_params(algorithm),supported_tag,missing_rate,start_run + i,imputation_method,missing_mechanism))
                        

                        params, results, param_results = \
                            metric_eval_alg(predicted, params, prob_predictions, predictions_list, actual, dict_sensitive_lists, privileged_vals, positive_val, dataset_obj, 
                            processed_dataset,sensitive, supported_tag)
                        
                        
                        
                        write_alg_results(detailed_files[supported_tag],
                                        algorithm.get_name(), params, start_run + i, results,algorithm)

                    print("Results written to:")
                    print("    %s" % dataset_obj.get_hyperparametrization_results_filename(sensitive, supported_tag, missing_rate,imputation_method,missing_mechanism))


                    lock.acquire()
                    for detailed_file in detailed_files.values():
                        detailed_file.close()
                    lock.release()

def write_alg_results(file_handle, alg_name, params, run_id, results_list,algorithm):
    line = alg_name + ','
    params = get_params(algorithm)
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    #print(line)
    file_handle.write(line)

def run_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                tag, filename):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()
    
    class_attr = dataset.get_class_attribute()
    params = algorithm.get_default_params()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    predicted, prob_predictions, predictions_list, X_test =  \
        algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes, privileged_vals, params)

    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()
    
    
    
    #Save results to csv
    data_array = np.array([actual,predicted,prob_predictions])
    data_array = np.concatenate((data_array,[np.array(dict_sensitive_lists[sens]) for sens in dict_sensitive_lists]))
    #print("-----------------------------------------")
    #print(data_array)
    #print("-----------------------------------------")
    #print(X_test.reset_index(drop=True))

    data_to_csv = pd.DataFrame(data=np.transpose(data_array),columns=['actual', 'predicted', 'probability']+all_sensitive_attributes)
    
    #Reset indice of X_Test for merge
    X_test = X_test.reset_index(drop=True)

    data_to_csv = pd.concat([data_to_csv,X_test.reset_index(drop=True)],axis=1)

    #print(data_to_csv)
    data_to_csv.to_csv(filename,index=False)

    
    return predicted, params,prob_predictions, predictions_list, actual, dict_sensitive_lists, privileged_vals, positive_val


def metric_eval_alg(predicted, params, prob_predictions, predictions_list, actual, dict_sensitive_lists, privileged_vals, positive_val, dataset, processed_data,
                 single_sensitive, tag):
    """
    Calculates Metrics for the runs performed
    """

    sensitive_dict = processed_data.get_sensitive_values(tag)
    one_run_results = []
    #print(dataset)
    #print(sensitive_dict)
    #print(tag)
    for metric in get_metrics(dataset, sensitive_dict, tag):
        #print(metric.name)
        result = metric.calc(actual, predicted,prob_predictions, dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        #("Result: "+str(result)+"\n")
        one_run_results.append(result)

    # handling the set of predictions returned by ParamGridSearch
    results_lol = []
    if len(predictions_list) > 0:
        for param_name, param_val, predictions in predictions_list:
            params_dict = { param_name : param_val }
            results = []
            for metric in get_metrics(dataset, sensitive_dict, tag):
                result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                                     privileged_vals, positive_val)
                results.append(result)
            results_lol.append( (params_dict, results) )

    return params, one_run_results, results_lol



def get_dict_sensitive_vals(dict_sensitive_lists):
    """
    Takes a dictionary mapping sensitive attributes to lists in the test data and returns a
    dictionary mapping sensitive attributes to lists containing each sensitive value only once.
    """
    newdict = {}
    for sens in dict_sensitive_lists:
         sensitive = dict_sensitive_lists[sens]
         newdict[sens] = list(set(sensitive))
    return newdict

def create_detailed_file(filename, dataset, sensitive_dict, tag):
    return results.ResultsFile(filename, dataset, sensitive_dict, tag)
    # f = open(filename, 'w')
    # f.write(get_detailed_metrics_header(dataset, sensitive_dict, tag) + '\n')
    # return f

def f_sum(a, b):
    return a + b

def init(l):
    global lock
    lock = l



def main():


    #run(['german'],Random_Forest(200, None, 'auto'),1,0,"knn","mcar")
    #run(['credit-dataset'],['SVM'],1,0,"knn")
    
    
    #imputation_method_lst = ["mean","knn","mice"]
    imputation_method_lst = ["mean"]
    dataset_lst= get_dataset_names()
    #dataset_lst= ["german"]
    algorithms_lst = ['SVM', 'Random_Forest']
    num_trials = 10
    missing_mechaninsms = ["mcar","mar","mnar"]
    start_run_lst = [0,10,20]

    # SVM HyperParametrization
    c_lst = [0.5, 1, 4, 16, 32]
    gamma_lst = ["auto", "scale"]

    # Random Forest HyperParametrization
    n_estimators_lst = [100, 200, 400, 800, 1600]
    max_depth_lst =  [20, 50, 100, None]
    max_features_lst = ['auto', 'sqrt']

    run_args = []
    for imputation_method in imputation_method_lst:
        for dataset in dataset_lst: 
            for alg_name in algorithms_lst:
                if alg_name == "SVM":
                    for c in c_lst:
                        for gamma in gamma_lst:
                            algorithm = SVM(c,gamma)
                            for start_run in start_run_lst:
                                temp_tuple = ([dataset],algorithm,num_trials,start_run,imputation_method,"mcar","oversample_then_imputation")
                                run_args.append(temp_tuple)
                elif alg_name == "Random_Forest":
                    for n_estimators in n_estimators_lst:
                        for max_depth in max_depth_lst:
                            for max_features in max_features_lst:
                                algorithm = Random_Forest(n_estimators,max_depth,max_features)
                                for start_run in start_run_lst:
                                    temp_tuple = ([dataset],algorithm,num_trials,start_run,imputation_method,"mcar","oversample_then_imputation")
                                    run_args.append(temp_tuple)

    l = mp.Lock()

    pool = mp.Pool(mp.cpu_count()-2,initializer=init, initargs=(l,))


    pool.starmap(run,run_args)
    
    
    
    #result = pool.map(run, [4,2,3])

if __name__ == '__main__':
    main()
