import pandas
import numpy as np
import glob
import os
from data.objects.list import DATASETS, get_dataset_names
from algorithms.list import ALGORITHMS
from metrics.list import get_metrics

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def analyse_results():
    folder = "results/"
    missing_rates_lst = [0,0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mcar","mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['accuracy', 'precision', 'recall', 'f1', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']

            for missing_mechanism in missing_mechanism_lst:
            
                for imputation_method in imputation_method_lst:
                    finalDF =  pandas.DataFrame(None, columns=['Algorithm', 'Missing rate', 'Mean Accuracy', 'STD Accuracy', 'Mean Precision', 'STD Precision', 'Mean Recall', 'STD Recall', 'Mean F1-Score', 'STD F1-Score', 'Mean DI BI',  'STD DI BI',
                                    "Mean Equal Opportunity", "STD Equal Opportunity", "Mean Equal Mis-Opportunity", "STD Equal Mis-Opportunity", 'Mean CV',  'STD CV', "Mean Calibration+", "STD Calibration+",
                                    "Mean Calibration-", "STD Calibration-", 'Mean Generalized_Entropy_Index', 'STD Generalized_Entropy_Index'])
                    for algorithm in algorithm_lst:
                            for missing_rate in missing_rates_lst:
                                file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_imputation_then_oversample_"+str(missing_rate)+"_numerical.csv"
                                print(file_name)
                                if os.path.isfile(file_name):
                                    df = pandas.read_csv(file_name)
                                    df = df.loc[df['algorithm'] == algorithm]
                                    
                                    row = np.array([algorithm,missing_rate])

                                    for metric in metric_lst:
                                        metric_val=df[metric].values
                                        mean = sum(metric_val)/len(metric_val)
                                        std = np.std(metric_val)

                                        row = np.append(row,[mean,std])

                                    

                                    finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                                else:
                                    print ("File not exist")

                    #print(finalDF)
                    finalDF.to_csv("results/Processed/"+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_imputation_then_oversample_numerical.csv",index=False)

def analyse_results_baseline():
    folder = "results/"
    missing_rates_lst = [0,0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mcar","mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['accuracy', 'precision', 'recall', 'f1', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']
            
            finalDF =  pandas.DataFrame(None, columns=['Algorithm', 'Missing rate', 'Mean Accuracy', 'STD Accuracy', 'Mean Precision', 'STD Precision', 'Mean Recall', 'STD Recall', 'Mean F1-Score', 'STD F1-Score', 'Mean DI BI',  'STD DI BI',
                                    "Mean Equal Opportunity", "STD Equal Opportunity", "Mean Equal Mis-Opportunity", "STD Equal Mis-Opportunity", 'Mean CV',  'STD CV', "Mean Calibration+", "STD Calibration+",
                                    "Mean Calibration-", "STD Calibration-", 'Mean Generalized_Entropy_Index', 'STD Generalized_Entropy_Index'])
            
            for algorithm in algorithm_lst:

                file_name = folder+dataset_name+"_"+sensitive+"_just_oversample_numerical.csv"
                print(file_name)
                if os.path.isfile(file_name):
                    df = pandas.read_csv(file_name)
                    df = df.loc[df['algorithm'] == algorithm]
                    
                    row = np.array([algorithm,str(0)])

                    for metric in metric_lst:
                        metric_val=df[metric].values
                        mean = sum(metric_val)/len(metric_val)
                        std = np.std(metric_val)

                        row = np.append(row,[mean,std])

                    

                    finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                else:
                    print ("File not exist")

            #print(finalDF)
            finalDF.to_csv("results/Processed/"+dataset_name+"_"+sensitive+"_just_oversample_numerical.csv",index=False)


def make_clustering_datasets():
    folder = "results/"
    missing_rates_lst = [0,0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mcar","mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)

        finalDF =  pandas.DataFrame(None, columns=['Sensitive','Missing Mechanism','Imputation Method','Algorithm', 'Missing rate', 'Mean Accuracy', 'Mean TPR', 'Mean TNR', 'Mean DI BI',
                                 "Mean Equal Opportunity", "Mean Equal Mis-Opportunity", 'Mean CV',  "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index'])

        for missing_mechanism in missing_mechanism_lst:
            for sensitive in sensitive_lst:
                metric_lst = ['accuracy', 'TPR', 'TNR', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']

                
                for imputation_method in imputation_method_lst:
                    for algorithm in algorithm_lst:
                            for missing_rate in missing_rates_lst:
                                file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_oversample_then_imputation_"+str(missing_rate)+"_numerical.csv"
                                print(file_name)
                                if os.path.isfile(file_name):
                                    df = pandas.read_csv(file_name)
                                    df = df.loc[df['algorithm'] == algorithm]
                                    
                                    row = np.array([sensitive,missing_mechanism,imputation_method,algorithm,missing_rate])

                                    for metric in metric_lst:
                                        metric_val=df[metric].values
                                        mean = sum(metric_val)/len(metric_val)
                                        std = np.std(metric_val)

                                        row = np.append(row,[mean])

                                    

                                    finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                                else:
                                    print ("File not exist")

        #print(finalDF)
        finalDF.to_csv("results/Processed/Clustering/"+dataset_name+"_numerical.csv",index=False)


                
    
    
    #metrics = ['accuracy',]

def analyse_hyperparameters():
    folder = "results/"
    missing_rates_lst = [0]
    missing_mechanism_lst =  ["mcar","mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)

    # SVM HyperParametrization
    c_lst = [0.5, 1, 4, 16, 32]
    gamma_lst = ["auto", "scale"]

    # Random Forest HyperParametrization
    n_estimators_lst = [100, 200, 400, 800, 1600]
    max_depth_lst =  [20, 50, 100, None]
    max_features_lst = ['auto', 'sqrt']
    
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['accuracy', 'TPR', 'TNR', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']

            for missing_mechanism in missing_mechanism_lst:

            
                for imputation_method in imputation_method_lst:
                    finalDF =  pandas.DataFrame(None, columns=['Algorithm',"Params", 'Missing rate', 'Mean Accuracy', 'STD Accuracy', 'Mean TPR', 'STD TPR', 'Mean TNR', 'STD TNR', 'Mean DI BI',  'STD DI BI',
                                    "Mean Equal Opportunity", "STD Equal Opportunity", "Mean Equal Mis-Opportunity", "STD Equal Mis-Opportunity", 'Mean CV',  'STD CV', "Mean Calibration+", "STD Calibration+",
                                    "Mean Calibration-", "STD Calibration-", 'Mean Generalized_Entropy_Index', 'STD Generalized_Entropy_Index'])
                    for algorithm in algorithm_lst:
                        if algorithm == "SVM":
                            for c in c_lst:
                                for gamma in gamma_lst:
                                    params = str(c)+';'+gamma
                                    for missing_rate in missing_rates_lst:
                                        file_name = folder+"hyperparametrization_smote_"+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_"+str(missing_rate)+"_numerical.csv"
                                        print(file_name)
                                        if os.path.isfile(file_name):
                                            df = pandas.read_csv(file_name)
                                            df = df.loc[(df['algorithm'] == algorithm) & (df['params'] == params)]
                                            
                                            row = np.array([algorithm,params,missing_rate])

                                            for metric in metric_lst:
                                                metric_val=df[metric].values
                                                mean = sum(metric_val)/len(metric_val)
                                                std = np.std(metric_val)

                                                row = np.append(row,[mean,std])

                                            

                                            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                                        else:
                                            print ("File not exist")
                        elif algorithm == "Random_Forest":
                            for n_estimators in n_estimators_lst:
                                for max_depth in max_depth_lst:
                                    for max_features in max_features_lst:
                                        params = str(n_estimators)+";"+ str(max_depth)+";"+max_features
                                        for missing_rate in missing_rates_lst:
                                            file_name = folder+"hyperparametrization_smote_"+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_"+str(missing_rate)+"_numerical.csv"
                                            print(file_name)
                                            if os.path.isfile(file_name):
                                                df = pandas.read_csv(file_name)
                                                df = df.loc[(df['algorithm'] == algorithm) & (df['params'] == params)]
                                                
                                                row = np.array([algorithm,params,missing_rate])

                                                for metric in metric_lst:
                                                    metric_val=df[metric].values
                                                    mean = sum(metric_val)/len(metric_val)
                                                    std = np.std(metric_val)

                                                    row = np.append(row,[mean,std])

                                                

                                                finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                                            else:
                                                print ("File not exist")

                    #print(finalDF)
                    finalDF.to_csv("results/Processed/Oversampled/hyperparametrization_"+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_numerical.csv",index=False)


                
    
    
    #metrics = ['accuracy',]


def table_preprocess():
    folder = "results/"
    missing_rates_lst = [0]
    missing_mechanism_lst =  ["true_mcar"]
    imputation_method_lst = ["mean","knn"]
    pipeline_lst  = ["missing_pipeline", "missing_pipeline_baseline"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    num_run = 30
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)
        for pipeline in pipeline_lst:
            for sensitive in sensitive_lst:
                metric_lst = ['accuracy', 'precision', 'recall', 'f1', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']

                for missing_mechanism in missing_mechanism_lst:
                
                    for imputation_method in imputation_method_lst:
                        finalDF =  pandas.DataFrame(None, columns=['Algorithm', 'Run', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'DI BI',
                                        "Equal Opportunity", "Equal Mis-Opportunity", 'CV', "Calibration+","Calibration-", 'Generalized_Entropy_Index'])
                        for algorithm in algorithm_lst:
                                for missing_rate in missing_rates_lst:
                                    for run in range(num_run):
                                        if pipeline == "missing_pipeline":
                                            file_name = folder+dataset_name+"_"+sensitive+"_"+imputation_method+"_"+pipeline+"_numerical.csv"
                                        else:
                                            file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_"+pipeline+"_"+str(missing_rate)+"_numerical.csv"
                                        print(file_name)
                                        if os.path.isfile(file_name):
                                            df = pandas.read_csv(file_name)
                                            df = df.loc[(df['algorithm'] == algorithm) & (df['run-id'] == run)]
                                            
                                            row = np.array([algorithm,run])

                                            for metric in metric_lst:
                                                metric_val=df[metric].values
                                                

                                                row = np.append(row,[metric_val])

                                            

                                            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                                        else:
                                            print ("File not exist")

                        #print(finalDF)
                        finalDF.to_csv("results/Processed/raw_data_full/"+dataset_name+"_"+sensitive+"_"+"_"+imputation_method+"_missing_numerical.csv",index=False)


def main():
    table_preprocess()
main()