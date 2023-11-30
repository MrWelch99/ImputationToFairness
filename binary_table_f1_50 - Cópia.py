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

def make_binary_table():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    finalDF =  pandas.DataFrame(None, columns=['Dataset', 'Sensitive', 'Missing Mechanism', 'Imputation Method', 'Algorithm', 'Missing rate','Mean Accuracy', 'Mean Precision',
     'Mean Recall', 'Mean F1-Score', 'Mean DI BI', "Mean Equal Opportunity", "Mean Equal Mis-Opportunity", 'Mean CV', "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index'])
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI', "Mean Equal Opportunity",
             "Mean Equal Mis-Opportunity",  'Mean CV', "Mean Calibration+", "Mean Calibration-",  'Mean Generalized_Entropy_Index']
            
            for missing_mechanism in missing_mechanism_lst:
                for imputation_method in imputation_method_lst:

                    #finalDF =  pandas.DataFrame(None, columns=['Dataset','Sensitive','Missing Mechanism','Imputation Method','Algorithm', 'Missing rate', 'Mean Accuracy', 'Mean TPR',  'Mean TNR',  'Mean DI BI',  
                    #                "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', "Mean Calibration+", "Mean Calibration-",  'Mean Generalized_Entropy_Index'])
                    
                    file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_oversample_then_imputation_numerical.csv"                
                    
                    if os.path.isfile(file_name):

                        for algorithm in algorithm_lst:
                            for missing_rate in missing_rates_lst:
                                df = pandas.read_csv(file_name)
                                df_baseline = df.loc[(df['Algorithm'] == algorithm )& (df['Missing rate'] == 0)]
                                df = df.loc[(df['Algorithm'] == algorithm )& (df['Missing rate'] == missing_rate)]
                                if df_baseline['Mean F1-Score'].values[0] >0.5 and df['Mean F1-Score'].values[0] >0.5:                                    
                                    row = np.array([dataset_name,sensitive,missing_mechanism,imputation_method,algorithm,missing_rate])
                                    ind=0
                                    for metric in metric_lst:
                                        if ind<=3:
                                            if df[metric].values[0] > df_baseline[metric].values[0]:
                                                row = np.append(row,[1])
                                            else:
                                                row = np.append(row,[0])
                                        else:
                                            dist = abs(1-df[metric].values[0])
                                            dist_baseline = abs(1-df_baseline[metric].values[0])
                                            if dist < dist_baseline:
                                                row = np.append(row,[1])
                                            else:
                                                row = np.append(row,[0])
                                        ind=ind+1
                                    

                                    finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)


                    else:
                        print ("File not exist")

    df = finalDF.to_csv("results/Processed/Binary_table_oversample_then_imputation_f1_50.csv",index=False)

def make_continuous_table():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()
    
    print(missing_rates_lst)
    print(algorithm_lst)
    print(dataset_lst)
    
    finalDF =  pandas.DataFrame(None, columns=['Dataset', 'Sensitive', 'Missing Mechanism', 'Imputation Method', 'Algorithm', 'Missing rate','Mean Accuracy', 'Mean Precision',
     'Mean Recall', 'Mean F1-Score', 'Mean DI BI', "Mean Equal Opportunity", "Mean Equal Mis-Opportunity", 'Mean CV', "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index'])
    #We need to get the dataset, the algorithm, the sensitive attribute and the missing rate in order to cycle through them
    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()
        print("\n"+dataset_name)
        sensitive_lst =  dataset.get_sensitive_attributes()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI', "Mean Equal Opportunity",
             "Mean Equal Mis-Opportunity",  'Mean CV', "Mean Calibration+", "Mean Calibration-",  'Mean Generalized_Entropy_Index']
            
            for missing_mechanism in missing_mechanism_lst:
                for imputation_method in imputation_method_lst:

                    #finalDF =  pandas.DataFrame(None, columns=['Dataset','Sensitive','Missing Mechanism','Imputation Method','Algorithm', 'Missing rate', 'Mean Accuracy', 'Mean TPR',  'Mean TNR',  'Mean DI BI',  
                    #                "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', "Mean Calibration+", "Mean Calibration-",  'Mean Generalized_Entropy_Index'])
                    
                    file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_oversample_then_imputation_numerical.csv"                
                    
                    if os.path.isfile(file_name):

                        for algorithm in algorithm_lst:
                            for missing_rate in missing_rates_lst:
                                df = pandas.read_csv(file_name)
                                df_baseline = df.loc[(df['Algorithm'] == algorithm )& (df['Missing rate'] == 0)]
                                df = df.loc[(df['Algorithm'] == algorithm )& (df['Missing rate'] == missing_rate)]
                                if df_baseline['Mean F1-Score'].values[0] >0.5 and df['Mean F1-Score'].values[0] >0.5: 
                                
                                    row = np.array([dataset_name,sensitive,missing_mechanism,imputation_method,algorithm,missing_rate])
                                    ind=0
                                    for metric in metric_lst:
                                        if ind<=3:
                                            row = np.append(row,[df[metric].values[0] - df_baseline[metric].values[0]])
                                        else:
                                            dist = abs(1-df[metric].values[0])
                                            dist_baseline = abs(1-df_baseline[metric].values[0])
                                            row = np.append(row,[(1-dist) - (1-dist_baseline)])

                                        ind=ind+1
                                    

                                    finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

                    else:
                        print ("File not exist")

    df = finalDF.to_csv("results/Processed/Continuous_table_oversample_then_imputation_f1_50.csv",index=False)


def analyse_parameter_binary(metric_lst,df_temp,row):
    #Calc freq and std of ones
    for metric in metric_lst:
        values = df_temp[metric].values
        
        #Count ones
        freq_of_1 = sum(values)/len(values)

        #This is very scuffed --- Change after reunion with prof 
        #std_of_1 = np.std(values)
        
        row = np.append(row,[freq_of_1])
    return row


def analyse_table():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Binary_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns=['Paramater', 'Accuracy Freq', 'Precision Freq', 'Recall Freq', 'F1-Score Freq', 'DI BI Freq',
                                    "Equal Opportunity Freq", "Equal Mis-Opportunity Freq", 'CV Freq', "Calibration+ Freq", "Calibration- Freq",
                                    'Mean Generalized_Entropy_Index Freq'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()

        df_temp = df.loc[(df['Dataset'] == dataset_name)]

        row = np.array([dataset_name])

        row = analyse_parameter_binary(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    finalDF = finalDF.append(pandas.Series(), ignore_index = True)
     
    for missing_mechanism in missing_mechanism_lst:
        df_temp = df.loc[(df['Missing Mechanism'] == missing_mechanism)]


        row = np.array([missing_mechanism])

        row = analyse_parameter_binary(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    for imputation_method in imputation_method_lst:
        df_temp = df.loc[(df['Imputation Method'] == imputation_method)]

        row = np.array([imputation_method])

        row = analyse_parameter_binary(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    for algorithm in algorithm_lst:
        df_temp = df.loc[(df['Algorithm'] == algorithm)]

        row = np.array([algorithm])

        row = analyse_parameter_binary(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    for missing_rate in missing_rates_lst:
        df_temp = df.loc[(df['Missing rate'] == missing_rate)]


        row = np.array([missing_rate])

        row = analyse_parameter_binary(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    df = finalDF.to_csv("results/Processed/Binary_table_oversample_then_imputation_processed_f1_50.csv",index=False)







'''
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------******PAIRWISE******-------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------

'''


def analyse_table_pairwise():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Binary_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns=['Paramater', 'Missing Rate', 'Accuracy Freq', 'Precision Freq', 'Recall Freq', 'F1-Score Freq',
                                    'DI BI Freq', "Equal Opportunity Freq", "Equal Mis-Opportunity Freq",  'CV Freq', "Calibration+ Freq", "Calibration- Freq",
                                    'Mean Generalized_Entropy_Index Freq'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score',  'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()

        for missing_rate in missing_rates_lst:


            df_temp = df.loc[(df['Dataset'] == dataset_name) & (df['Missing rate'] == missing_rate)]


            row = np.array([dataset_name,str(missing_rate)])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    for missing_mechanism in missing_mechanism_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Missing Mechanism'] == missing_mechanism) & (df['Missing rate'] == missing_rate)]


            row = np.array([missing_mechanism,str(missing_rate)])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 


    for imputation_method in imputation_method_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing rate'] == missing_rate)]


            row = np.array([imputation_method,str(missing_rate)])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    for algorithm in algorithm_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Algorithm'] == algorithm)& (df['Missing rate'] == missing_rate)]


            row = np.array([algorithm,missing_rate])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    for imputation_method in imputation_method_lst:
        for missing_mechanism in missing_mechanism_lst:
            df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing Mechanism'] == missing_mechanism)]


            row = np.array([imputation_method,missing_mechanism])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    for missing_mechanism in missing_mechanism_lst:
        for imputation_method in imputation_method_lst:
            df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing Mechanism'] == missing_mechanism)]


            row = np.array([missing_mechanism,imputation_method])

            row = analyse_parameter_binary(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    df = finalDF.to_csv("results/Processed/Binary_table_pairwise_oversample_then_imputation_processed_f1_50.csv",index=False)

def analyse_table_three_pairs():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Binary_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns=['Imputation Method', 'Missing Mechanism','Missing Rate', 'Accuracy Freq', 'Precision Freq', 'Recall Freq', 'F1-Score Freq', 'DI BI Freq',  
                                    "Equal Opportunity Freq", "Equal Mis-Opportunity Freq",  'CV Freq', "Calibration+ Freq", "Calibration- Freq",
                                    'Mean Generalized_Entropy_Index Freq'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score',  'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for imputation_method in imputation_method_lst:
        for missing_mechanism in missing_mechanism_lst:
            for missing_rate in missing_rates_lst:
                df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing Mechanism'] == missing_mechanism) & (df['Missing rate'] == missing_rate)]


                row = np.array([imputation_method,missing_mechanism,missing_rate])

                row = analyse_parameter_binary(metric_lst,df_temp,row)

                finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
            finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    df = finalDF.to_csv("results/Processed/Binary_table_three_pairs_oversample_then_imputation_processed_f1_50.csv",index=False)



'''
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------******CONTINUOUS******-------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''

def analyse_parameter_continuous(metric_lst,df_temp,row):
    #Calc freq and std of ones
    for metric in metric_lst:
        values = df_temp[metric].values
        
        #Calc Mean
        mean = sum(values)/len(values)

        #Calc STD
        std = np.std(values)
        
        row = np.append(row,[mean,std])
    return row


def analyse_continuous_table():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Continuous_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns= ['Paramater', 'Accuracy Freq', 'Accuracy STD', 'Precision Freq', 'Precision STD', 'Recall Freq', 'Recall STD', 'F1-Score Freq', 'F1-Score STD',
                                    'DI BI Freq', 'DI BI STD', "Equal Opportunity Freq", "Equal Opportunity STD", "Equal Mis-Opportunity Freq", "Equal Mis-Opportunity STD",  'CV Freq', 'CV STD',
                                    "Calibration+ Freq", "Calibration+ STD", "Calibration- Freq", "Calibration- STD", 'Mean Generalized_Entropy_Index Freq', 'Mean Generalized_Entropy_Index STD'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()

        df_temp = df.loc[(df['Dataset'] == dataset_name)]

        row = np.array([dataset_name])

        row = analyse_parameter_continuous(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

        
    for missing_mechanism in missing_mechanism_lst:
        df_temp = df.loc[(df['Missing Mechanism'] == missing_mechanism)]

        row = np.array([missing_mechanism])

        row = analyse_parameter_continuous(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    for imputation_method in imputation_method_lst:
        df_temp = df.loc[(df['Imputation Method'] == imputation_method)]

        row = np.array([imputation_method])

        row = analyse_parameter_continuous(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    for algorithm in algorithm_lst:
        df_temp = df.loc[(df['Algorithm'] == algorithm)]

        row = np.array([algorithm])

        row = analyse_parameter_continuous(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    for missing_rate in missing_rates_lst:
        df_temp = df.loc[(df['Missing rate'] == missing_rate)]

        row = np.array([missing_rate])

        row = analyse_parameter_continuous(metric_lst,df_temp,row)

        finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)

    df = finalDF.to_csv("results/Processed/Continuous_table_oversample_then_imputation_processed_f1_50.csv",index=False)

def analyse_table_pairwise_continuous():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Continuous_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns= ['Paramater_1', 'Paramater_2', 'Accuracy Freq', 'Accuracy STD', 'Precision Freq', 'Precision STD', 'Recall Freq', 'Recall STD', 'F1-Score Freq', 'F1-Score STD',
                                    'DI BI Freq', 'DI BI STD', "Equal Opportunity Freq", "Equal Opportunity STD", "Equal Mis-Opportunity Freq", "Equal Mis-Opportunity STD",  'CV Freq', 'CV STD',
                                    "Calibration+ Freq", "Calibration+ STD", "Calibration- Freq", "Calibration- STD", 'Mean Generalized_Entropy_Index Freq', 'Mean Generalized_Entropy_Index STD'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_lst:
                continue
        
        dataset_name = dataset.get_dataset_name()

        for missing_rate in missing_rates_lst:


            df_temp = df.loc[(df['Dataset'] == dataset_name) & (df['Missing rate'] == missing_rate)]


            row = np.array([dataset_name,str(missing_rate)])

            row = analyse_parameter_continuous(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True)

    for missing_mechanism in missing_mechanism_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Missing Mechanism'] == missing_mechanism) & (df['Missing rate'] == missing_rate)]


            row = np.array([missing_mechanism,str(missing_rate)])

            row = analyse_parameter_continuous(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 


    for imputation_method in imputation_method_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing rate'] == missing_rate)]


            row = np.array([imputation_method,str(missing_rate)])

            row = analyse_parameter_continuous(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    for algorithm in algorithm_lst:
        for missing_rate in missing_rates_lst:
            df_temp = df.loc[(df['Algorithm'] == algorithm)& (df['Missing rate'] == missing_rate)]


            row = np.array([algorithm,missing_rate])

            row = analyse_parameter_continuous(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    for imputation_method in imputation_method_lst:
        for missing_mechanism in missing_mechanism_lst:
            df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing Mechanism'] == missing_mechanism)]


            row = np.array([imputation_method,missing_mechanism])

            row = analyse_parameter_continuous(metric_lst,df_temp,row)

            finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
        finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    df = finalDF.to_csv("results/Processed/Continuous_table_pairwise_oversample_then_imputation_processed_f1_50.csv",index=False)

def analyse_table_three_pairs_continuous():
    folder = "results/Processed/"
    missing_rates_lst = [0.05,0.1,0.2,0.4]
    missing_mechanism_lst =  ["mar","mnar","true_mcar"]
    imputation_method_lst = ["mean","knn","mice"]
    dataset_lst = get_dataset_names()
    algorithm_lst = get_algorithm_names()

    df =  pandas.read_csv("results/Processed/Continuous_table_oversample_then_imputation_f1_50.csv")

    finalDF =  pandas.DataFrame(None, columns= ['Paramater_1', 'Paramater_2', 'Paramater_3', 'Accuracy Freq', 'Accuracy STD', 'Precision Freq', 'Precision STD', 'Recall Freq', 'Recall STD', 'F1-Score Freq', 'F1-Score STD',
                                    'DI BI Freq', 'DI BI STD', "Equal Opportunity Freq", "Equal Opportunity STD", "Equal Mis-Opportunity Freq", "Equal Mis-Opportunity STD",  'CV Freq', 'CV STD',
                                    "Calibration+ Freq", "Calibration+ STD", "Calibration- Freq", "Calibration- STD", 'Mean Generalized_Entropy_Index Freq', 'Mean Generalized_Entropy_Index STD'])

    metric_lst = ['Mean Accuracy', 'Mean Precision',  'Mean Recall', 'Mean F1-Score', 'Mean DI BI',  
                                    "Mean Equal Opportunity", "Mean Equal Mis-Opportunity",  'Mean CV', 
                                    "Mean Calibration+", "Mean Calibration-", 'Mean Generalized_Entropy_Index']

    for imputation_method in imputation_method_lst:
        for missing_mechanism in missing_mechanism_lst:
            for missing_rate in missing_rates_lst:
                df_temp = df.loc[(df['Imputation Method'] == imputation_method) & (df['Missing Mechanism'] == missing_mechanism) & (df['Missing rate'] == missing_rate)]


                row = np.array([imputation_method,missing_mechanism,missing_rate])

                row = analyse_parameter_continuous(metric_lst,df_temp,row)

                finalDF = finalDF.append(pandas.DataFrame(row.reshape(1,-1), columns=list(finalDF)), ignore_index=True)
            finalDF = finalDF.append(pandas.Series(), ignore_index = True)
    finalDF = finalDF.append(pandas.Series(), ignore_index = True) 

    df = finalDF.to_csv("results/Processed/Continuous_table_three_pairs_oversample_then_imputation_processed_f1_50.csv",index=False)



def main():
    #make_binary_table()
    #make_continuous_table()

    #analyse_table()
    analyse_table_pairwise()
    #analyse_table_three_pairs()

    #analyse_continuous_table()
    #analyse_table_pairwise_continuous()
    #analyse_table_three_pairs_continuous()

main()