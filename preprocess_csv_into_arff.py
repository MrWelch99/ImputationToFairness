import pandas
import arff
import numpy as np
import glob
import os
from data.objects.list import DATASETS, get_dataset_names
from algorithms.list import ALGORITHMS
from metrics.list import get_metrics


def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def save_to_arff():
    arff.dump('filename.arff'
      , df.values
      , relation='relation name'
      , names=df.columns)

def main():

    # Load and save main files
    folder_original = "data/preprocessed/"
    folder_arff = "datasets-arff/"
    missing_rates_lst = [0,0.05,0.1,0.2,0.4]
    missing_mechanism =  "mar"
    imputation_method_lst = ["mean","knn"]
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
        file_name =  folder_original+dataset_name+"_numerical.csv"
        if os.path.isfile(file_name):
            df = pandas.read_csv(file_name)

            new_file_name = folder_arff+dataset_name+"_numerical.arff"
            arff.dump(new_file_name
            , df.values
            , relation='relation name'
            , names=df.columns)

            print(file_name+" was transfered to arff") 
        else:
            print("File not exist") 


        '''
        sensitive_lst =  dataset.get_sensitive_attributes_with_joint()
        print(sensitive_lst)

        for sensitive in sensitive_lst:
            metric_lst = ['accuracy', 'TPR', 'TNR', 'FPR', 'FNR', 'DIbinary', sensitive+"-TPRDiff", sensitive+"-FPRDiff", 'CV', sensitive+"-calibration+Diff",sensitive+"-calibration-Diff", 'Generalized_Entropy_Index']


            
            for imputation_method in imputation_method_lst:
                finalDF =  pandas.DataFrame(None, columns=['Algorithm', 'Missing rate', 'Mean Accuracy', 'STD Accuracy', 'Mean TPR', 'STD TPR', 'Mean TNR', 'STD TNR','Mean FPR', 'STD FPR', 'Mean FNR', 'STD FNR', 'Mean DI BI',  'STD DI BI',
                                 "Mean Equal Opportunity", "STD Equal Opportunity", "Mean Equal Mis-Opportunity", "STD Equal Mis-Opportunity", 'Mean CV',  'STD CV', "Mean Calibration+", "STD Calibration+",
                                 "Mean Calibration-", "STD Calibration-", 'Mean Generalized_Entropy_Index', 'STD Generalized_Entropy_Index'])
                for algorithm in algorithm_lst:
                        for missing_rate in missing_rates_lst:
                            file_name = folder+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_"+str(missing_rate)+"_numerical.csv"
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
                                print("File not exist")

                #print(finalDF)
                finalDF.to_csv("results/Processed/Oversampled/"+dataset_name+"_"+sensitive+"_"+missing_mechanism+"_"+imputation_method+"_numerical.csv",index=False)
        '''

                

    

main()