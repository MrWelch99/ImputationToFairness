from data.objects.list import DATASETS
from data.objects.ProcessedData import ProcessedData


def analyse_dataset():
    '''
    Check Unbalance in the datasets
    '''

    for dataset_obj in DATASETS:
        print("Dataset: '%s'" % dataset_obj.get_dataset_name())

        class_name = dataset_obj.get_class_attribute()
        positive_value = dataset_obj.get_positive_class_val("numerical")
        
        print(class_name)

        processed_dataset = ProcessedData(dataset_obj)

        class_col = processed_dataset[class_name]
        
        pos_num = 0
        for value  in class_col:
            if value == positive_value:
                pos_num += 1
        
        print("Positive Percentage: "+ str(pos_num/len(class_col)*100))

        
def count_missing():
    '''
    Count missing values in a dataset
    '''
    print("Hey")
    for dataset_obj in DATASETS:
        print("Dataset: '%s'" % dataset_obj.get_dataset_name())

        class_name = dataset_obj.get_class_attribute()
        positive_value = dataset_obj.get_positive_class_val("numerical")
        
        print(class_name)
        
        processed_dataset = ProcessedData(dataset_obj,True).dfs['numerical']

        missing_num  = processed_dataset.isna().sum().sum()

        print("Total Individuals " + str(len(processed_dataset)))
        
        print("Total missing values " + str(missing_num))
        print("Percentage of missing values " + str(missing_num/processed_dataset.size))
        

        print("Number of instances with missing values " + str(processed_dataset.isna().any(axis=1).sum()))




        



count_missing()