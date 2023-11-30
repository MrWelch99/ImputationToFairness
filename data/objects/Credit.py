from data.objects.Data import Data
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
	
class Credit(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'credit-dataset'
        self.class_attr = 'default payment next month'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'education', 'marriage']
        self.privileged_class_names = [1, 2, 2]
        self.categorical_features = ['pay-0', 'pay-2','pay-3', 'pay-4', 'pay-5', 'pay-6']
        self.features_to_keep = ['limit-bal', 'sex', 'education', 'marriage', 'age', 'pay-0', 'pay-2',
       							'pay-3', 'pay-4', 'pay-5', 'pay-6', 'bill-amt1', 'bill-amt2',
       							'bill-amt3', 'bill-amt4', 'bill-amt5', 'bill-amt6', 'pay-amt1',
       							'pay-amt2', 'pay-amt3', 'pay-amt4', 'pay-amt5', 'pay-amt6',
       							'default payment next month']
        self.categorical_nominal_features = []
        self.categorical_ordinal_features = ['pay-0', 'pay-2','pay-3', 'pay-4', 'pay-5', 'pay-6']
        self.missing_val_indicators = ['?']


    def convert_ordinal_2_numerical(self, dataframe):
        '''
        This function will correct problems with the scalling of categorical ordinal features
        '''
        #Get ordinal features from dataset
        ordinal_features = self.get_categorical_ordinal_features()
        
        #Ordinal Encoder
        encoder = OrdinalEncoder()

        ordinal_dict = {}   
        for feature in ordinal_features:
            temp = dataframe[feature].values.reshape(-1, 1)
            dataframe[feature]=encoder.fit_transform(temp)
            #print(dataframe[feature])

            #Invert scale
            feature_max = dataframe[feature].max()
            dataframe[feature] = [feature_max-num for num in dataframe[feature].values.tolist()]


            ordinal_dict[feature] = encoder.categories_

        return dataframe, ordinal_dict

    def convert_numerical_2_ordinal(self, dataframe, ordinal_dict):
        '''
        This function will correct problems with the scalling of categorical ordinal features
        '''

        #Get ordinal features from dataset
        ordinal_features = self.get_categorical_ordinal_features()

        for feature in ordinal_features:
            new_col_values = []
            feature_max = dataframe[feature].max()
            for item in dataframe[feature]:
                if not np.isnan(item):
                    #reinvert scale
                    inv_value = feature_max - item

                    new_col_values.append(ordinal_dict[feature][0][int(inv_value)])
                else:
                    new_col_values.append(np.nan)
            

            dataframe[feature] = new_col_values
        

        return dataframe
