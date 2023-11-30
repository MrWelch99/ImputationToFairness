from cmath import isnan
import pandas as pd
import numpy as np
from data.objects.Data import Data
from sklearn.preprocessing import OrdinalEncoder

class German(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'german'
        self.class_attr = 'credit'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_class_names = ['male', 'adult']
        self.categorical_features = ['status', 'credit-history', 'purpose', 'savings', 'employment',
                                     'other-debtors', 'property', 'installment-plans',
                                     'housing', 'skill-level', 'telephone', 'foreign-worker']
        self.features_to_keep = [ 'status', 'month', 'credit-history', 'purpose', 'credit-amount',
                                  'savings', 'employment', 'investment-as-income-percentage',
                                  'personal-status', 'other-debtors', 'residence-since',
                                  'property', 'age', 'installment-plans', 'housing',
                                  'number-of-credits', 'skill-level', 'people-liable-for',
                                  'telephone', 'foreign-worker', 'credit' ]
        self.categorical_nominal_features = ['status','credit-history', 'purpose', 'other-debtors', 'property',
                                 'installment-plans', 'housing', 'skill-level', 'telephone', 'foreign-worker']
        self.categorical_ordinal_features = ['savings', 'employment']
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # adding a derived sex attribute based on personal_status
        sexdict = {'A91' : 'male', 'A93' : 'male', 'A94' : 'male',
                   'A92' : 'female', 'A95' : 'female'}
        dataframe['personal-status'] =  dataframe['personal-status'].replace(to_replace = sexdict)
        dataframe = dataframe.rename(columns = {'personal-status' : 'sex'})

        # adding a derived binary age attribute (youth vs. adult) such that >= 25 is adult
        # this is based on an analysis by Kamiran and Calders
        # http://ieeexplore.ieee.org/document/4909197/
        # showing that this division creates the most discriminatory possibilities.
        old = dataframe['age'] >= 25
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        return dataframe

    def convert_ordinal_2_numerical(self, dataframe):
        '''
        This function will correct problems with the scalling of categorical ordinal features
        '''
        #Get ordinal features from dataset
        ordinal_features = self.get_categorical_ordinal_features() 
        
        #Ordinal Encoder
        encoder = OrdinalEncoder()

        #Put no savings in it's correct place
        dataframe['savings'] = dataframe['savings'].replace("A65", "A60", regex=True)

        ordinal_dict = {}  
        for feature in ordinal_features:
            temp = dataframe[feature].values.reshape(-1, 1)
            dataframe[feature]=encoder.fit_transform(temp)
            #print(dataframe[feature])

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
            for item in dataframe[feature]:
                if not np.isnan(item):
                    new_col_values.append(ordinal_dict[feature][0][int(item)])
                else:
                    new_col_values.append(np.nan)
            
            dataframe[feature] = new_col_values
        
        dataframe['savings'] = dataframe['savings'].replace("A60","A65", regex=True)

        return dataframe

