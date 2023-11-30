from cmath import isnan
import pandas as pd
import numpy as np
from data.objects.Data import Data
from sklearn.preprocessing import OrdinalEncoder

class Student_Por(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'student-por'
        self.class_attr = 'G3'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_class_names = ['M', 'teenager']
        self.categorical_features = ['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                                    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                                    'higher', 'internet', 'romantic']
        self.features_to_keep = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                                'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                                'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                                'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                                'Walc', 'health', 'absences', 'G3']
        self.categorical_nominal_features = ['school', 'address', 'Pstatus', 'Mjob', 'Fjob',
                                    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                                    'higher', 'internet', 'romantic']
        self.categorical_ordinal_features = ['famsize']
        self.missing_val_indicators = []



    def convert_ordinal_2_numerical(self, dataframe):
        '''
        This function will correct problems with the scalling of categorical ordinal features
        '''
        #Get ordinal features from dataset
        ordinal_features = self.get_categorical_ordinal_features() 
        
        #Ordinal Encoder
        encoder = OrdinalEncoder()

        #Put no savings in it's correct place
        dataframe['famsize'] = dataframe['famsize'].replace("LE3", "AE3", regex=True)

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
        
        dataframe['famsize'] = dataframe['famsize'].replace("AE3", "LE3", regex=True)

        return dataframe

