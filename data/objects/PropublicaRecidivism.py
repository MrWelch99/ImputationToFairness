from data.objects.Data import Data
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class PropublicaRecidivism(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-recidivism'
        self.class_attr = 'two-year-recid'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_class_names = ['Male', 'Caucasian']
        self.categorical_features = ['age-cat',"score-text",'c-charge-degree', 'c-charge-desc','r-charge-desc']
        # days_b_screening_arrest, score_text, decile_score, and is_recid will be dropped after
        # data specific processing is done
        self.features_to_keep = ["sex", "age", "age-cat", "race", "juv-fel-count", "juv-misd-count",
                                 "juv-other-count", "priors-count", "c-charge-degree",
                                 "c-charge-desc", "decile-score", "score-text", "two-year-recid",
                                 "days-b-screening-arrest", "is-recid","c-days-from-compas","r-days-from-arrest",
                                 'r-charge-desc']

        self.categorical_nominal_features = ['c-charge-degree', 'c-charge-desc','r-charge-desc']
        self.categorical_ordinal_features = ['age-cat',"score-text"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        '''dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        dataframe = dataframe.drop(columns = ['days_b_screening_arrest', 'is_recid',
                                              'decile_score', 'score_text'])
        '''
        return dataframe


    def convert_ordinal_2_numerical(self, dataframe):
        '''
        This function will correct problems with the scalling of categorical ordinal features
        '''
        #Get ordinal features from dataset
        ordinal_features = self.get_categorical_ordinal_features() 

        ordered_education = ['Less than 25','25 - 45','Greater than 45']
        
        ordered_education_2 = ['Low','Medium','High']
        
        #Ordinal Encoder
        encoder = OrdinalEncoder(categories=[ordered_education])

        encoder_2 = OrdinalEncoder(categories=[ordered_education_2])

        ordinal_dict = {}  
        for feature in ordinal_features:
            if feature == "age-cat":
                temp = dataframe[feature].values.reshape(-1, 1)
                dataframe[feature]=encoder.fit_transform(temp)
                #print(dataframe[feature])

                ordinal_dict[feature] = encoder.categories_
            elif feature == "score-text":
                temp = dataframe[feature].values.reshape(-1, 1)
                dataframe[feature]=encoder_2.fit_transform(temp)
                #print(dataframe[feature])

                ordinal_dict[feature] = encoder_2.categories_

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

        return dataframe