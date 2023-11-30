from data.objects.Data import Data
	
class Bank(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'bank-full'
        self.class_attr = 'y'
        self.positive_class_val = 'yes'
        self.sensitive_attrs = ['age', 'marital']
        self.privileged_class_names = ['adult', 'married']
        self.categorical_features = [ 'job', 'marital', 'education', 'default',
                                  'housing', 'loan', 'contact', 'month',
                                  'poutcome' ]
        self.features_to_keep = [ 'age','job','marital','education', 'default', 'balance',
                                  'housing', 'loan', 'contact', 'day', 'month',
                                  'duration', 'campaign', 'pdays',
                                  'previous', 'poutcome', 'y' ]
        self.missing_val_indicators = ['unknown']


    def data_specific_processing(self, dataframe):
            # adding a derived binary age attribute (youth vs. adult) such that >= 25 is adult
            # this is based on an analysis by Kamiran and Calders
            # http://ieeexplore.ieee.org/document/4909197/
            # showing that this division creates the most discriminatory possibilities.
            adult = (dataframe['age'] >= 25) & (dataframe['age'] <= 60)
            dataframe.loc[adult, 'age'] = 'adult'
            young = dataframe['age'] != 'adult'
            dataframe.loc[young, 'age'] = 'not adult'
            return dataframe