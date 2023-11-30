from data.objects.Data import Data
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

	
class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'income-per-year'
        self.positive_class_val = 1
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = [' White', ' Male']
        self.categorical_features = [ 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country' ]
        self.features_to_keep = [ 'age', 'workclass', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                  'capital-loss', 'hours-per-week', 'native-country',
                                  'income-per-year' ]
        self.categorical_nominal_features = ['workclass','marital-status','occupation', 'relationship', 'native-country','education']
        self.categorical_ordinal_features = []
        self.missing_val_indicators = [' ?']