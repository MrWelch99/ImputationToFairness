from data.objects.Data import Data
	
class KDDIncome(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'kdd-income'
        self.class_attr = 'income'
        self.positive_class_val = 1
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = [' White', ' Male']
        self.categorical_features = [ 'class of worker','education', 'marital status', 'major industry code',
                                      'member of a labor union', 'full or part time employment stat', 'federal income tax liability', 
                                      'detailed household summary in household','live in this house 1 year ago','migration prev res in sunbelt',
                                      'country of birth father','country of birth mother','country of birth self',
                                      'citizenship','own business or self employed','veterans benefits']
        self.features_to_keep = ['age', 'class of worker','education', 'wage per hour', 'marital status', 'major industry code',
                                        'race', 'sex', 'member of a labor union', 'full or part time employment stat', 'capital gains', 'capital losses',
                                        'divdends from stocks', 'federal income tax liability', 'detailed household summary in household',
                                        'live in this house 1 year ago','migration prev res in sunbelt','num persons worked for employer','country of birth father',
                                        'country of birth mother','country of birth self','citizenship','own business or self employed',
                                        'veterans benefits','weeks worked in year','year','income']
        self.categorical_nominal_features = ['class of worker','education', 'marital status', 'major industry code',
                                      'member of a labor union', 'full or part time employment stat', 'federal income tax liability', 
                                      'detailed household summary in household','live in this house 1 year ago','migration prev res in sunbelt',
                                      'country of birth father','country of birth mother','country of birth self',
                                      'citizenship','own business or self employed','veterans benefits']
        self.categorical_ordinal_features = []
        self.missing_val_indicators = [' ?']
