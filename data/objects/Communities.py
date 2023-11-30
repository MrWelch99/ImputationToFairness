from data.objects.Data import Data
	
class Communities(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'communities.csv'
        self.class_attr = 'ViolentCrimesPerPop'
        self.positive_class_val = high_crime
        self.sensitive_attrs = ['race']
        self.privileged_class_names = ['black']
        self.categorical_features = ['pay_0', 'pay_2','pay_3', 'pay_4', 'pay_5', 'pay_6']
        self.features_to_keep = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2',
       							'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2',
       							'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1',
       							'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
       							'default payment next month']
        self.missing_val_indicators = ['?']
