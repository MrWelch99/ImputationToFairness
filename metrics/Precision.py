from metrics.Metric import Metric
from sklearn.metrics import precision_score

class Precision(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'precision'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return precision_score(actual, predicted)