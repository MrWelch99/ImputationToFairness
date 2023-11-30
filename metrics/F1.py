from metrics.Metric import Metric
from sklearn.metrics import f1_score

class F1(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'f1'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return f1_score(actual, predicted)
