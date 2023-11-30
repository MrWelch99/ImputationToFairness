from metrics.Metric import Metric
from metrics.TPR import TPR

class FNR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FNR'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tpr = TPR()
        tpr_val = tpr.calc(actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        return 1 - tpr_val
