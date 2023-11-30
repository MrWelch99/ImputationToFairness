from metrics.Metric import Metric
from metrics.TNR import TNR

class FPR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FPR'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tnr = TNR()
        tnr_val = tnr.calc(actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        return 1 - tnr_val
