from metrics.Metric import Metric
from sklearn.metrics import recall_score

class Recall(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'recall'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return recall_score(actual, predicted)
