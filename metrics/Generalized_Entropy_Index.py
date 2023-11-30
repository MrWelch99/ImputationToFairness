""" Equal opportunity - Protected and unprotected False postives ratio"""
import math
import sys
import numpy

from metrics.utils import calc_b
from metrics.Metric import Metric

class Generalized_Entropy_Index(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'Generalized_Entropy_Index'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]

        
        
        ge_index=0.0
        #Alpha value
        alpha = 2
        #b_array
        #print(predicted)
        #print(actual)
        b_array = calc_b(actual, predicted, positive_pred)
        #print(b_array)
        #print(sum(b_array))
        
        mean_b =  sum(b_array)/len(b_array)

        for b in b_array:
            ge_index += pow((b/mean_b),alpha)-1
        
        ge_index = 1/(len(b_array)*alpha*(alpha-1)) *ge_index 


        return ge_index
