from metrics.Metric import Metric
from sklearn.metrics import confusion_matrix
from metrics.utils import compute_roc
import numpy as np
import pandas as pd
from abroca import compute_abroca

class ABROCA(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'ABROCA'

    def calc(self, actual, predicted, prob_predictions, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):

        #print(self.name)
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        sensitive_values = list(set(sensitive))

        # this list should only have one item in it

        single_unprotected = [val for val in sensitive_values if val in unprotected_vals][0]
        processed_actual = []
        for val in actual:
            if val == positive_pred:
                processed_actual.append(1);
            else:
                processed_actual.append(0);



        abroca_np = np.array([prob_predictions,processed_actual,sensitive]);
        #print("Abroca_np \n"+str(abroca_np))

        abroca_df = pd.DataFrame(abroca_np.transpose(), columns=['prob', 'actual', 'sensitive'])
        #print("abroca_df \n" +str(abroca_df))

        
        abroca_df['prob'] = abroca_df['prob'].astype(float)
        abroca_df['actual'] = abroca_df['actual'].astype(float)
        #print("DataFrame Prob")
        #print(abroca_df['prob'])
        #print("DataFrame actual")
        #print(abroca_df['actual'])
        #print("DataFrame Sensitive")
        #print(abroca_df['sensitive'])
        #print("Single unprotected")
        #print(single_unprotected)


        slice = compute_abroca(abroca_df,pred_col = "prob", label_col = "actual", 
            protected_attr_col = "sensitive", compare_type="multiple", majority_protected_attr_val = single_unprotected)
        print(slice)
        #print("\n\n\n");
        abroca_value=np.array(list(slice.values())).mean()
        #print(abroca_value)
        #print("\n\n\n");
        return abroca_value



