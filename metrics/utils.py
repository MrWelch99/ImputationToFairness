import sklearn.metrics as metrics


def calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns P(C=YES|sensitive=privileged) and P(C=YES|sensitive=not privileged) in that order where
    C is the predicited classification and where all not privileged values are considered
    equivalent.  Assumes that predicted and sensitive have the same lengths.
    """
    unprotected_positive = 0.0
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        if protected_val in unprotected_vals:
            if str(predicted_val) == str(positive_pred):
                unprotected_positive += 1
            else:
                unprotected_negative += 1
        else:
            if str(predicted_val) == str(positive_pred):
                protected_positive += 1
            else:
                protected_negative += 1

    protected_pos_percent = 0.0
    if protected_positive + protected_negative > 0:
        protected_pos_percent = protected_positive / (protected_positive + protected_negative)
    unprotected_pos_percent = 0.0
    if unprotected_positive + unprotected_negative > 0:
        unprotected_pos_percent = unprotected_positive /  \
                                  (unprotected_positive + unprotected_negative)

    return unprotected_pos_percent, protected_pos_percent


def calc_prob_class_given_sensitive(predicted, sensitive, predicted_goal, sensitive_goal):
    """
    Returns P(predicted = predicted_goal | sensitive = sensitive_goal).  Assumes that predicted
    and sensitive have the same length.  If there are no attributes matching the given
    sensitive_goal, this will error.
    """
    match_count = 0.0
    total = 0.0
    for sens, pred in zip(sensitive, predicted):
        if str(sens) == str(sensitive_goal):
            total += 1
            if str(pred) == str(predicted_goal):
                match_count += 1

    return match_count / total

def calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns False positive and false negative for protected and unprotected group.
    """
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    fp_protected = 0.0
    fp_unprotected = 0.0
    fn_protected=0.0
    fn_unprotected=0.0
    fp_diff =0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        actual_val= actual[i]
        if protected_val in unprotected_vals:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fp_unprotected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fn_unprotected+=1
        else:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                    fp_protected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                    fn_protected+=1
    return fp_unprotected,fp_protected, fn_protected, fn_unprotected


def calc_tp_fp_tn_fn_protected(actual, predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns True positive False positive and false negative for protected and unprotected group.
    """
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    tp_protected = 0.0
    tp_unprotected = 0.0
    fp_protected = 0.0
    fp_unprotected = 0.0
    tn_protected = 0.0
    tn_unprotected = 0.0
    fn_protected=0.0
    fn_unprotected=0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        actual_val= actual[i]
        if protected_val in unprotected_vals:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)==str(predicted_val)):
                tp_unprotected+=1
            elif (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fp_unprotected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
                tn_unprotected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fn_unprotected+=1
        else:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)==str(predicted_val)):
                tp_protected+=1
            elif (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fp_protected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
                tn_protected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fn_protected+=1
    return tp_unprotected, tp_protected, fp_unprotected,fp_protected, tn_protected, tn_unprotected,fn_protected, fn_unprotected


def calc_tp_fp_tn_fn(actual, predicted, positive_pred):
    """
    Returns True positive False positive and false negative for protected and unprotected group.
    """
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0

    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(0, len(predicted)):
        predicted_val = predicted[i]
        actual_val= actual[i]

        if (str(predicted_val)==str(positive_pred))&(str(actual_val)==str(predicted_val)):
            tp+=1
        elif (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
            fp+=1
        elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
            tn+=1
        elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
            fn+=1
    return tp, fp, tn, fn

def calc_b(actual, predicted, positive_pred):
    """
    Returns True positive False positive and false negative for protected and unprotected group.
    """
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0


    b=[]
    for i in range(0, len(predicted)):
        predicted_val = predicted[i]
        actual_val= actual[i]
        if (str(predicted_val)==str(positive_pred))&(str(actual_val)==str(predicted_val)):
            b.append(1) 
        elif (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
            b.append(2)
        elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
            b.append(1)
        elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)!=str(predicted_val)):
            b.append(0)
    return b


def compute_roc(y_scores, y_true, pos_label):
    """
    Function to compute the Receiver Operating Characteristic (ROC) curve for a set of predicted probabilities and the true class labels.
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns FPR and TPR values
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores,pos_label=pos_label)
    return fpr, tpr

