import numpy as np
import pandas as pd
import itertools


class MultivariateMCAR:

    @staticmethod
    def amputate(data_obj, dataset: pd.DataFrame, sensitive_atributes, missing_rate: float) -> pd.DataFrame:
        dataset = dataset.copy()
        num_obs = len(dataset)
        num_feat = len(dataset.columns.values)
        sum_mvs = round(missing_rate * num_obs * num_feat)
        
        #print("Len of Dataset:")
        #print(num_obs)
        sensitive_class_lst=[]
        
        for sen in sensitive_atributes:
           sensitive_class_lst.append(dataset[sen].unique())

        #Create pairwise list of sensitive classes
        sensitive_class_pairwise_lst=list(itertools.product(*sensitive_class_lst))
        #print(sensitive_class_pairwise_lst)

        #Create pairwise list of idx for each sensitive classes
        sensitive_class_pairwise_idx_lst=[]
        for sen in sensitive_class_pairwise_lst[:]:
            cond_lst= [True for i in range(num_obs)]
            for i in range(len(sensitive_atributes)):
                temp = dataset[sensitive_atributes[i]] == sen[i]
                cond_lst=[a and b for a, b in zip(cond_lst, temp)]
            
            # Get rid of 0 len combinations
            temp_lst = dataset.index[cond_lst].tolist()

            if len(temp_lst)!=0:
                sensitive_class_pairwise_idx_lst.append(temp_lst)
            else:
                #print("OI")
                sensitive_class_pairwise_lst.remove(sen)




        # Maximum missing rate per feature must be clipped at 90%.
        # With moderate missing rates (up to 60%), it never happens.
        max_mvs_feat_lst = []
        sum_mvs_lst = []
        #print("Sensitive_class_pairwise_idx_lst Len:")

        for sen in sensitive_class_pairwise_idx_lst:
            #print(len(sen))
            num_per_feature = round(min(missing_rate + (missing_rate / 2.0), 0.9) * len(sen))
            max_mvs_feat_lst.append(num_per_feature)
            if num_per_feature == 0:
                sum_mvs_lst.append(num_per_feature)
            else:
                sum_mvs_lst.append(round(missing_rate * len(sen) * num_feat))


        # Randomly distribute missing values by the features.
        num_mvs_per_feat = np.zeros((len(max_mvs_feat_lst),num_feat))

        for j in range(len(max_mvs_feat_lst)):
            while sum_mvs_lst[j] > 0:
                for i in range(num_feat):
                    if num_mvs_per_feat[j][i] < max_mvs_feat_lst[j]:
                        num_mvs_it = np.random.randint(0, min(max_mvs_feat_lst[j] - num_mvs_per_feat[j][i] + 1, sum_mvs_lst[j] + 1))
                        sum_mvs_lst[j] -= num_mvs_it
                        num_mvs_per_feat[j][i] += num_mvs_it
                        if sum_mvs_lst[j] == 0:
                            break

        # Amputate the values.
        for j in range(len(max_mvs_feat_lst)):
            for i, col in enumerate(dataset.columns.values):
                num_mv = round(num_mvs_per_feat[j][i])
                num_mv = num_mv if num_mv > 0 else 0
                #print(len(sensitive_class_pairwise_idx_lst[j]))
                idx_nan_rows = np.random.choice(len(sensitive_class_pairwise_idx_lst[j]), int(num_mv), replace=False)
                dataset.loc[np.array(sensitive_class_pairwise_idx_lst[j])[idx_nan_rows], col] = np.nan

        return dataset


# Example with the Iris dataset.
if __name__ == '__main__':
    iris_ds = pd.read_csv('data/preprocessed/ricci_numerical_binsensitive.csv')
    iris_ds = MultivariateMCAR.amputate(iris_ds, missing_rate=0.4)
    print(iris_ds, end="\n\n")

    sum_mvs_cols = 0
    for c in iris_ds.columns.values:
        num_mvs = pd.isnull(iris_ds.loc[:, c]).values.astype(int).sum()
        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(iris_ds)) * 100)}%).")
        sum_mvs_cols += num_mvs

    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(iris_ds) * len(iris_ds.columns.values))) * 100)}%.")
