import numpy as np
import pandas as pd


class TrueMultivariateMCAR:

    @staticmethod
    def amputate(dataset: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
        dataset = dataset.copy()
        num_obs = len(dataset)
        num_feat = len(dataset.columns.values)
        sum_mvs = round(missing_rate * num_obs * num_feat)

        # Maximum missing rate per feature must be clipped at 90%.
        # With moderate missing rates (up to 60%), it never happens.
        max_mvs_feat = round(min(missing_rate + (missing_rate / 2.0), 0.9) * num_obs)

        # Randomly distribute missing values by the features.
        num_mvs_per_feat = np.zeros(num_feat)
        while sum_mvs > 0:
            for i in range(num_feat):
                if num_mvs_per_feat[i] < max_mvs_feat:
                    num_mvs_it = np.random.randint(0, min(max_mvs_feat - num_mvs_per_feat[i] + 1, sum_mvs + 1))
                    sum_mvs -= num_mvs_it
                    num_mvs_per_feat[i] += num_mvs_it
                    if sum_mvs == 0:
                        break

        np.random.shuffle(num_mvs_per_feat)

        # Amputate the values.
        for i, col in enumerate(dataset.columns.values):
            num_mv = round(num_mvs_per_feat[i])
            num_mv = num_mv if num_mv > 0 else 0
            idx_nan_rows = np.random.choice(num_obs, num_mv, replace=False)
            dataset.loc[dataset.index[idx_nan_rows], col] = np.nan
        '''
        sum_mvs_cols = 0
        for c in dataset.columns.values:
            num_mvs = pd.isnull(dataset.loc[:, c]).values.astype(int).sum()
            print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(dataset)) * 100)}%).")
            sum_mvs_cols += num_mvs

        print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(dataset) * len(dataset.columns.values))) * 100)}%.")
        '''
        return dataset


# Example with the Iris dataset.
if __name__ == '__main__':
    iris_ds = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    iris_ds = TrueMultivariateMCAR.amputate(iris_ds, missing_rate=0.4)
    print(iris_ds, end="\n\n")

    sum_mvs_cols = 0
    for c in iris_ds.columns.values:
        num_mvs = pd.isnull(iris_ds.loc[:, c]).values.astype(int).sum()
        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(iris_ds)) * 100)}%).")
        sum_mvs_cols += num_mvs

    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(iris_ds) * len(iris_ds.columns.values))) * 100)}%.")