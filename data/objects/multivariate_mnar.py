import numpy as np
import pandas as pd
from typing import List


class MultivariateMNAR:

    @staticmethod
    def get_ordered_indices(col, dataset, ascending):
        x_f = dataset.loc[:, col].values
        tie_breaker = np.random.random(x_f.size)
        if ascending:
            return np.lexsort((tie_breaker, x_f))
        else:
            return np.lexsort((tie_breaker, x_f))[::-1]

    @staticmethod
    def amputate(dataset: pd.DataFrame, missing_rate: float,
                 ascending: bool = True, depend_on_external: List = None) -> pd.DataFrame:

        if depend_on_external is None:
            depend_on_external = []

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
        hidden_f = np.random.normal(size=num_obs)
        tie_breaker = np.random.random(hidden_f.size)

        # Amputate the values.
        for i, col in enumerate(dataset.columns.values):
            num_mv = round(num_mvs_per_feat[i])
            num_mv = num_mv if num_mv > 0 else 0

            if col in depend_on_external:
                start_n = end_n = int(num_mv / 2)
                if num_mv % 2 == 1:
                    end_n += 1

                indices_start = np.lexsort((tie_breaker, hidden_f))[:start_n]
                indices_end = np.lexsort((tie_breaker, -hidden_f))[:end_n]

                dataset.loc[dataset.index[indices_start], col] = np.nan
                dataset.loc[dataset.index[indices_end], col] = np.nan
            else:
                ordered_indices = MultivariateMNAR.get_ordered_indices(col, dataset, ascending)
                dataset.loc[dataset.index[ordered_indices[:num_mv]], col] = np.nan

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
    iris_ds = MultivariateMNAR.amputate(iris_ds, missing_rate=0.4, ascending=True, depend_on_external=["species"])
    print(iris_ds, end="\n\n")

    sum_mvs_cols = 0
    for c in iris_ds.columns.values:
        num_mvs = pd.isnull(iris_ds.loc[:, c]).values.astype(int).sum()
        print(f"Feature '{c}' has {num_mvs} missing values ({round((num_mvs / len(iris_ds)) * 100)}%).")
        sum_mvs_cols += num_mvs

    print(f"\nGlobal missing rate is {round((sum_mvs_cols / (len(iris_ds) * len(iris_ds.columns.values))) * 100)}%.")
