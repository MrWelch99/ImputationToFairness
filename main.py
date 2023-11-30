import pandas as pd
import glob
import os
import csv

_RAW_DATA_PATH = "results/Processed/raw_data_full/"
_FILE_PATH_STEP_1 = "results/Processed/step_1-incomplete_imputation_full_results.csv"
_FILE_PATH_STEP_2 = "results/Processed/step_2-agg_oversample_then_imputation_MM_results.csv"
_FILE_PATH_STEP_3 = "results/Processed/step_3-diff_oversample_then_imputation_MM_results.csv"

_METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "DI BI", "Equal Opportunity",
            "Equal Mis-Opportunity", "CV", "Calibration+", "Calibration-", "Generalized Entropy Index"]

_MEAN = "Mean"
_STD = "STD"
_FILTER_F1_GT_50 = True

# Possible Factors: Dataset, Sensitive, Mechanism, MissingRate, Imputation, Algorithm
_FACTORS = ["Dataset","Sensitive","Imputation","MissingRate","Algorithm"]  # MissingRate must be included...
_MISSING_RATES = [0,0.05, 0.1, 0.2, 0.4]

#Put all of the data into csv
def raw_data_to_agg_csv():
    raw_files = sorted(glob.glob(f"{_RAW_DATA_PATH}/*"))
    with open(_FILE_PATH_STEP_1, 'w', newline='') as csvfile:
        writer = None
        for rf in raw_files:
            name = os.path.basename(rf)
            
            name = name.replace("_oversample_then_imputation_numerical.csv", "")
            name_parts = name.split("_")

            if name_parts[2]=="true":
                name_parts[2] = name_parts[2]+"_"+name_parts[3]
                name_parts[3] = name_parts[4]
            elif name_parts[2]=="mcar":
                continue

            '''
            if name_parts[2]=="true_mcar":
                continue
            '''
            
            dataset, sensitive_att, mechanism, imputation = name_parts[0], name_parts[1], name_parts[2], name_parts[3]
            df = pd.read_csv(rf)
            for _, row_df in df.iterrows():
                row = {
                    "Dataset": dataset,
                    "Sensitive": sensitive_att,
                    "Mechanism": mechanism,
                    "MissingRate": row_df["Missing rate"],
                    "Imputation": imputation,
                    "Algorithm": row_df["Algorithm"],
                    "Run": row_df["Run"]
                }

                for m in _METRICS:
                    row[f"{m}"] = row_df[f"{m}"]

                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=[*row])
                    writer.writeheader()

                writer.writerow(row)
            csvfile.flush()

#Group intances by the Factors
def agg_by_factors():
    ds = pd.read_csv(_FILE_PATH_STEP_1)
    if _FILTER_F1_GT_50:
        ds = ds[ds['F1-Score'] > 0.5]

    ds_cp = ds.copy(deep=True)
    #Calc mean of db grouped by chosen parameterers
    ds_mean = ds_cp.groupby(_FACTORS).mean()
    #Calc std of db grouped by chosen parameterers
    ds_std = ds_cp.groupby(_FACTORS).std()
    f_vals = list(ds_mean.index)
    with open(_FILE_PATH_STEP_2, 'w', newline='') as csvfile:
        writer = None
        for iv in f_vals:
            row = {}

            for i, f in enumerate(_FACTORS):
                if isinstance(iv,tuple):
                    row[f] = iv[i]
                else:
                    row[f] = iv

            for m in _METRICS:
                #Add Mean and STD of means to row 
                row[f"{_MEAN} {m}"] = ds_mean.loc[iv, f'{m}']
                row[f"{_STD} {m}"] = ds_std.loc[iv, f'{m}']

            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=[*row])
                writer.writeheader()

            writer.writerow(row)
        csvfile.flush()

#Calc Missing Rate
def calc_missing_rate_diffs():
    ds = pd.read_csv(_FILE_PATH_STEP_2)
    copy_factors = _FACTORS.copy()
    copy_factors.remove("MissingRate")
    if len(copy_factors)!=0:
        ds_grouped = dict(tuple(ds.groupby(copy_factors)))
    else:
        ds_grouped = {"key":ds}
    with open(_FILE_PATH_STEP_3, 'w', newline='') as csvfile:
        writer = None
        for key, ds_iv in ds_grouped.items():
            row_mr_0 = ds_iv[ds_iv["MissingRate"] == 0]
            for mr in _MISSING_RATES:
                row_mr = ds_iv[ds_iv["MissingRate"] == mr]
                if not row_mr.empty:
                    new_row = {"MissingRate": mr}

                    for f in copy_factors:
                        new_row[f] = row_mr[f].values[0]

                    for itr in range(len(_METRICS)):
                        m = _METRICS[itr]
                        if itr<=3:
                            new_row[f"{_MEAN} {m}"] = row_mr[f"{_MEAN} {m}"].values[0] - row_mr_0[f"{_MEAN} {m}"].values[0] 
                        else:
                            dist = abs(1-row_mr[f"{_MEAN} {m}"].values[0])
                            dist_0 = abs(1-row_mr_0[f"{_MEAN} {m}"].values[0])
                            new_row[f"{_MEAN} {m}"] = dist_0 - dist 
                        
                        new_row[f"{_STD} {m}"] = row_mr[f"{_STD} {m}"].values[0] - row_mr_0[f"{_STD} {m}"].values[0]

                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=[*new_row])
                        writer.writeheader()

                    writer.writerow(new_row)
            
            writer.writerow({})

            csvfile.flush()


if __name__ == '__main__':
    raw_data_to_agg_csv()
    #agg_by_factors()
    #calc_missing_rate_diffs()
