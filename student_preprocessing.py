import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import glob
import numpy as np


np.set_printoptions(threshold=np.inf)
import random
import os
from collections import Counter



def main():
    files_to_prepocess = ["data/raw/student-mat.csv","data/raw/student-por.csv"]

    for f in files_to_prepocess:
        df = pandas.read_csv(f)

        #print(df)
        print(df.columns)

        #df['G1'] = [1 if num >=10 else 0 for num in df['G1']]
        #df['G2'] = [1 if num >=10 else 0 for num in df['G2']]
        #df['G3'] = [1 if num >=10 else 0 for num in df['G3']]

        df['age'] = ["adult" if num >=18 else "teenager" for num in df['age']]

        print (df)
        df.to_csv(f, index=False)
    #i=1+1
    #df.to_csv("/home/xavier/tempDataset/credit-dataset.csv", index=False)

main()