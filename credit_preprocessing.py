import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import glob
import numpy as np


np.set_printoptions(threshold=np.inf)
import random
import os
from collections import Counter



def main():
    lst1 = glob.glob("/home/xavier/tempDataset/*")
    KDD_collumns = ['age', 'class of worker', 'industry code', 'occupation code',
                                        'education', 'wage per hour', 'enrolled in edu inst last wk', 'marital status', 'major industry code',
                                        'major occupation code', 'race', 'hispanic Origin', 'sex', 'member of a labor union',
                                        'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses',
                                        'divdends from stocks', 'federal income tax liability', 'region of previous residence',
                                        'state of previous residence', 'detailed household and family stat', 'detailed household summary in household','instance weight',
                                        'migration code-change in msa','migration code-change in reg','migration code-move within reg','live in this house 1 year ago',
                                        'migration prev res in sunbelt','num persons worked for employer','family members under 18','country of birth father',
                                        'country of birth mother','country of birth self','citizenship','own business or self employed',
                                        "fill inc questionnaire for veteran's admin",'veterans benefits','weeks worked in year','year','income']

    df = pandas.read_excel("/home/xavier/tempDataset/default of credit card clients.xls", skiprows=1,usecols=[i+1 for i in range(26)])
    #df = pandas.read_csv("/home/xavier/tempDataset/default of credit card clients.xls")
    df.columns = [collum.lower() for collum in df.columns]
    print(df)
    print(df.columns)
    i=1+1
    df.to_csv("/home/xavier/tempDataset/credit-dataset.csv", index=False)
    #df2 = pandas.read_csv("/home/xavier/tempDataset/census-income.test", names=KDD_collumns)
    #frames=[df,df2]
    #finalDf = pandas.concat(frames)

    #i=1+2
    #df.to_csv("/home/xavier/tempDataset/kdd-income.csv", index=False)

main()